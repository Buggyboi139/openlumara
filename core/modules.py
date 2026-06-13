import core
import re
import inspect
import sys
import subprocess
import ast

# modules that should have their prompts inserted even when tools are off
nonagentic = ("characters", "time")

def _extract_deps_from_file(file_path):
    """extract dependencies list from module file without importing it"""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == 'dependencies':
                                if isinstance(item.value, ast.List):
                                    return [
                                        elt.value for elt in item.value.elts
                                        if isinstance(elt, ast.Constant)
                                    ]
    except Exception as e:
        core.log("warning", f"could not parse dependencies from {file_path}: {e}")
    return []

def _install_deps(module_name, packages):
    """install pip packages"""
    if not packages:
        return
    core.log(module_name, f"installing dependencies: {packages}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        core.log_error("dependency install failed", e)

def _uninstall_deps(module_name, packages):
    """uninstall pip packages"""
    if not packages:
        return
    core.log(module_name, f"uninstalling dependencies: {packages}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "--quiet"] + packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        core.log(module_name, f"uninstalled: {packages}")
    except subprocess.CalledProcessError as e:
        core.log_error("dependency uninstall failed", e)

def _get_module_file_path(package, module_name):
    """get the file path for a module without importing it"""
    import importlib.util
    
    spec = importlib.util.find_spec(f"{package.__name__}.{module_name}")
    if spec and spec.origin:
        return spec.origin
    return None

def uninstall_module_deps(module_name, is_user_module=False):
    """uninstall dependencies for a disabled module"""
    import modules
    import user_modules
    
    package = user_modules if is_user_module else modules
    
    file_path = _get_module_file_path(package, module_name)
    if not file_path:
        core.log("warning", f"could not find file for module {module_name}")
        return
    
    deps = _extract_deps_from_file(file_path)
    if deps:
        _uninstall_deps(module_name, deps)
    else:
        core.log(module_name, "no dependencies to uninstall")

def load(package, base_class = None, filter: list = None, reload: bool = False):
    """
    loops through the specified package imported with `import whatever`, then checks inside those packages for any classes that derive from base_class, and return a tuple of those classes so we can use them as modules, channels etc

    this is what powers dynamic module/channel importing. we use it like so:
    import my_folder_with_classes as dynamic_folder
    self.load_modules(dynamic_folder, core.module.Module)
    """
    import importlib
    import pkgutil

    discovered = []

    if not hasattr(package, '__path__'):
        return ()

    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if filter and modname not in filter:
            # dont even import unloaded modules
            continue

        # get file path to the module
        module_file_path = _get_module_file_path(package, modname)

        # extract and install dependencies
        deps = _extract_deps_from_file(module_file_path)
        if deps:
            # check which are actually missing
            missing = []
            for dep in deps:
                dep_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                try:
                    __import__(dep_name)
                except ImportError:
                    missing.append(dep)
            if missing:
                _install_deps(modname, missing)

        try:
            # Import the module relative to the package
            module = importlib.import_module(f"{package.__name__}.{modname}")

            # if the reload flag is true, force a reload of the module code so that new changes are applied
            # NOTE: this is only intended to be used upon a total restart of openlumara.
            # it can mess things up severely if modules/channels are still loaded
            if reload:
                importlib.reload(module)

            for attr_name in dir(module):
                target_class = getattr(module, attr_name)

                # Ensure it is a class
                if not isinstance(target_class, type):
                    continue

                # Filter by base class if provided
                if base_class:
                    if target_class is base_class:
                        continue
                    if not issubclass(target_class, base_class):
                        continue

                # skip modules not in filter if filter is enabled
                if filter and core.modules.get_name(target_class) not in filter:
                    continue

                discovered.append(target_class)
        except core.exceptions.DependencyMissing as e:
            # silence these warnings for now
            # need a better way to deal with missing dependencies
            pass
        except Exception as e:
            # Catching Exception prevents the program from crashing on faulty modules.
            # We simply log the warning and continue to the next module.
            core.log_error(f"failed to load module {modname}", e)
            continue

    return tuple(discovered)

def get_name(obj):
    """converts a name like LifeOrganizer to `life_organizer`"""

    name = None
    if inspect.isclass(obj):
        name = obj.__name__
    else:
        name = obj.__class__.__name__

    re_snakecase = re.compile('(?!^)([A-Z]+)')
    name_snakecase = re.sub(re_snakecase, r'_\1', name).lower()

    return name_snakecase
