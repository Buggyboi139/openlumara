import os
import yaml
import copy
import core
import modules
import user_modules
import channels
import pkgutil
import hashlib
import json
import inspect

config = None
_registry_cache = None

SCHEMA_CACHE_FILE = ".module_cache.json"

default_config = {
    "core": {
        "data_folder": "data",
        "auto_resume_chats": True,
        "cmd_prefix": "/"
    },
    "api": {
        "url": "http://localhost:5001/v1",
        "key": "KEY_HERE",
        "max_context": 8192,
        "max_output_tokens": 8192,
        "max_messages": 200,
        "check_developer_role_support": False,
        "custom_fields": {}
    },
    "model": {
        "name": "",
        "temperature": 0.7,
        "enable_thinking": True,
        "keep_reasoning_in_context": True,
        "reasoning_effort": None,
        "use_tools": True
    },
    "tools": {
        "legacy_auto_tools": True,
        "max_recursion": 4,
        "max_calls_per_turn": 12,
        "max_repeated_calls": 2,
        "timeout_seconds": 30
    },
    "channels": {
        "enabled": [],
        "disabled": [],
        "settings": {}
    },
    "modules": {
        "enabled": [],
        "disabled": [],
        "settings": {}
    },
    "user_modules": {
        "path": "user_modules",
        "enabled": [],
        "disabled": [],
        "settings": {}
    }
}

DEFAULT_MODULES = (
    "tutorial",
    "docs",
    "identity",
    "writing_style",
    "models",
    "channel",
    "modules",
    "chats",
    "context",
    "memory",
    "notes",
    "lists",
    "scheduler",
    "calendar",
    "calculator",
    "token_threshold",
    "time",
    "web_search",
    "web_reader"
)

DEFAULT_CHANNELS = ["cli", "webui"]

class ConfigManager:
    def __init__(self, config, base_path=None):
        self.root_config = config
        self.base_path = base_path or []

    def get(self, *args, **kwargs):
        """Shorthand for accessing nested config values.
        Usage: config.get("api", "url") or config.get("api", "url", default_value)
        """
        default = kwargs.get("default", None)
        if not args:
            return default

        keys = list(args)
        # If the last argument is not a string, or is empty, treat it as an explicit default
        if keys and not isinstance(keys[-1], str) or not keys[-1]:
            default = keys.pop()

        # Start from the root config and traverse through the base path
        current = self.root_config
        for k in self.base_path:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        # Then traverse through the provided keys
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def to_dict(self):
        # Start from the root config and traverse through the base path
        current = self.root_config
        for k in self.base_path:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return {}

        return dict(current)

    def __getitem__(self, key):
        """Access items using bracket notation: config['key']"""
        current = self.root_config
        for k in self.base_path + [key]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                raise KeyError(key)
        return current

    def __setitem__(self, key, value):
        """Set items using bracket notation: config['key'] = value"""
        current = self.root_config
        for k in self.base_path:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        current[key] = value
        if hasattr(self.root_config, 'save'):
            self.root_config.save()

    def __contains__(self, key):
        """Check if key exists: 'key' in config"""
        current = self.root_config
        for k in self.base_path:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
        return isinstance(current, dict) and key in current

def _discover_available_names(package):
    """
    Discover module names from filesystem WITHOUT importing them.
    This allows the config to know what modules exist without loading them.
    """
    if not hasattr(package, '__path__'):
        return []
    return [modname for _, modname, _ in pkgutil.iter_modules(package.__path__)]

def _get_registry_data(enabled_channels=None, enabled_modules=None, enabled_user_modules=None):
    """
    Build registry data, importing ONLY enabled modules/channels.

    Available names are discovered via filesystem scanning.
    Instances are only created for enabled items.
    """
    global _registry_cache

    # Build cache key from enabled lists
    cache_key = (
        tuple(enabled_channels or []),
        tuple(enabled_modules or []),
        tuple(enabled_user_modules or [])
    )

    if _registry_cache is not None and _registry_cache.get('key') == cache_key:
        return _registry_cache['data']

    # Discover all available names from filesystem (no imports!)
    available_channels = _discover_available_names(channels)
    available_modules = _discover_available_names(modules)
    available_user_modules = _discover_available_names(user_modules)

    # Only import and instantiate ENABLED items
    chan_inst = list(core.modules.load(
        channels, core.channel.Channel, filter=enabled_channels
    )) if enabled_channels else []

    mod_inst = list(core.modules.load(
        modules, core.module.Module, filter=enabled_modules
    )) if enabled_modules else []

    user_mod_inst = list(core.modules.load(
        user_modules, core.module.Module, filter=enabled_user_modules
    )) if enabled_user_modules else []

    result = [
        {
            "section_key": "channels",
            "instances": chan_inst,
            "available_names": available_channels,
            "names": [core.modules.get_name(m) for m in chan_inst],
            "default_names": DEFAULT_CHANNELS
        },
        {
            "section_key": "modules",
            "instances": mod_inst,
            "available_names": available_modules,
            "names": [core.modules.get_name(m) for m in mod_inst],
            "default_names": DEFAULT_MODULES
        },
        {
            "section_key": "user_modules",
            "instances": user_mod_inst,
            "available_names": available_user_modules,
            "names": [core.modules.get_name(m) for m in user_mod_inst],
            "default_names": []
        }
    ]

    _registry_cache = {'key': cache_key, 'data': result}
    return result

def _inject_settings_into_dict(target_dict, instances, section_key):
    """Helper to build the schema by injecting class settings defaults."""
    section = target_dict.setdefault(section_key, {})
    settings = section.setdefault("settings", {})
    for inst in instances:
        name = core.modules.get_name(inst)
        defaults = getattr(inst, 'settings', {})
        if isinstance(defaults, dict) and defaults:
            # We inject the full dict (including descriptions) into the schema.
            # sync_config will later replace these dicts with flat values
            # if the user has provided them in the config file.
            settings[name] = defaults.copy()

def _get_module_schema_cache():
    """
    Returns a dictionary containing the cached schemas and checksums for all modules/channels.
    If the cache is missing or outdated, it performs a refresh.
    """
    cache_path = os.path.abspath(os.path.join(core.get_path(), SCHEMA_CACHE_FILE))
    cache = {"channels": {}, "modules": {}, "user_modules": {}}

    # Load existing cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            core.log_error("error while loading module cache", e)
    else:
        core.log("core", f"creating module cache at {cache_path}")

    package_map = {
        "channels": (channels, core.channel.Channel),
        "modules": (modules, core.module.Module),
        "user_modules": (user_modules, core.module.Module)
    }

    sections_to_refresh = set()

    # 1. Check for deletions or changes in existing cache
    for section_key, (package, _) in package_map.items():
        available_names = _discover_available_names(package)

        for name in list(cache[section_key].keys()):
            if name not in available_names:
                del cache[section_key][name]
                sections_to_refresh.add(section_key)
                continue

            # Find the file path to check checksum
            found_file = None
            for sub_path in package.__path__:
                # Try module.py
                f1 = os.path.join(sub_path, f"{name}.py")
                if os.path.exists(f1):
                    found_file = f1
                    break
                # Try module/__init__.py
                f2 = os.path.join(sub_path, name, "__init__.py")
                if os.path.exists(f2):
                    found_file = f2
                    break

            if found_file:
                if cache[section_key][name].get("checksum") != _get_file_checksum(found_file):
                    sections_to_refresh.add(section_key)
            else:
                sections_to_refresh.add(section_key)

        # 2. Check for new modules
        if section_key not in sections_to_refresh:
            for name in available_names:
                if name not in cache[section_key]:
                    sections_to_refresh.add(section_key)
                    break

    # 3. Refresh cache if needed
    if sections_to_refresh:
        for section_key in sections_to_refresh:
            package, base_class = package_map[section_key]
            try:
                # skip reloading modules because we just want the data
                classes = core.modules.load(package, base_class, reload=False)

                for cls in classes:
                    name = core.modules.get_name(cls)
                    settings = getattr(cls, 'settings', {})

                    # Capture docstring and the unsafe class attribute
                    docstring = inspect.getdoc(cls) or ""
                    unsafe = getattr(cls, 'unsafe', False)

                    module = inspect.getmodule(cls)
                    checksum = ""
                    if module and hasattr(module, '__file__') and module.__file__:
                        py_file = module.__file__.replace('.pyc', '')
                        checksum = _get_file_checksum(py_file) if os.path.exists(py_file) else _get_file_checksum(module.__file__)

                    cache[section_key][name] = {
                        "schema": settings,
                        "checksum": checksum,
                        "metadata": {
                            "docstring": docstring,
                            "unsafe": unsafe  # Added to cache
                        }
                    }

            except Exception as e:
                core.log_error(f"Failed to refresh cache for {section_key}", e)

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            core.log_error("failed to save module cache", e)

    return cache

def _get_file_checksum(filepath):
    """Calculate MD5 checksum of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""

def _sync_section(config_data, section_key, available_names, enabled_names=None, default_names=None):
    enabled_names = enabled_names or []
    default_names = default_names or []
    section = config_data.setdefault(section_key, {})

    section.setdefault("enabled", [])
    section.setdefault("disabled", [])
    section.setdefault("settings", {})

    disabled = set(section.get("disabled", []))
    current_enabled = set(section.get("enabled", []))
    defaults = set(default_names)

    if current_enabled:
        desired_enabled = current_enabled - disabled
    else:
        desired_enabled = defaults - disabled

    # only include names that actually exist
    desired_enabled = [name for name in sorted(desired_enabled) if name in available_names]

    section["enabled"] = desired_enabled
    section["disabled"] = sorted(disabled)

    # preserve settings for all available items + any existing custom keys
    settings = section.setdefault("settings", {})
    for name in available_names:
        settings.setdefault(name, {})

def sync_config(config_data):
    # Bootstrap minimal structure first
    for section_key in ["core", "api", "model", "channels", "modules", "user_modules"]:
        if section_key not in config_data:
            config_data[section_key] = copy.deepcopy(default_config.get(section_key, {}))

    # Merge default scalar values into existing config without clobbering user data
    for section_key, defaults in default_config.items():
        if isinstance(defaults, dict):
            section = config_data.setdefault(section_key, {})
            for key, value in defaults.items():
                section.setdefault(key, copy.deepcopy(value))

    enabled_channels = config_data.get("channels", {}).get("enabled")
    enabled_modules = config_data.get("modules", {}).get("enabled")
    enabled_user_modules = config_data.get("user_modules", {}).get("enabled")

    registry_data = _get_registry_data(enabled_channels, enabled_modules, enabled_user_modules)

    for item in registry_data:
        _sync_section(
            config_data,
            item["section_key"],
            item["available_names"],
            enabled_names=item["names"],
            default_names=item["default_names"]
        )
        _inject_settings_into_dict(config_data, item["instances"], item["section_key"])

    return config_data
