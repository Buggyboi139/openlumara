import core
import asyncio
import os
import re
import platform
import time
from datetime import datetime


class SystemProbe(core.module.Module):
    """
    Read-only Linux system diagnostics for AMD GPU systems.
    Reports CPU name, CPU temperature, AMD GPU names, GPU temperatures, GPU usage,
    memory usage, disk usage, uptime, load average, and top processes.
    This module does not use terminal access, subprocess, shell commands, file writes,
    network access, or user-supplied paths.
    """

    # 1. SETTINGS
    settings = {
        "process_limit": {
            "default": 10,
            "description": "Number of top processes to show."
        },
        "max_process_limit": {
            "default": 25,
            "description": "Hard maximum number of processes that can be shown."
        },
        "include_usernames": {
            "default": False,
            "description": "Include process owner usernames in process output."
        },
        "include_hostname": {
            "default": False,
            "description": "Include hostname in full status output."
        },
        "include_gpu_processes": {
            "default": True,
            "description": "Try to show processes with open AMD GPU device handles."
        },
        "disk_path": {
            "default": "/",
            "description": "Disk path to report usage for. This is config-only, not chat input."
        }
    }

    # 2. INITIALIZATION
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sys_drm_path = "/sys/class/drm"
        self.dev_dri_path = "/dev/dri"

    # 3. STARTUP TASKS
    async def on_ready(self):
        pass

    # INTERNAL HELPERS
    def _cfg(self, key, default=None):
        value = self.config.get(key, default=default)
        return default if value is None else value

    def _bool_cfg(self, key, default=False):
        return bool(self._cfg(key, default=default))

    def _int_cfg(self, key, default):
        try:
            return int(self._cfg(key, default=default))
        except Exception:
            return int(default)

    def _clip(self, value, limit=140):
        text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
        text = re.sub(r"\s+", " ", text)
        return text[:limit].rstrip() + "..." if len(text) > limit else text

    def _read_text(self, path):
        try:
            path = os.path.abspath(path)

            if not path.startswith("/sys/") and not path.startswith("/proc/"):
                return None

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            return None

    def _read_int(self, path):
        text = self._read_text(path)

        if text is None or text == "":
            return None

        try:
            return int(text.split()[0])
        except Exception:
            return None

    def _bytes_to_gb(self, value):
        try:
            return round(float(value) / (1024 ** 3), 2)
        except Exception:
            return None

    def _bytes_to_mib(self, value):
        try:
            return round(float(value) / (1024 ** 2), 1)
        except Exception:
            return None

    def _fmt_value(self, value, suffix=""):
        if value is None:
            return "unavailable"
        return f"{value}{suffix}"

    def _get_cpu_name(self):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return self._clip(line.split(":", 1)[1].strip(), 160)
        except Exception:
            pass

        name = platform.processor()
        return self._clip(name, 160) if name else "Unknown CPU"

    def _get_cpu_temp(self):
        try:
            import psutil

            temps = psutil.sensors_temperatures(fahrenheit=False)

            if not temps:
                return None

            preferred = [
                "k10temp",
                "coretemp",
                "zenpower",
                "cpu_thermal",
                "acpitz"
            ]

            def valid_temp(value):
                try:
                    value = float(value)
                    return value if 0 < value < 130 else None
                except Exception:
                    return None

            for key in preferred:
                entries = temps.get(key)

                if not entries:
                    continue

                readings = []

                for item in entries:
                    temp = valid_temp(getattr(item, "current", None))

                    if temp is not None:
                        readings.append(temp)

                if readings:
                    return round(max(readings), 1)

            fallback = []
            skip_words = ["gpu", "amdgpu", "nvme", "drive", "disk", "ssd", "hdd"]

            for key, entries in temps.items():
                key_lower = str(key).lower()

                if any(word in key_lower for word in skip_words):
                    continue

                for item in entries:
                    temp = valid_temp(getattr(item, "current", None))

                    if temp is not None:
                        fallback.append(temp)

            return round(max(fallback), 1) if fallback else None

        except Exception:
            return None

    def _safe_drm_cards(self):
        try:
            entries = os.listdir(self.sys_drm_path)
        except Exception:
            return []

        cards = []

        for entry in entries:
            if not entry.startswith("card"):
                continue

            suffix = entry[4:]

            if not suffix.isdigit():
                continue

            card_path = os.path.abspath(os.path.join(self.sys_drm_path, entry))
            device_path = os.path.abspath(os.path.join(card_path, "device"))

            if not card_path.startswith(self.sys_drm_path):
                continue

            vendor = self._read_text(os.path.join(device_path, "vendor"))
            uevent = self._read_text(os.path.join(device_path, "uevent")) or ""

            is_amd = vendor and vendor.lower() == "0x1002"
            uses_amdgpu = "DRIVER=amdgpu" in uevent

            if is_amd or uses_amdgpu:
                cards.append({
                    "card": entry,
                    "card_index": int(suffix),
                    "card_path": card_path,
                    "device_path": device_path
                })

        cards.sort(key=lambda item: item["card_index"])
        return cards

    def _parse_uevent(self, device_path):
        raw = self._read_text(os.path.join(device_path, "uevent")) or ""
        parsed = {}

        for line in raw.splitlines():
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()

        return parsed

    def _get_gpu_name(self, card, device_path):
        product_name = self._read_text(os.path.join(device_path, "product_name"))

        if product_name:
            return self._clip(product_name, 160)

        product_number = self._read_text(os.path.join(device_path, "product_number"))

        if product_number:
            return self._clip(product_number, 160)

        uevent = self._parse_uevent(device_path)
        pci_id = uevent.get("PCI_ID")
        slot = uevent.get("PCI_SLOT_NAME")

        if pci_id and slot:
            return f"AMD GPU {card} ({pci_id}, {slot})"

        if pci_id:
            return f"AMD GPU {card} ({pci_id})"

        return f"AMD GPU {card}"

    def _get_hwmon_dirs(self, device_path):
        hwmon_root = os.path.join(device_path, "hwmon")

        try:
            names = os.listdir(hwmon_root)
        except Exception:
            return []

        dirs = []

        for name in names:
            if not name.startswith("hwmon"):
                continue

            path = os.path.abspath(os.path.join(hwmon_root, name))

            if path.startswith(os.path.abspath(hwmon_root)):
                dirs.append(path)

        return dirs

    def _get_gpu_temps(self, device_path):
        temps = []

        for hwmon in self._get_hwmon_dirs(device_path):
            hwmon_name = self._read_text(os.path.join(hwmon, "name")) or ""

            try:
                files = os.listdir(hwmon)
            except Exception:
                continue

            for file_name in files:
                if not file_name.startswith("temp") or not file_name.endswith("_input"):
                    continue

                sensor_id = file_name.replace("temp", "").replace("_input", "")
                input_path = os.path.join(hwmon, file_name)
                label_path = os.path.join(hwmon, f"temp{sensor_id}_label")

                raw = self._read_int(input_path)

                if raw is None:
                    continue

                temp_c = round(raw / 1000, 1) if raw > 1000 else round(float(raw), 1)

                if temp_c <= 0 or temp_c >= 130:
                    continue

                label = self._read_text(label_path) or hwmon_name or f"temp{sensor_id}"

                temps.append({
                    "label": self._clip(label, 60),
                    "temp_c": temp_c
                })

        priority = ["edge", "junction", "hotspot", "mem", "memory"]
        primary = None

        for wanted in priority:
            for item in temps:
                if wanted in item["label"].lower():
                    primary = item
                    break

            if primary:
                break

        if primary is None and temps:
            primary = max(temps, key=lambda item: item["temp_c"])

        return {
            "primary": primary,
            "all": temps
        }

    def _get_gpu_power_watts(self, device_path):
        candidates = []

        for hwmon in self._get_hwmon_dirs(device_path):
            for name in ["power1_average", "power1_input"]:
                value = self._read_int(os.path.join(hwmon, name))

                if value is not None and value > 0:
                    candidates.append(value)

        if not candidates:
            return None

        value = max(candidates)

        if value > 100000:
            return round(value / 1000000, 1)

        if value > 1000:
            return round(value / 1000, 1)

        return round(float(value), 1)

    def _get_gpu_device_nodes(self, card_info):
        nodes = []
        card = card_info["card"]
        card_node = os.path.join(self.dev_dri_path, card)

        if os.path.exists(card_node):
            nodes.append(card_node)

        drm_dir = os.path.join(card_info["device_path"], "drm")

        try:
            entries = os.listdir(drm_dir)
        except Exception:
            return nodes

        for entry in entries:
            if entry.startswith("renderD"):
                node = os.path.join(self.dev_dri_path, entry)

                if os.path.exists(node):
                    nodes.append(node)

        return sorted(set(nodes))

    def _get_amd_gpus(self):
        gpus = []

        for card_info in self._safe_drm_cards():
            card = card_info["card"]
            card_index = card_info["card_index"]
            device_path = card_info["device_path"]

            temps = self._get_gpu_temps(device_path)

            gpu_busy = self._read_int(os.path.join(device_path, "gpu_busy_percent"))
            mem_busy = self._read_int(os.path.join(device_path, "mem_busy_percent"))

            vram_total = self._read_int(os.path.join(device_path, "mem_info_vram_total"))
            vram_used = self._read_int(os.path.join(device_path, "mem_info_vram_used"))

            gtt_total = self._read_int(os.path.join(device_path, "mem_info_gtt_total"))
            gtt_used = self._read_int(os.path.join(device_path, "mem_info_gtt_used"))

            gpus.append({
                "card": card,
                "index": card_index,
                "name": self._get_gpu_name(card, device_path),
                "temp_c": temps["primary"]["temp_c"] if temps["primary"] else None,
                "temp_label": temps["primary"]["label"] if temps["primary"] else None,
                "all_temps": temps["all"],
                "gpu_util_percent": gpu_busy,
                "memory_util_percent": mem_busy,
                "vram_used_mib": self._bytes_to_mib(vram_used),
                "vram_total_mib": self._bytes_to_mib(vram_total),
                "gtt_used_mib": self._bytes_to_mib(gtt_used),
                "gtt_total_mib": self._bytes_to_mib(gtt_total),
                "power_w": self._get_gpu_power_watts(device_path),
                "device_nodes": self._get_gpu_device_nodes(card_info)
            })

        return gpus

    def _get_gpu_processes(self, gpus):
        if not self._bool_cfg("include_gpu_processes", True):
            return []

        try:
            import psutil
        except Exception:
            return []

        node_to_gpu = {}

        for gpu in gpus:
            for node in gpu.get("device_nodes", []):
                node_to_gpu[node] = gpu["card"]

        if not node_to_gpu:
            return []

        output = []
        seen = set()

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                pid = int(proc.info["pid"])
                fd_dir = f"/proc/{pid}/fd"

                try:
                    fds = os.listdir(fd_dir)
                except Exception:
                    continue

                for fd in fds:
                    fd_path = os.path.join(fd_dir, fd)

                    try:
                        target = os.readlink(fd_path)
                    except Exception:
                        continue

                    gpu_card = node_to_gpu.get(target)

                    if not gpu_card:
                        continue

                    key = (gpu_card, pid)

                    if key in seen:
                        continue

                    seen.add(key)

                    output.append({
                        "gpu": gpu_card,
                        "pid": pid,
                        "name": self._clip(proc.info.get("name") or "unknown", 100)
                    })

            except Exception:
                continue

        output.sort(key=lambda item: (item["gpu"], item["pid"]))
        return output[:50]

    def _get_memory(self):
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "ram_total_gb": self._bytes_to_gb(vm.total),
            "ram_used_gb": self._bytes_to_gb(vm.used),
            "ram_percent": vm.percent,
            "swap_total_gb": self._bytes_to_gb(swap.total),
            "swap_used_gb": self._bytes_to_gb(swap.used),
            "swap_percent": swap.percent
        }

    def _get_disk(self):
        import psutil

        configured = str(self._cfg("disk_path", "/") or "/")

        if not configured.startswith("/"):
            configured = "/"

        try:
            usage = psutil.disk_usage(configured)
            path = configured
        except Exception:
            usage = psutil.disk_usage("/")
            path = "/"

        return {
            "path": path,
            "total_gb": self._bytes_to_gb(usage.total),
            "used_gb": self._bytes_to_gb(usage.used),
            "free_gb": self._bytes_to_gb(usage.free),
            "percent": usage.percent
        }

    def _get_top_processes(self):
        import psutil

        limit = self._int_cfg("process_limit", 10)
        max_limit = self._int_cfg("max_process_limit", 25)
        limit = max(1, min(limit, max_limit))

        processes = []

        for proc in psutil.process_iter(["pid", "name", "memory_percent", "username"]):
            try:
                proc.cpu_percent(interval=None)
            except Exception:
                pass

        time.sleep(0.25)

        include_usernames = self._bool_cfg("include_usernames", False)

        for proc in psutil.process_iter(["pid", "name", "memory_percent", "username"]):
            try:
                info = proc.info

                item = {
                    "pid": int(info.get("pid")),
                    "name": self._clip(info.get("name") or "unknown", 100),
                    "cpu_percent": round(proc.cpu_percent(interval=None), 1),
                    "memory_percent": round(float(info.get("memory_percent") or 0), 1)
                }

                if include_usernames:
                    item["user"] = self._clip(info.get("username") or "unknown", 80)

                processes.append(item)

            except Exception:
                continue

        processes.sort(
            key=lambda item: (item["cpu_percent"], item["memory_percent"]),
            reverse=True
        )

        return processes[:limit]

    def _get_load(self):
        try:
            one, five, fifteen = os.getloadavg()

            return {
                "1m": round(one, 2),
                "5m": round(five, 2),
                "15m": round(fifteen, 2)
            }
        except Exception:
            return {
                "1m": None,
                "5m": None,
                "15m": None
            }

    def _get_uptime(self):
        import psutil

        boot_time = psutil.boot_time()
        boot = datetime.fromtimestamp(boot_time)
        seconds = int(time.time() - boot_time)

        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60

        return {
            "boot_time": boot.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime": f"{days}d {hours}h {minutes}m"
        }

    def _snapshot_sync(self):
        import psutil

        gpus = self._get_amd_gpus()

        return {
            "hostname": platform.node() if self._bool_cfg("include_hostname", False) else None,
            "os": platform.platform(),
            "cpu": {
                "name": self._get_cpu_name(),
                "temp_c": self._get_cpu_temp(),
                "usage_percent": psutil.cpu_percent(interval=0.25),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True)
            },
            "gpus": gpus,
            "gpu_processes": self._get_gpu_processes(gpus),
            "memory": self._get_memory(),
            "disk": self._get_disk(),
            "load": self._get_load(),
            "uptime": self._get_uptime(),
            "processes": self._get_top_processes()
        }

    def _format_temps(self, snapshot):
        lines = []
        cpu = snapshot["cpu"]
        gpus = snapshot["gpus"]

        lines.append("Temperatures")
        lines.append("")
        lines.append(f"CPU: {cpu['name']}")
        lines.append(f"Temp: {self._fmt_value(cpu['temp_c'], '°C')}")
        lines.append("")
        lines.append("AMD GPUs")

        if not gpus:
            lines.append("No AMD GPUs found through /sys/class/drm.")
        else:
            for gpu in gpus:
                label = f" ({gpu['temp_label']})" if gpu["temp_label"] else ""

                lines.append(f"{gpu['card']}: {gpu['name']}")
                lines.append(f"Temp{label}: {self._fmt_value(gpu['temp_c'], '°C')}")

                extra = [
                    item for item in gpu.get("all_temps", [])
                    if item["temp_c"] != gpu["temp_c"] or item["label"] != gpu["temp_label"]
                ]

                for item in extra:
                    lines.append(f"{item['label']}: {item['temp_c']}°C")

        return "\n".join(lines)

    def _format_gpu(self, snapshot):
        lines = []
        gpus = snapshot["gpus"]

        lines.append("AMD GPU Status")
        lines.append("")

        if not gpus:
            lines.append("No AMD GPUs found through /sys/class/drm.")
            return "\n".join(lines)

        for gpu in gpus:
            lines.append(f"{gpu['card']}: {gpu['name']}")
            lines.append(f"Temp: {self._fmt_value(gpu['temp_c'], '°C')}")
            lines.append(f"GPU Usage: {self._fmt_value(gpu['gpu_util_percent'], '%')}")
            lines.append(f"Memory Usage: {self._fmt_value(gpu['memory_util_percent'], '%')}")
            lines.append(f"VRAM: {self._fmt_value(gpu['vram_used_mib'], ' MiB')} / {self._fmt_value(gpu['vram_total_mib'], ' MiB')}")
            lines.append(f"GTT: {self._fmt_value(gpu['gtt_used_mib'], ' MiB')} / {self._fmt_value(gpu['gtt_total_mib'], ' MiB')}")
            lines.append(f"Power: {self._fmt_value(gpu['power_w'], ' W')}")

            if gpu.get("all_temps"):
                temp_parts = []

                for item in gpu["all_temps"]:
                    temp_parts.append(f"{item['label']} {item['temp_c']}°C")

                lines.append(f"Sensors: {', '.join(temp_parts)}")

            lines.append("")

        lines.append("GPU Processes")

        if not snapshot["gpu_processes"]:
            lines.append("No GPU process handles found or permission denied.")
        else:
            for proc in snapshot["gpu_processes"]:
                lines.append(f"{proc['gpu']} PID {proc['pid']}: {proc['name']}")

        return "\n".join(lines)

    def _format_processes(self, snapshot):
        lines = []
        lines.append("Top Processes")
        lines.append("")

        for proc in snapshot["processes"]:
            base = f"PID {proc['pid']}: {proc['name']} CPU {proc['cpu_percent']}% / MEM {proc['memory_percent']}%"

            if "user" in proc:
                base += f" / USER {proc['user']}"

            lines.append(base)

        return "\n".join(lines)

    def _format_memory(self, snapshot):
        mem = snapshot["memory"]

        return "\n".join([
            "Memory",
            "",
            f"RAM: {mem['ram_used_gb']} GB / {mem['ram_total_gb']} GB ({mem['ram_percent']}%)",
            f"Swap: {mem['swap_used_gb']} GB / {mem['swap_total_gb']} GB ({mem['swap_percent']}%)"
        ])

    def _format_disk(self, snapshot):
        disk = snapshot["disk"]

        return "\n".join([
            "Disk",
            "",
            f"Path: {disk['path']}",
            f"Used: {disk['used_gb']} GB / {disk['total_gb']} GB ({disk['percent']}%)",
            f"Free: {disk['free_gb']} GB"
        ])

    def _format_load(self, snapshot):
        load = snapshot["load"]
        uptime = snapshot["uptime"]

        return "\n".join([
            "Load and Uptime",
            "",
            f"Load 1m: {load['1m']}",
            f"Load 5m: {load['5m']}",
            f"Load 15m: {load['15m']}",
            f"Booted: {uptime['boot_time']}",
            f"Uptime: {uptime['uptime']}"
        ])

    def _format_status(self, snapshot):
        lines = []

        lines.append("System Status")
        lines.append("")

        if snapshot["hostname"]:
            lines.append(f"Host: {snapshot['hostname']}")

        lines.append(f"OS: {snapshot['os']}")
        lines.append("")
        lines.append(self._format_temps(snapshot))
        lines.append("")
        lines.append(self._format_gpu(snapshot))
        lines.append("")
        lines.append(self._format_memory(snapshot))
        lines.append("")
        lines.append(self._format_disk(snapshot))
        lines.append("")
        lines.append(self._format_load(snapshot))
        lines.append("")
        lines.append(self._format_processes(snapshot))

        return "\n".join(lines)

    async def _snapshot(self):
        return await asyncio.to_thread(self._snapshot_sync)

    # 4. AI TOOLS
    async def get_system_status(self):
        """
        Get a full read-only Linux system diagnostic report.
        Use this when the user asks for overall system status, diagnostics,
        hardware status, CPU/GPU information, temperatures, memory, disk,
        uptime, load average, or top running processes.
        This tool does not run shell commands and does not modify the system.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_status(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def get_temperatures(self):
        """
        Get CPU and AMD GPU temperatures.
        Use this when the user asks about system temps, CPU temp, GPU temps,
        thermal status, overheating, or hardware temperature checks.
        This tool reads local sensor information only and does not modify the system.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_temps(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def get_gpu_status(self):
        """
        Get AMD GPU names, temperatures, GPU usage, memory usage, VRAM usage,
        GTT usage, power usage, and visible GPU process handles.
        Use this when the user asks about GPU usage, GPU load, VRAM, graphics cards,
        AMD GPU status, or what is using the GPUs.
        This tool does not run terminal commands or change GPU settings.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_gpu(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def get_processes(self):
        """
        Get the top running processes sorted by CPU and memory usage.
        Use this when the user asks what processes are using system resources.
        This tool returns process IDs, process names, CPU percentage, and memory percentage.
        It does not return command-line arguments and cannot kill or modify processes.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_processes(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def get_memory_status(self):
        """
        Get RAM and swap usage.
        Use this when the user asks about memory usage, RAM pressure, swap usage,
        or basic memory diagnostics.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_memory(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def get_disk_status(self):
        """
        Get disk usage for the configured disk path.
        Use this when the user asks about disk usage, free space, or storage status.
        The path comes only from module config, not from chat input.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_disk(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def get_load_status(self):
        """
        Get Linux load average and system uptime.
        Use this when the user asks about system load, uptime, boot time,
        or whether the machine is under heavy CPU scheduling pressure.
        """
        try:
            snapshot = await self._snapshot()
            return self._format_load(snapshot)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    # 5. USER COMMANDS
    @core.module.command("sys_status")
    async def sys_status_cmd(self, args: list):
        """
        Usage: /sys_status
        """
        return await self.get_system_status()

    @core.module.command("sys_temps")
    async def sys_temps_cmd(self, args: list):
        """
        Usage: /sys_temps
        """
        return await self.get_temperatures()

    @core.module.command("sys_gpu")
    async def sys_gpu_cmd(self, args: list):
        """
        Usage: /sys_gpu
        """
        return await self.get_gpu_status()

    @core.module.command("sys_processes")
    async def sys_processes_cmd(self, args: list):
        """
        Usage: /sys_processes
        """
        return await self.get_processes()

    @core.module.command("sys_memory")
    async def sys_memory_cmd(self, args: list):
        """
        Usage: /sys_memory
        """
        return await self.get_memory_status()

    @core.module.command("sys_disk")
    async def sys_disk_cmd(self, args: list):
        """
        Usage: /sys_disk
        """
        return await self.get_disk_status()

    @core.module.command("sys_load")
    async def sys_load_cmd(self, args: list):
        """
        Usage: /sys_load
        """
        return await self.get_load_status()