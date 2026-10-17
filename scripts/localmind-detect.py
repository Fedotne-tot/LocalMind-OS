#!/usr/bin/env python3
"""localmind-detect v0 — read-only аппаратный детектор LocalMind OS.

Этап E1 плана (docs/ARCHITECTURE.md).

Гарантии:
  * только чтение: /sys, /dev, lspci; ничего не устанавливает, не пишет
    вне stdout; не требует root;
  * вывод соответствует config/report-schema.json (поле schema_version);
  * отсутствие GPU = честный отчёт с пустым списком gpu, а НЕ выдуманные
    значения и НЕ CPU-«совместимость».

Использование:
  ./localmind-detect            # человекочитаемый вывод
  ./localmind-detect --json     # JSON-отчёт
  LM_FAKE_SYS=/path/to/fake ./localmind-detect --json   # для тестов (фикстуры)
"""

import json
import os
import shutil
import socket
import subprocess
import sys

SCHEMA_VERSION = "0.1.0"
TOOL_VERSION = "0.1.0"

# PCI vendor IDs (публичные данные PCI SIG / upstream hwdata pci.ids)
VENDOR_IDS = {
    "0x1002": "AMD/ATI",
    "0x10de": "NVIDIA",
    "0x8086": "Intel",
}

# Известные AMD gfx-алиасы из driver_name amdgpu (источник: ядро Linux,
# drivers/gpu/drm/amd — по мере расширения whitelist обновлять здесь).
GFX_ALIASES = {
    "gfx1030": "RDNA2", "gfx1031": "RDNA2", "gfx1032": "RDNA2",
    "gfx1034": "RDNA2", "gfx1035": "RDNA2", "gfx1036": "RDNA2",
    "gfx1100": "RDNA3", "gfx1101": "RDNA3", "gfx1102": "RDNA3",
    "gfx1103": "RDNA3", "gfx1150": "RDNA3.5",
}


def sysfs_root() -> str:
    """Корень /sys (или подмена для тестов через LM_FAKE_SYS)."""
    return os.environ.get("LM_FAKE_SYS", "/")


def read_file(path: str) -> str | None:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def dev_exists(relpath: str) -> bool:
    return os.path.exists(os.path.join(sysfs_root(), relpath.lstrip("/")))


def detect_drm_gpus() -> list[dict]:
    """Карта DRM-устройств из /sys/class/drm (только чтение)."""
    gpus = []
    drm_dir = os.path.join(sysfs_root(), "sys/class/drm")
    if not os.path.isdir(drm_dir):
        return gpus
    for card in sorted(os.listdir(drm_dir)):
        if not card.startswith("card") or "-" in card:
            continue
        base = os.path.join(drm_dir, card, "device")
        vendor_raw = read_file(os.path.join(base, "vendor"))
        device_raw = read_file(os.path.join(base, "device"))
        if vendor_raw is None or device_raw is None:
            continue
        try:
            vendor = int(vendor_raw, 16)
            device = int(device_raw, 16)
        except ValueError:
            continue
        # в реальном sysfs driver — symlink; имя драйвера = базовый компонент
        drv_link = os.path.join(base, "driver")
        driver = None
        if os.path.islink(drv_link):
            driver = os.path.basename(os.readlink(drv_link))
        elif os.path.exists(drv_link):
            driver = read_file(drv_link)
        entry = {
            "card": card,
            "vendor_id": f"0x{vendor:04x}",
            "device_id": f"0x{device:04x}",
            "vendor": VENDOR_IDS.get(f"0x{vendor:04x}", "unknown"),
            "driver": driver,
            "model": read_file(os.path.join(base, "label")) or None,
        }
        mem_info = read_file(os.path.join(base, "mem_info_vram_total"))
        if mem_info:
            try:
                entry["vram_bytes"] = int(mem_info)
            except ValueError:
                pass
        gpus.append(entry)
    return gpus


def detect_kfd_nodes() -> list[dict]:
    """Узлы ROCm/KFD: /sys/devices/virtual/kfd/kfd/topology/nodes/N."""
    nodes = []
    top = os.path.join(sysfs_root(), "sys/devices/virtual/kfd/kfd/topology/nodes")
    if not os.path.isdir(top):
        return nodes
    for node in sorted(os.listdir(top), key=lambda n: int(n) if n.isdigit() else 999):
        ndir = os.path.join(top, node)
        props = {}
        pfile = os.path.join(ndir, "properties")
        if os.path.isfile(pfile):
            with open(pfile, "r", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2:
                        props[parts[0]] = parts[1]
        entry = {"node": node}
        if "name" in props:
            entry["name"] = props["name"]
        if "cpu_count" in props:
            entry["cpu_count"] = int(props["cpu_count"])
        # GPU-узел KFD отличается наличием vdhip/unique_id без cpu-ядер;
        # сохраняем сырые ключи, интерпретация — задача probe-слоя (E3).
        entry["raw_keys"] = sorted(props.keys())
        nodes.append(entry)
    return nodes


def lspci_display() -> list[dict]:
    """lspci -nn, только дисплейные/3D классы; None если lspci недоступен."""
    if not shutil.which("lspci"):
        return []
    try:
        out = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True, timeout=10
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    devices = []
    for line in out.splitlines():
        low = line.lower()
        if "display controller" in low or "3d controller" in low or "vga compatible" in low:
            devices.append({"raw": line.strip()})
    return devices


def detect_stack_presence() -> dict:
    """Наличие артефактов ROCm/Vulkan в файловой системе (факт, не verdict)."""
    stacks = {}
    rocm_dir = os.path.join(sysfs_root(), "opt/rocm")
    stacks["rocm_install_dir"] = rocm_dir if os.path.isdir(rocm_dir) else None
    ver = None
    if os.path.isdir(rocm_dir):
        ver = read_file(os.path.join(rocm_dir, ".info/version"))
    stacks["rocm_version_file"] = ver
    stacks["rocminfo_binary"] = bool(shutil.which("rocminfo"))
    stacks["vulkaninfo_binary"] = bool(shutil.which("vulkaninfo"))
    icd_paths = [
        "usr/share/vulkan/icd.d",
        "etc/vulkan/icd.d",
    ]
    stacks["vulkan_icd_dirs_present"] = [
        p for p in icd_paths if os.path.isdir(os.path.join(sysfs_root(), p))
    ]
    return stacks


def classify(gpus: list[dict], kfd: list[dict]) -> dict:
    """Честная первичная классификация. NOT_TESTED ≠ PASS (POLICY P4)."""
    amd_gpu = any(g["vendor_id"] == "0x1002" for g in gpus)
    kfd_present = dev_exists("dev/kfd")
    amdgpu_driver = any(g.get("driver") == "amdgpu" for g in gpus)
    gfx = [a for a in GFX_ALIASES if a in json.dumps(kfd)]
    return {
        "amd_gpu_detected": amd_gpu,
        "amdgpu_driver_bound": amdgpu_driver,
        "kfd_device_present": kfd_present,
        "known_gfx_aliases_in_kfd": gfx,
        "preliminary_verdict": (
            "ROCM_CANDIDATE" if (amd_gpu and kfd_present and amdgpu_driver)
            else "VULKAN_ONLY_CANDIDATE" if (amd_gpu and amdgpu_driver)
            else "NO_AMD_GPU"
        ),
        "note": "verdict предварительный; ROCm/Vulkan PASS выдаёт только probe (E3/E4)",
    }


def build_report() -> dict:
    uname_r = None
    try:
        uname_r = os.uname().release
    except AttributeError:
        pass
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "os_pretty_name": read_file("etc/os-release"),
        "kernel_release": uname_r,
        "drm_gpus": detect_drm_gpus(),
        "kfd_nodes": detect_kfd_nodes(),
        "lspci_display": lspci_display(),
        "stacks": detect_stack_presence(),
        "classification": classify(detect_drm_gpus(), detect_kfd_nodes()),
        "policy": {"gpu_strict_default": True, "silent_cpu_fallback": "forbidden"},
    }
    return report


def human(report: dict) -> str:
    lines = [
        f"LocalMind detect v{report['tool_version']} (schema {report['schema_version']})",
        f"OS:     {(report['os_pretty_name'] or 'unknown').splitlines()[0] if report['os_pretty_name'] else 'unknown'}",
        f"Kernel: {report['kernel_release']}",
        f"GPU (DRM): {len(report['drm_gpus'])} шт.",
    ]
    for g in report["drm_gpus"]:
        vram = g.get("vram_bytes")
        lines.append(
            f"  - {g['card']}: {g['vendor']} [{g['vendor_id']}:{g['device_id']}] "
            f"driver={g['driver'] or 'none'}"
            + (f" vram={vram // (1024*1024)} MiB" if vram else "")
        )
    if not report["drm_gpus"]:
        lines.append("  (DRM-видеоустройства не найдены)")
    c = report["classification"]
    lines += [
        f"/dev/kfd: {'есть' if c['kfd_device_present'] else 'нет'}",
        f"Предварительный вердикт: {c['preliminary_verdict']}",
        f"{c['note']}",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if "--schema-version" in argv:
        print(SCHEMA_VERSION)
        return 0
    report = build_report()
    if "--json" in argv:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(human(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
