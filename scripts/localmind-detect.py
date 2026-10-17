#!/usr/bin/env python3
"""localmind-detect v0.2 — read-only аппаратный детектор LocalMind OS.

Этап E1 плана (docs/ARCHITECTURE.md) + требования первой функциональной
задачи (CPU/RAM/GPU/статусы/диагностика).

Гарантии:
  * только чтение: /sys, /proc, /dev, lspci, lsmod; ничего не устанавливает,
    не пишет вне stdout; не требует root; нагрузочные тесты не запускает;
  * вывод соответствует config/report-schema.json (поле schema_version);
  * отсутствие GPU = честный отчёт с пустым списком gpu, а НЕ выдуманные
    значения и НЕ CPU-«совместимость»;
  * каждый параметр имеет явный статус: detected | unavailable | unknown |
    unsupported | error (см. STATUS_* ниже);
  * наличие ROCm/Vulkan в системе — ФАКТ окружения, а НЕ вердикт о поддержке
    конкретной модели GPU (POLICY P4: NOT_TESTED != PASS).

Использование:
  ./localmind-detect              # человекочитаемый вывод
  ./localmind-detect --json       # JSON-отчёт
  ./localmind-detect --doctor     # первичная диагностика GPU (exit 0/1)
  LM_FAKE_SYS=/path/to/fake ./localmind-detect --json   # фикстуры (тесты)
"""

import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys

SCHEMA_VERSION = "0.2.0"
TOOL_VERSION = "0.2.0"

# Явные статусы параметров (требование ТЗ).
STATUS_DETECTED = "detected"
STATUS_UNAVAILABLE = "unavailable"    # ресурса/устройства нет в системе
STATUS_UNKNOWN = "unknown"            # есть, но данные неполны/непонятны
STATUS_UNSUPPORTED = "unsupported"    # инструмент/функция не поддерживается
STATUS_ERROR = "error"                # попытка сбора завершилась ошибкой

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


def read_lines(path: str) -> list[str] | None:
    """Чтение построчно; None если файл недоступен (отличается от пустого)."""
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return None


def dev_exists(relpath: str) -> bool:
    return os.path.exists(os.path.join(sysfs_root(), relpath.lstrip("/")))


# --------------------------------------------------------------------------
# CPU / RAM (procfs, только чтение)
# --------------------------------------------------------------------------

def detect_cpu() -> dict:
    """CPU model и архитектура из /proc/cpuinfo + platform.machine()."""
    arch = None
    try:
        arch = platform.machine()
    except Exception:
        pass
    info = {
        "architecture": arch or STATUS_UNKNOWN,
        "model": None,
        "cores": None,
        "status": STATUS_UNAVAILABLE,
    }
    lines = read_lines(os.path.join(sysfs_root(), "proc/cpuinfo"))
    if lines is None:
        return info
    model = None
    cores = 0
    for line in lines:
        low = line.lower()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        # x86: "model name"; ARM: "Model"/"Hardware"; ppc: "model"
        if key in ("model name", "hardware", "model") and not model and val:
            # "model" на x86 — числовой ревизии-идентификатор (0x...), не имя;
            # берём его только когда это читаемое строковое имя (ARM/ppc)
            if key == "model" and (val.startswith("0x") or val.isdigit()):
                pass
            else:
                model = val
        if key == "processor":
            cores += 1
        if key == "cpu architecture" and arch in (None, STATUS_UNKNOWN):
            info["architecture"] = val  # s390x-style cpuinfo
    if model or cores:
        info["status"] = STATUS_DETECTED
    info["model"] = model
    info["cores"] = cores or None
    return info


def detect_memory() -> dict:
    """Системная RAM из /proc/meminfo (MemTotal/MemAvailable)."""
    mem = {
        "status": STATUS_UNAVAILABLE,
        "mem_total_bytes": None,
        "mem_available_bytes": None,
    }
    lines = read_lines(os.path.join(sysfs_root(), "proc/meminfo"))
    if lines is None:
        return mem
    wanted = {"MemTotal": "mem_total_bytes", "MemAvailable": "mem_available_bytes"}
    found = False
    parse_failed = False
    for line in lines:
        key, _, rest = line.partition(":")
        if key in wanted:
            parts = rest.split()
            if parts:
                try:
                    mem[wanted[key]] = int(parts[0]) * 1024  # kB -> bytes
                    found = True
                except ValueError:
                    # испорченное значение: error имеет приоритет над detected
                    # (инженерная проверка B1: раньше status=detected при total=None)
                    parse_failed = True
    if parse_failed:
        mem["status"] = STATUS_ERROR
    elif found:
        mem["status"] = STATUS_DETECTED
    else:
        mem["status"] = STATUS_UNKNOWN
    return mem


# Обобщённые label-строки amdgpu: по ним нельзя отличить конкретную модель
# (например, RX 7600 и iGPU дают одинаковый "AMD Radeon Graphics").
GENERIC_MODEL_LABELS = ("AMD Radeon Graphics", "AMD ATI Radeon Graphics")


def _model_status(model) -> str:
    """Честный статус модели (инженерная проверка B2).

    detected — только точное имя из таблицы subsystem-ID (проверяемое);
    unknown  — обобщённый DRM label (не позволяет назвать модель);
    unavailable — имени нет вовсе.
    """
    if model is None:
        return STATUS_UNAVAILABLE
    if any(g.lower() in model.lower() for g in GENERIC_MODEL_LABELS):
        return STATUS_UNKNOWN
    return STATUS_DETECTED


def detect_drm_gpus() -> list[dict]:
    """Карта DRM-устройств из /sys/class/drm (только чтение).

    Каждый GPU получает явные статусы полей (vram/driver/model) и
    compute_status — предварительную оценку вычислительной готовности.
    unsupported/unknown здесь НЕ означает FAIL probe: финальный verdict
    выдаёт только probe-слой (E3/E4) на реальном железе.
    """
    gpus = []
    drm_dir = os.path.join(sysfs_root(), "sys/class/drm")
    if not os.path.isdir(drm_dir):
        return gpus
    cards = [c for c in sorted(os.listdir(drm_dir))
             if c.startswith("card") and "-" not in c]
    # нумерация renderD НЕ совпадает с номером card (реальный Linux);
    # соответствие cardN → renderD M восстанавливается по общему PCI-узлу
    # (basename realpath device-каталога), как это делают udev/libdrm.
    renders = {}
    for r in os.listdir(drm_dir):
        if r.startswith("renderD"):
            rp = os.path.basename(
                os.path.realpath(os.path.join(drm_dir, r, "device"))
            )
            if rp:
                renders[rp] = f"sys/class/drm/{r}"
    for card in cards:
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
        pci_path = os.path.basename(os.path.realpath(base)) or None
        entry = {
            "card": card,
            "vendor_id": f"0x{vendor:04x}",
            "device_id": f"0x{device:04x}",
            "vendor": VENDOR_IDS.get(f"0x{vendor:04x}", "unknown"),
            "driver": driver,
            "driver_status": STATUS_DETECTED if driver else STATUS_UNAVAILABLE,
            "model": read_file(os.path.join(base, "label")) or None,
            # v0.2 (инженерная проверка B2): label/таблица не дают точной модели —
            # это предположение, а не подтверждение; final model — задача probe E3
            "pci_path": pci_path,
        }
        entry["model_status"] = _model_status(entry["model"])
        # render node (DRM renderD для Vulkan/compute-доступа к карте)
        entry["render_node"] = renders.get(pci_path)
        mem_info = read_file(os.path.join(base, "mem_info_vram_total"))
        if mem_info is None:
            # файла нет: у дискретных amdgpu он есть; у iGPU VRAM — часть RAM
            entry["vram_bytes"] = None
            entry["vram_status"] = (
                STATUS_UNKNOWN if driver == "amdgpu" else STATUS_UNAVAILABLE
            )
        else:
            try:
                entry["vram_bytes"] = int(mem_info)
                entry["vram_status"] = STATUS_DETECTED
            except ValueError:
                entry["vram_bytes"] = None
                entry["vram_status"] = STATUS_ERROR
        entry["compute_status"] = _gpu_compute_status(entry)
        gpus.append(entry)
    return gpus


def _gpu_compute_status(g: dict) -> str:
    """Предварительная оценка вычислительной готовности GPU (НЕ PASS!).

    detected   — amdgpu + KFD-узел с известным gfx-алиасом (ROCm-кандидат);
    unsupported— драйвер не amdgpu (например nouveau/игровой режим NVIDIA)
                 ИЛИ заведомо неподдерживаемый ROCm iGPU (RDNA1 gfx103x без KFD);
    unknown    — всё остальное: данных достаточно для кандидата, но
                 недостаточно для вывода (требует probe E3/E4).
    """
    if g["vendor_id"] != "0x1002":
        return STATUS_UNKNOWN  # чужие GPU: вердикт не наша задача на этом слое
    if g.get("driver") != "amdgpu":
        return STATUS_UNSUPPORTED
    if g.get("kfd_gfx"):
        return STATUS_DETECTED
    if g.get("kfd_node_present"):
        return STATUS_UNKNOWN
    if g.get("device_id") in RDNA1_IGPU_NO_ROCM:
        return STATUS_UNSUPPORTED
    return STATUS_UNKNOWN


# AMD iGPU RDNA1 (Van Gogh), официально НЕ поддерживаются ROCm
# (поддержка начата с RDNA2; источник: docs.amd.com/rocm support matrix).
# Список расширяется/сужается ТОЛЬКО по данным golden-отчётов (POLICY P5).
RDNA1_IGPU_NO_ROCM = {"0x1638"}


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
            # инженерная проверка B4: open() без обработки ошибок — на живой
            # системе (не root) нечитаемый properties валился с traceback;
            # теперь это честный status=error, детектор продолжает работу
            try:
                with open(pfile, "r", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                nodes.append({"node": node, "status": STATUS_ERROR})
                continue
        else:
            content = ""
        for line in content.splitlines():
            parts = line.split()
            if len(parts) == 2:
                props[parts[0]] = parts[1]
        entry = {"node": node, "status": STATUS_DETECTED}
        if "name" in props:
            entry["name"] = props["name"]
        if "cpu_count" in props:
            try:
                entry["cpu_count"] = int(props["cpu_count"])
            except ValueError:
                # повреждённое sysfs-значение (защита от нестабильности
                # после обновлений ядра/ROCm): поле — unknown, не падение
                entry["status"] = STATUS_ERROR
        # GPU-узел KFD отличается наличием vdhip/unique_id без cpu-ядер;
        # сохраняем сырые ключи, интерпретация — задача probe-слоя (E3).
        entry["raw_keys"] = sorted(props.keys())
        nodes.append(entry)
    return nodes


def detect_subsystem_names() -> dict:
    """Модельные имена из PCI subvendor/subdevice (только чтение /sys).

    Для AMD GPU пара subsystem_vendor:subsystem_device — стабильный ключ к
    маркетинговому имени карты (источник данных: upstream hwdata pci.ids).
    Возвращает {sub_key: "Radeon RX 7900 XTX"}; неизвестные пары пропускаются.
    Логировать сырые sub*-ID НЕ нужно — они не персональные, но в отчёт
    кладём только итоговое имя (принцип минимизации данных).
    """
    names = {}
    drm_dir = os.path.join(sysfs_root(), "sys/class/drm")
    if not os.path.isdir(drm_dir):
        return names
    for card in sorted(os.listdir(drm_dir)):
        if not card.startswith("card") or "-" in card:
            continue
        base = os.path.join(drm_dir, card, "device")
        sv = read_file(os.path.join(base, "subsystem_vendor"))
        sd = read_file(os.path.join(base, "subsystem_device"))
        dv = read_file(os.path.join(base, "vendor"))
        dd = read_file(os.path.join(base, "device"))
        if not all((sv, sd, dv, dd)):
            continue
        key = f"{dv}:{dd}:{sv}:{sd}"
        name = AMD_SUBSYSTEM_NAMES.get(key)
        if name:
            names[key] = name
    return names


# PCI subsystem-пары дискретных Radeon (данные: hwdata pci.ids, public domain).
# Расширяется ТОЛЬКО проверенными upstream-данными или golden-отчётами (P5).
AMD_SUBSYSTEM_NAMES = {
    "0x1002:0x744c:0x0732:0x0008": "Radeon RX 7900 XTX",
    "0x1002:0x744d:0x0732:0x0008": "Radeon RX 7900 XT",
    "0x1002:0x744e:0x0732:0x0008": "Radeon RX 7900 GRE",
    "0x1002:0x7480:0x0732:0x0008": "Radeon RX 7800 XT",
    "0x1002:0x7482:0x0732:0x0008": "Radeon RX 7700 XT",
}


def detect_loaded_modules() -> dict:
    """Загруженные ядровые модули интересующих стеков (только чтение).

    Источник — /proc/modules (readable, не требует root); lsmod не вызывается.
    Нет файла => unavailable (не error: возможно, окружение без procfs).
    """
    mods = {"amdgpu": None, "drm": None, "amdkfd": None}
    lines = read_lines(os.path.join(sysfs_root(), "proc/modules"))
    if lines is None:
        return {k: STATUS_UNAVAILABLE for k in mods}
    try:
        loaded = {line.split()[0] for line in lines if line.strip()}
    except Exception:
        # инженерная проверка C1: битая/неожидаланная строка валила весь
        # отчёт NameError'ом; теперь честный status=error, детектор живёт
        return {k: STATUS_ERROR for k in mods}
    return {m: (STATUS_DETECTED if m in loaded else STATUS_UNAVAILABLE)
            for m in mods}


def run_cmd(argv: list[str]) -> str | None:
    """Безопасное выполнение read-only команды; None при любой неудаче."""
    exe = shutil.which(argv[0])
    if not exe:
        return None
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def lspci_display() -> list[dict]:
    """lspci -nn, только дисплейные/3D классы; [] если lspci недоступен.

    Отсутствие lspci — это unsupported инструмента, а НЕ отсутствие GPU:
    статус фиксируется отдельно в stacks.tool_status.lspci.
    """
    if not shutil.which("lspci"):
        return []
    out = run_cmd(["lspci", "-nn"])
    if out is None:
        return []
    devices = []
    for line in out.splitlines():
        low = line.lower()
        if ("display controller" in low or "3d controller" in low
                or "vga compatible" in low):
            devices.append({"raw": line.strip()})
    return devices


def detect_stack_presence() -> dict:
    """Наличие артефактов ROCm/Vulkan в файловой системе (факт, не verdict).

    ВАЖНО: tool_status/dirs отвечают на вопрос «есть ли инструмент/библиотека»,
    а НЕ «поддерживается ли конкретный GPU стеком». Вердикт поддержки — только
    probe-слой (E3/E4) и golden-отчёты (POLICY P4).
    """
    stacks = {}
    rocm_dir = os.path.join(sysfs_root(), "opt/rocm")
    stacks["rocm_install_dir"] = rocm_dir if os.path.isdir(rocm_dir) else None
    ver = None
    if os.path.isdir(rocm_dir):
        ver = read_file(os.path.join(rocm_dir, ".info/version"))
    stacks["rocm_version_file"] = ver
    # статусы инструментов: detected | unavailable(нет бинаря) | error(запуск упал)
    def tool_status(name: str, args: list[str] | None = None) -> str:
        if not shutil.which(name):
            return STATUS_UNAVAILABLE
        # инженерная проверка C4 (нестабильность после обновлений): bинарь
        # мог оказаться в битом окружении (напр. удалённая libstdc++), где
        # subprocess.run бросает OSError; раньше это валило весь отчёт —
        # теперь честный status=error
        try:
            out = run_cmd(args or [name, "--version"])
        except Exception:
            return STATUS_ERROR
        return STATUS_DETECTED if out is not None else STATUS_ERROR
    stacks["rocminfo_binary"] = bool(shutil.which("rocminfo"))
    stacks["vulkaninfo_binary"] = bool(shutil.which("vulkaninfo"))
    stacks["tool_status"] = {
        "lspci": tool_status("lspci", ["lspci", "-nn"]),
        # инженерная проверка: rocminfo/vulkaninfo без аргументов НЕ запускаем —
        # rocminfo перечисляет устройства и может виснуть без /dev/kfd, это
        # выходит за рамки read-only детектора; presence = availability bинаря,
        # полноценный probe — слой E3/E4 (POLICY P4)
        "rocminfo": (STATUS_DETECTED if stacks["rocminfo_binary"]
                     else STATUS_UNAVAILABLE),
        "vulkaninfo": (STATUS_DETECTED if stacks["vulkaninfo_binary"]
                       else STATUS_UNAVAILABLE),
        "rocm_smi_lib": (STATUS_DETECTED
                         if glob.glob(os.path.join(sysfs_root(),
                              "opt/rocm*/lib/librocmsmi*"))
                         else STATUS_UNAVAILABLE),
    }
    icd_paths = [
        "usr/share/vulkan/icd.d",
        "etc/vulkan/icd.d",
    ]
    stacks["vulkan_icd_dirs_present"] = [
        p for p in icd_paths if os.path.isdir(os.path.join(sysfs_root(), p))
    ]
    # наличие RADV ICD (Mesa Vulkan для AMD) — факт, не вердикт о GPU
    radv = glob.glob(os.path.join(sysfs_root(), "usr/share/vulkan/icd.d/radeon_icd*.json"))
    stacks["radv_icd_files"] = bool(radv)
    return stacks


def correlate_kfd_to_gpus(gpus: list[dict], kfd: list[dict]) -> None:
    """Связывает DRM-GPU с KFD-узлами по gfx-алиасам (in-place).

    Инженерная проверка B3: раньше ANY известный gfx-алиас приписывался
    EVERY AMD-карте — на смешанной системе вторая AMD-карта получала чужой
    gfx и ложный compute_status=detected. Теперь алиасы распределяются по
    одному разу; лишним AMD-картам — unknown (точное соответствие dev→node
    даёт только probe E3 через /dev/kfd ioctl).
    """
    amd_cards = [gpu for gpu in gpus if gpu["vendor_id"] == "0x1002"]
    # Инженерная проверка C2: раньше узлы с ОТСУТСТВИЕМ поля name (битый/
    # нечитаемый properties, старые ядра без CRAT) молча отбрасывались —
    # на системе с реальной RX-картой это давало kfd_node_present=False и
    # ложный compute_status=unsupported. Теперь такие узлы считаются
    # «KFD есть» (unknown), unsupported остаётся только для доказанных
    # случаев (не-amdgpu драйвер, whitelist-модель).
    named_nodes = [n for n in kfd if n.get("status") == STATUS_DETECTED]
    gfx_in_nodes = [n.get("name") for n in named_nodes if n.get("name")]
    known_gfx = [g for g in gfx_in_nodes if g in GFX_ALIASES]
    unknown_gfx_nodes = [n for n in named_nodes if not n.get("name")]
    pool = list(known_gfx)  # каждый алиас можно использовать один раз
    # если AMD-карт столько же, сколько известных gfx-узлов, порядок cardN
    # обычно совпадает с порядком узлов — но это эвристика v0, поэтому при
    # неоднозначности (разные количества) распределение НЕ выполняем вовсе
    exact = len(pool) == len(amd_cards)
    has_kfd_device = dev_exists("dev/kfd") or bool(kfd)
    for gpu in amd_cards:
        if gpu.get("device_id") in RDNA1_IGPU_NO_ROCM:
            gpu["kfd_node_present"] = False
        elif pool and exact:
            gfx = pool.pop(0)
            gpu["kfd_node_present"] = True
            gpu["kfd_gfx"] = gfx
            gpu["architecture"] = GFX_ALIASES[gfx]
        elif known_gfx or unknown_gfx_nodes:
            # неоднозначно или имя узла неизвестно: KFD есть, привязать нельзя
            gpu["kfd_node_present"] = True
        else:
            gpu["kfd_node_present"] = has_kfd_device


def classify(gpus: list[dict], kfd: list[dict]) -> dict:
    """Честная первичная классификация. NOT_TESTED ≠ PASS (POLICY P4)."""
    amd_gpu = any(g["vendor_id"] == "0x1002" for g in gpus)
    kfd_present = dev_exists("dev/kfd") or bool(kfd)  # C3: узлы topology —
    # тоже доказательство KFD, даже если /dev/kfd не виден из песочницы/контейнера
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


def os_release_pretty() -> str | None:
    """PRETTY_NAME из /etc/os-release (с учётом подмены корня для фикстур).

    Исправление дефекта v0.1: раньше читался относительный путь "etc/os-release"
    без sysfs_root(), поэтому поле всегда было null даже при наличии файла.
    """
    path = os.path.join(sysfs_root(), "etc/os-release")
    lines = read_lines(path)
    if lines is None:
        return None
    for line in lines:
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def build_report() -> dict:
    kernel_release = None
    try:
        kernel_release = os.uname().release
    except AttributeError:
        pass
    gpus = detect_drm_gpus()
    kfd = detect_kfd_nodes()
    correlate_kfd_to_gpus(gpus, kfd)
    names = detect_subsystem_names()
    for g in gpus:
        key = f"{g['vendor_id']}:{g['device_id']}"
        # уточнение модели по subsystem-паре, если она есть в таблице
        for k, name in names.items():
            if k.startswith(key + ":"):
                g["model"] = name
                g["model_status"] = STATUS_DETECTED
                break
    # compute_status пересчитывается ПОСЛЕ корреляции с KFD (v0.2: порядок
    # вызовов — ошибка ревью; в v0.1 статус считался до появления kfd_gfx)
    for g in gpus:
        g["compute_status"] = _gpu_compute_status(g)
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "os_pretty_name": os_release_pretty(),
        "kernel_release": kernel_release,
        "cpu": detect_cpu(),
        "memory": detect_memory(),
        "drm_gpus": gpus,
        "kfd_nodes": kfd,
        "lspci_display": lspci_display(),
        "loaded_modules": detect_loaded_modules(),
        "stacks": detect_stack_presence(),
        "classification": classify(gpus, kfd),
        "policy": {"gpu_strict_default": True, "silent_cpu_fallback": "forbidden"},
    }
    return report


def gpu_diagnosis(report: dict) -> list[str]:
    """Первичная диагностика GPU — человекочитаемые строки с явными статусами.

    Правила честности:
      * отсутствие инструмента != отсутствие GPU;
      * ROCm/Vulkan «присутствует» != «поддерживает данную модель»;
      * вердикт только предварительный, финальный — за probe (E3/E4).
    """
    d = []
    c = report["classification"]
    tools = report["stacks"]["tool_status"]
    amd = [g for g in report["drm_gpus"] if g["vendor_id"] == "0x1002"]
    if not report["drm_gpus"]:
        d.append("GPU: не обнаружено DRM-устройств (статус: unavailable)")
        return d
    for g in report["drm_gpus"]:
        cs = g["compute_status"]
        model = g.get("model") or "неизвестна"
        d.append(f"GPU {g['card']}: {g['vendor']} [{g['vendor_id']}:{g['device_id']}] "
                 f"модель={model} драйвер={g['driver'] or 'нет'} ({g['driver_status']})")
        vr = g.get("vram_bytes")
        if g["vram_status"] == STATUS_DETECTED and vr:
            d.append(f"  VRAM: {vr // (1024*1024)} MiB (detected)")
        elif g["vram_status"] == STATUS_UNKNOWN:
            d.append("  VRAM: unknown (amdgpu без mem_info_vram_total — "
                     "обычно iGPU, VRAM делит ОЗУ)")
        else:
            d.append(f"  VRAM: {g['vram_status']}")
        if g["vendor_id"] == "0x1002":
            if cs == STATUS_DETECTED:
                d.append(f"  Compute: кандидат ROCm (KFD gfx={g.get('kfd_gfx')}, "
                         f"архитектура {g.get('architecture', 'unknown')}); "
                         "финальная поддержка — только после probe (NOT_TESTED ≠ PASS)")
            elif cs == STATUS_UNSUPPORTED:
                if g.get("driver") != "amdgpu":
                    d.append("  Compute: unsupported — драйвер не amdgpu "
                             "(вычислительный стек недоступен)")
                else:
                    d.append("  Compute: unsupported — модель вне whitelist ROCm "
                             "(проверить Vulkan-путь через probe E4)")
            else:
                d.append("  Compute: unknown — данных достаточно для кандидата, "
                         "требуется probe ROCm/Vulkan (E3/E4)")
    if amd:
        if tools["rocminfo"] == STATUS_UNAVAILABLE:
            d.append("ROCm: инструмент rocminfo unavailable — это НЕ означает "
                     "отсутствие поддержки GPU; наличие /opt/rocm: "
                     + ("да" if report["stacks"]["rocm_install_dir"] else "нет"))
        elif tools["rocminfo"] == STATUS_ERROR:
            d.append("ROCm: rocminfo найден, но запуск завершился ошибкой (error)")
        if tools["vulkaninfo"] == STATUS_UNAVAILABLE:
            d.append("Vulkan: vulkaninfo unavailable — проверка невозможна; "
                     "ICD-каталоги: "
                     + (", ".join(report["stacks"]["vulkan_icd_dirs_present"]) or "нет"))
        elif tools["vulkaninfo"] == STATUS_ERROR:
            d.append("Vulkan: vulkaninfo найден, но запуск завершился ошибкой (error)")
    if not c["kfd_device_present"] and amd:
        d.append("/dev/kfd: unavailable — ROCm-путь для этих GPU закрыт "
                 "(драйвер/ядро или unsupported модель); Vulkan может быть доступен")
    d.append(c["note"])
    return d


def human(report: dict) -> str:
    lines = [
        f"LocalMind detect v{report['tool_version']} (schema {report['schema_version']})",
        f"OS:     {report['os_pretty_name'] or 'unknown'}",
        f"Kernel: {report['kernel_release']}",
    ]
    cpu = report["cpu"]
    lines.append(
        f"CPU:    {cpu['model'] or 'unknown'} ({cpu['architecture']}, "
        f"cores={cpu['cores'] or '?'}, status={cpu['status']})"
    )
    mem = report["memory"]
    if mem["mem_total_bytes"]:
        lines.append(
            f"RAM:    total {mem['mem_total_bytes'] // (1024**3)} GiB, "
            f"available {(mem['mem_available_bytes'] or 0) // (1024**3)} GiB "
            f"({mem['status']})"
        )
    else:
        lines.append(f"RAM:    {mem['status']}")
    lines.append(f"GPU (DRM): {len(report['drm_gpus'])} шт.")
    lines.extend(gpu_diagnosis(report))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if "--schema-version" in argv:
        print(SCHEMA_VERSION)
        return 0
    report = build_report()
    if "--doctor" in argv:
        # первичная диагностика GPU; exit 0 — отчёт собран, 1 — ошибка сбора
        print("\n".join(gpu_diagnosis(report)))
        return 0
    if "--json" in argv:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(human(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
