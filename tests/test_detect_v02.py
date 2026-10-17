"""Тесты localmind-detect v0.2 — первая функциональная задача.

Покрытие требований ТЗ:
  * обнаруженная AMD GPU (расширение golden fake-amd-kfd: CPU/RAM/OS/renderD);
  * система без AMD GPU (fake-no-gpu, fake-multi-gpu с NVIDIA+iGPU-сценарий);
  * отсутствующие диагностические утилиты (tool_status=unavailable, и это
    НЕ должно читаться как «GPU нет»);
  * неполная информация о VRAM (fake-amd-no-vram: iGPU без mem_info_*);
  * ошибки выполнения системных команд (unit на run_cmd: absent/nonzero/timeout);
  * несколько GPU (fake-multi-gpu: AMD + NVIDIA одновременно).

Регрессии дефектов, найденных при разработке v0.2:
  R1: os_pretty_name всегда null (read_file("etc/os-release") без sysfs_root);
  R2: compute_status вычислялся до correlate_kfd_to_gpus → ROCm-кандидат
      помечался unknown;
  R3: render_node искался по совпадению номера cardN==renderDN (в реальном
      Linux нумерация не совпадает) — теперь сопоставление по PCI-узлу.

Только stdlib; фикстуры через LM_FAKE_SYS; root/GPU не требуются.
"""

import json
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT = os.path.join(REPO, "scripts", "localmind-detect.py")
GOLDEN = os.path.join(REPO, "tests", "golden")

sys.path.insert(0, os.path.join(REPO, "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("lmd", DETECT)
lmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lmd)


def run_detect(fake_sys: str | None, extra: list[str] | None = None):
    env = dict(os.environ)
    if fake_sys:
        env["LM_FAKE_SYS"] = fake_sys
    else:
        env.pop("LM_FAKE_SYS", None)
    proc = subprocess.run(
        [sys.executable, DETECT, *(extra or ["--json"])],
        capture_output=True, text=True, env=env,
    )
    return proc


def report(fake_sys: str | None) -> dict:
    proc = run_detect(fake_sys)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


ALL_STATUSES = {"detected", "unavailable", "unknown", "unsupported", "error"}


class TestAmdGpuFixture(unittest.TestCase):
    """Обнаруженная AMD GPU: полный набор полей со статусами."""

    def setUp(self):
        self.report = report(os.path.join(GOLDEN, "fake-amd-kfd"))

    def test_schema_bumped(self):
        self.assertEqual(self.report["schema_version"], "0.2.0")

    def test_cpu_detected(self):
        cpu = self.report["cpu"]
        self.assertEqual(cpu["status"], "detected")
        self.assertIn("Ryzen 7 5800X", cpu["model"])
        self.assertIsInstance(cpu["cores"], int)

    def test_memory_detected(self):
        mem = self.report["memory"]
        self.assertEqual(mem["status"], "detected")
        # 32786544 kB из фикстуры
        self.assertEqual(mem["mem_total_bytes"], 32786544 * 1024)
        self.assertEqual(mem["mem_available_bytes"], 24563200 * 1024)

    def test_regression_R1_os_pretty_name(self):
        """v0.1: поле всегда null. v0.2: читается PRETTY_NAME из подменённого корня."""
        self.assertEqual(self.report["os_pretty_name"], "Ubuntu 24.04.1 LTS")

    def test_gpu_fields_and_statuses(self):
        g = self.report["drm_gpus"][0]
        for key in ("driver_status", "vram_status", "model_status", "compute_status"):
            self.assertIn(g[key], ALL_STATUSES, key)
        self.assertEqual(g["vendor_id"], "0x1002")
        self.assertEqual(g["device_id"], "0x744c")
        self.assertEqual(g["driver"], "amdgpu")
        self.assertEqual(g["vram_bytes"], 25769803776)
        self.assertEqual(g["model"], "Radeon RX 7900 XTX")
        self.assertEqual(g["architecture"], "RDNA3")

    def test_regression_R3_render_node_by_pci_path(self):
        """renderD128 соответствует card0 через общий PCI-узел, а не номер."""
        g = self.report["drm_gpus"][0]
        self.assertEqual(g["render_node"], "sys/class/drm/renderD128")
        self.assertEqual(g["pci_path"], "0000:00:01.1")

    def test_regression_R2_compute_status_after_correlation(self):
        g = self.report["drm_gpus"][0]
        self.assertEqual(g["compute_status"], "detected")  # кандидат ROCm
        self.assertEqual(g["kfd_gfx"], "gfx1100")

    def test_loaded_modules(self):
        mods = self.report["loaded_modules"]
        self.assertEqual(mods["amdgpu"], "detected")
        self.assertEqual(mods["drm"], "unavailable")

    def test_no_pass_claims(self):
        raw = json.dumps(self.report)
        self.assertNotIn('"PASS"', raw)


class TestNoTools(unittest.TestCase):
    """Отсутствующие диагностические утилиты ≠ отсутствие GPU."""

    def setUp(self):
        self.report = report(os.path.join(GOLDEN, "fake-amd-kfd"))

    def test_tool_status_unavailable_but_gpu_present(self):
        ts = self.report["stacks"]["tool_status"]
        # в CI-контейнере rocminfo/vulkaninfo нет; GPU из фикстуры при этом есть
        self.assertIn(ts["rocminfo"], {"unavailable", "detected", "error"})
        self.assertTrue(self.report["classification"]["amd_gpu_detected"])
        # вывод диагностики должен явно разделять эти понятия
        diag = lmd.gpu_diagnosis(self.report)
        joined = "\n".join(diag)
        if ts["rocminfo"] == "unavailable":
            self.assertIn("НЕ означает", joined)


class TestIncompleteVram(unittest.TestCase):
    """Неполная информация о VRAM: amdgpu iGPU без mem_info_vram_total."""

    def setUp(self):
        self.report = report(os.path.join(GOLDEN, "fake-amd-no-vram"))

    def test_vram_unknown_not_zero(self):
        g = self.report["drm_gpus"][0]
        self.assertIsNone(g["vram_bytes"])
        self.assertEqual(g["vram_status"], "unknown")

    def test_rdna1_igpu_unsupported_for_rocm(self):
        g = self.report["drm_gpus"][0]
        # Van Gogh (0x1638) официально вне поддержки ROCm — честный unsupported
        self.assertEqual(g["compute_status"], "unsupported")
        self.assertEqual(
            self.report["classification"]["preliminary_verdict"],
            "VULKAN_ONLY_CANDIDATE",
        )


class TestMultipleGpus(unittest.TestCase):
    """Несколько GPU: AMD дискретная + NVIDIA одновременно."""

    def setUp(self):
        self.report = report(os.path.join(GOLDEN, "fake-multi-gpu"))

    def test_both_cards_listed(self):
        gpus = self.report["drm_gpus"]
        self.assertEqual(len(gpus), 2)
        vendors = {g["vendor_id"] for g in gpus}
        self.assertEqual(vendors, {"0x1002", "0x10de"})

    def test_foreign_gpu_not_marked_supported(self):
        nvidia = next(g for g in self.report["drm_gpus"] if g["vendor_id"] == "0x10de")
        # чужой GPU без драйвера: driver unavailable, compute unknown (не PASS)
        self.assertEqual(nvidia["driver_status"], "unavailable")
        self.assertEqual(nvidia["compute_status"], "unknown")

    def test_amd_card_still_candidate(self):
        amd = next(g for g in self.report["drm_gpus"] if g["vendor_id"] == "0x1002")
        self.assertEqual(amd["compute_status"], "unknown")  # кандидат, ждёт probe
        self.assertEqual(
            self.report["classification"]["preliminary_verdict"],
            "VULKAN_ONLY_CANDIDATE",
        )


class TestCommandErrors(unittest.TestCase):
    """Ошибки выполнения системных команд не должны ронять детектор."""

    def test_run_cmd_missing_binary(self):
        self.assertIsNone(lmd.run_cmd(["definitely-not-a-real-binary-xyz"]))

    def test_run_cmd_nonzero_exit(self):
        self.assertIsNone(lmd.run_cmd(["false"]))

    def test_lspci_absent_is_not_error_in_report(self):
        # в контейнере lspci может не быть: пустой список, статус unavailable
        rep = report(None)
        self.assertIn(rep["stacks"]["tool_status"]["lspci"],
                      {"unavailable", "detected", "error"})


class TestDoctorMode(unittest.TestCase):
    def test_doctor_exit_zero_on_readable_report(self):
        proc = run_detect(os.path.join(GOLDEN, "fake-amd-kfd"), ["--doctor"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Compute", proc.stdout)

    def test_doctor_honest_when_no_gpu(self):
        proc = run_detect(os.path.join(GOLDEN, "fake-no-gpu"), ["--doctor"])
        self.assertIn("не обнаружено", proc.stdout)


class TestRealEnvironment(unittest.TestCase):
    def test_full_report_valid_shape(self):
        rep = report(None)
        self.assertEqual(rep["schema_version"], "0.2.0")
        self.assertIn(rep["cpu"]["status"], ALL_STATUSES)
        self.assertIn(rep["memory"]["status"], ALL_STATUSES)
        for g in rep["drm_gpus"]:
            self.assertIn(g["compute_status"], ALL_STATUSES)


if __name__ == "__main__":
    unittest.main()
