"""Unit-тесты localmind-detect v0 (Этап E1).

Тесты не требуют GPU и root: используют фикстуры tests/golden/fake-*
через подмену корня LM_FAKE_SYS. Только stdlib (unittest) — зависимостей нет.

Запуск:  make test   или   python3 -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT = os.path.join(REPO, "scripts", "localmind-detect.py")
GOLDEN = os.path.join(REPO, "tests", "golden")
SCHEMA = os.path.join(REPO, "config", "report-schema.json")


def run_detect(fake_sys: str | None) -> dict:
    env = dict(os.environ)
    if fake_sys:
        env["LM_FAKE_SYS"] = fake_sys
    else:
        env.pop("LM_FAKE_SYS", None)
    proc = subprocess.run(
        [sys.executable, DETECT, "--json"], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestSchemaShape(unittest.TestCase):
    """Минимальная проверка формы отчёта без внешних jsonschema-библиотек."""

    def test_required_top_level_keys(self):
        report = run_detect(os.path.join(GOLDEN, "fake-amd-kfd"))
        with open(SCHEMA) as fh:
            schema = json.load(fh)
        for key in schema["required"]:
            self.assertIn(key, report, f"missing {key}")
        self.assertEqual(report["schema_version"], "0.1.0")

    def test_policy_block_immutable(self):
        report = run_detect(os.path.join(GOLDEN, "fake-amd-kfd"))
        self.assertTrue(report["policy"]["gpu_strict_default"])
        self.assertEqual(report["policy"]["silent_cpu_fallback"], "forbidden")


class TestFakeAmdKfd(unittest.TestCase):
    def setUp(self):
        self.report = run_detect(os.path.join(GOLDEN, "fake-amd-kfd"))

    def test_amd_gpu_detected(self):
        gpus = self.report["drm_gpus"]
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["vendor_id"], "0x1002")
        self.assertEqual(gpus[0]["device_id"], "0x744c")
        self.assertEqual(gpus[0]["driver"], "amdgpu")
        self.assertEqual(gpus[0]["vram_bytes"], 25769803776)

    def test_kfd_node_parsed(self):
        nodes = self.report["kfd_nodes"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["name"], "gfx1100")

    def test_verdict_rocm_candidate(self):
        c = self.report["classification"]
        self.assertTrue(c["amd_gpu_detected"])
        self.assertTrue(c["amdgpu_driver_bound"])
        self.assertTrue(c["kfd_device_present"])
        self.assertIn("gfx1100", c["known_gfx_aliases_in_kfd"])
        self.assertEqual(c["preliminary_verdict"], "ROCM_CANDIDATE")

    def test_no_false_pass_claims(self):
        """Детектор НЕ выдаёт PASS — только предварительный вердикт (POLICY P4)."""
        raw = json.dumps(self.report)
        self.assertNotIn('"PASS"', raw)


class TestFakeNoGpu(unittest.TestCase):
    def setUp(self):
        self.report = run_detect(os.path.join(GOLDEN, "fake-no-gpu"))

    def test_honest_empty_report(self):
        self.assertEqual(self.report["drm_gpus"], [])
        self.assertEqual(self.report["classification"]["preliminary_verdict"], "NO_AMD_GPU")
        self.assertFalse(self.report["classification"]["kfd_device_present"])


class TestRealEnvironment(unittest.TestCase):
    def test_runs_without_root_and_reads_only(self):
        """На текущей машине (GPU может отсутствовать) детектор обязан
        завершиться успешно и не выдумывать устройства."""
        report = run_detect(None)
        self.assertIsInstance(report["drm_gpus"], list)
        verdicts = {"ROCM_CANDIDATE", "VULKAN_ONLY_CANDIDATE", "NO_AMD_GPU"}
        self.assertIn(report["classification"]["preliminary_verdict"], verdicts)


if __name__ == "__main__":
    unittest.main()
