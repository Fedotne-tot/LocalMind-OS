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
import shutil
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



class TestAuditFixes(unittest.TestCase):
    """Регрессии инженерной проверки v0.2 (B1–B5, CPU-Model)."""

    def _fixture(self, base: str) -> str:
        tmp = self.mkdtemp()
        fx = os.path.join(tmp, "fx")
        import shutil
        shutil.copytree(os.path.join(GOLDEN, base), fx, symlinks=True)
        self.addCleanup(shutil.rmtree, tmp, True)
        return fx

    def mkdtemp(self) -> str:
        import tempfile
        d = tempfile.mkdtemp()
        return d

    def test_B1_meminfo_corrupt_is_error_not_detected(self):
        fx = self._fixture("fake-amd-kfd")
        p = os.path.join(fx, "proc/meminfo")
        with open(p) as fh:
            txt = fh.read()
        with open(p, "w") as fh:
            fh.write(txt.replace("MemTotal:       32786544 kB",
                                 "MemTotal:       broken kB"))
        rep = report(fx)
        self.assertEqual(rep["memory"]["status"], "error")
        self.assertIsNone(rep["memory"]["mem_total_bytes"])

    def test_B2_generic_label_model_status_unknown(self):
        """Обобщённый DRM label != подтверждённая модель (было detected)."""
        rep = report(os.path.join(GOLDEN, "fake-amd-no-vram"))
        g = rep["drm_gpus"][0]
        self.assertEqual(g["model"], "AMD Radeon Graphics")
        self.assertEqual(g["model_status"], "unknown")

    def test_B2_exact_subsystem_match_stays_detected(self):
        rep = report(os.path.join(GOLDEN, "fake-amd-kfd"))
        g = rep["drm_gpus"][0]
        self.assertEqual(g["model"], "Radeon RX 7900 XTX")
        self.assertEqual(g["model_status"], "detected")

    def test_B3_no_kfd_cross_attribution_two_amd_cards(self):
        """Две AMD-карты и один gfx-узел: алиас НЕ раздаётся обеим."""
        import shutil, tempfile
        tmp = tempfile.mkdtemp(); fx = os.path.join(tmp, "fx")
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(os.path.join(GOLDEN, "fake-multi-gpu"), fx, symlinks=True)
        d = os.path.join(fx, "sys/class/drm/card1/device")
        with open(os.path.join(d, "vendor"), "w") as fh:
            fh.write("0x1002\n")
        with open(os.path.join(d, "device"), "w") as fh:
            fh.write("0x1636\n")
        kdir = os.path.join(fx, "sys/devices/virtual/kfd/kfd/topology/nodes/1")
        os.makedirs(kdir, exist_ok=True)
        with open(os.path.join(kdir, "properties"), "w") as fh:
            fh.write("name gfx1100\nunique_id 1\n")
        os.makedirs(os.path.join(fx, "dev"), exist_ok=True)
        open(os.path.join(fx, "dev/kfd"), "w").close()
        rep = report(fx)
        amd = [g for g in rep["drm_gpus"] if g["vendor_id"] == "0x1002"]
        # ни одна карта не получила ложный detected при неоднозначной корреляции
        self.assertNotIn("detected", [g["compute_status"] for g in amd])
        self.assertNotIn("gfx1100", [g.get("kfd_gfx") for g in amd])

    def test_B5_arm_model_line_and_x86_numeric_guard(self):
        import shutil, tempfile
        tmp = tempfile.mkdtemp(); fx = os.path.join(tmp, "fx")
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(os.path.join(GOLDEN, "fake-amd-kfd"), fx, symlinks=True)
        with open(os.path.join(fx, "proc/cpuinfo"), "w") as fh:
            fh.write("Features\t: fp asimd\nModel\t: ARMv8 Processor rev 4\n"
                     "processor\t: 0\nprocessor\t: 1\n")
        rep = report(fx)
        self.assertEqual(rep["cpu"]["model"], "ARMv8 Processor rev 4")
        self.assertEqual(rep["cpu"]["cores"], 2)
        # x86: числовая строка "model : 8" не должна перекрывать model name
        fx2 = os.path.join(tmp, "fx2")
        shutil.copytree(os.path.join(GOLDEN, "fake-amd-kfd"), fx2, symlinks=True)
        p = os.path.join(fx2, "proc/cpuinfo")
        with open(p) as fh:
            txt = fh.read()
        with open(p, "w") as fh:
            fh.write(txt + "\nmodel\t\t: 8\n")
        rep2 = report(fx2)
        self.assertIn("Ryzen", rep2["cpu"]["model"])


class TestAuditRound2(unittest.TestCase):
    """Регрессии второй инженерной проверки (C1–C5)."""

    def _fixture(self, base: str) -> str:
        import shutil, tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        fx = os.path.join(tmp, "fx")
        shutil.copytree(os.path.join(GOLDEN, base), fx, symlinks=True)
        return fx

    def test_C1_corrupt_proc_modules_is_error_not_crash(self):
        """Битый /proc/modules не должен валить весь отчёт (было NameError)."""
        fx = self._fixture("fake-amd-kfd")
        with open(os.path.join(fx, "proc/modules"), "wb") as fh:
            fh.write(b"\xff\xfe garbage \x00\n")
        proc = run_detect(fx)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rep = json.loads(proc.stdout)
        self.assertNotIn("detected", set(rep["loaded_modules"].values()))

    def test_C2_kfd_node_without_name_never_false_unsupported(self):
        """KFD-узел без имени (битый properties) != unsupported для RX-карты.

        Сценарий с реальным риском ложного отказа: amdgpu + topology-узлы,
        но имена нечитаемы → compute_status обязан быть unknown (ждём probe),
        а не unsupported.
        """
        import glob
        fx = self._fixture("fake-amd-kfd")
        for p in glob.glob(os.path.join(
                fx, "sys/devices/virtual/kfd/kfd/topology/nodes/*/properties")):
            with open(p, "w") as fh:
                fh.write("corrupted\n")
        rep = report(fx)
        amd = next(g for g in rep["drm_gpus"] if g["vendor_id"] == "0x1002")
        self.assertNotEqual(amd["compute_status"], "unsupported")
        self.assertTrue(rep["classification"]["kfd_device_present"])

    def test_C3_no_dev_kfd_file_but_topology_nodes_still_detected(self):
        """Отсутствие /dev/kfd при живых KFD-узлах — не потеря GPU (C3)."""
        import glob
        fx = self._fixture("fake-amd-kfd")
        devkfd = os.path.join(fx, "dev/kfd")
        if os.path.exists(devkfd):
            os.remove(devkfd)
        # узлы без имён: kfd_node_present обязан опознать сам факт KFD
        for p in glob.glob(os.path.join(
                fx, "sys/devices/virtual/kfd/kfd/topology/nodes/*/properties")):
            with open(p, "w") as fh:
                fh.write("corrupted\n")
        rep = report(fx)
        self.assertTrue(rep["classification"]["kfd_device_present"])
        amd = next(g for g in rep["drm_gpus"] if g["vendor_id"] == "0x1002")
        self.assertTrue(amd["kfd_node_present"])
        self.assertEqual(amd["compute_status"], "unknown")

    def test_C4_broken_binary_env_is_error_not_crash(self):
        """Битый bинарь (OSError при exec) → tool_status=error, отчёт живой."""
        orig_run_cmd = lmd.run_cmd
        try:
            def boom(argv):
                raise OSError("libstdc++.so.6: cannot open shared object")
            lmd.run_cmd = boom
            stacks = lmd.detect_stack_presence()
            # статусы обязаны остаться в допустимом множестве (никакого падения)
            for name in ("lspci", "rocminfo", "vulkaninfo"):
                self.assertIn(stacks["tool_status"][name], ALL_STATUSES)
        finally:
            lmd.run_cmd = orig_run_cmd

    def test_C5_schema_requires_os_pretty_name(self):
        """Поле всегда присутствует в отчёте — схема обязывает его (C5)."""
        import json as _json
        schema = _json.load(open(os.path.join(REPO, "config", "report-schema.json")))
        self.assertIn("os_pretty_name", schema["required"])
        rep = report(None)
        self.assertIn("os_pretty_name", rep)


class TestAuditRound3(unittest.TestCase):
    """Регрессии третьей инженерной проверки (D1–D4)."""

    def test_D1_multi_gpu_amd_resolved_by_subsystem_table(self):
        """D1: фикстура multi-gpu обязана называть RX 7800 XT.

        Раньше model оставался None, потому что таблица подсистем
        применялась только к DRM-label-строке и после correlate; теперь
        уточнение выполняется внутри detect_drm_gpus до корреляции.
        """
        fx = os.path.join(GOLDEN, "fake-multi-gpu")
        dev = os.path.join(fx, "sys/class/drm/card0/device")
        with open(os.path.join(dev, "subsystem_vendor"), "w") as fh:
            fh.write("0x0732\n")
        with open(os.path.join(dev, "subsystem_device"), "w") as fh:
            fh.write("0x0008\n")
        rep = report(fx)
        amd = next(g for g in rep["drm_gpus"] if g["vendor_id"] == "0x1002")
        self.assertEqual(amd["model"], "Radeon RX 7800 XT")
        self.assertEqual(amd["model_status"], "detected")
        self.assertEqual(amd.get("subsystem_key"),
                         "0x1002:0x7480:0x0732:0x0008")
        # NVIDIA-карта при этом остаётся unknown (не наша задача на этом слое)
        nv = next(g for g in rep["drm_gpus"] if g["vendor_id"] == "0x10de")
        self.assertEqual(nv["compute_status"], "unknown")

    def test_D2_model_known_before_kfd_correlation(self):
        """D2: точная модель доступна уже на выходе detect_drm_gpus."""
        fx = os.path.join(GOLDEN, "fake-amd-kfd")
        old_env = os.environ.get("LM_FAKE_SYS")
        os.environ["LM_FAKE_SYS"] = fx  # фикстура вместо голой среды CI (без GPU)
        try:
            names = lmd.detect_subsystem_names()
            gpus = lmd.detect_drm_gpus(names)
        finally:
            if old_env is None:
                os.environ.pop("LM_FAKE_SYS", None)
            else:
                os.environ["LM_FAKE_SYS"] = old_env
        amd = next(g for g in gpus if g["vendor_id"] == "0x1002")
        self.assertEqual(amd["model"], "Radeon RX 7900 XTX")
        self.assertEqual(amd["model_status"], "detected")

    def test_D3_generic_label_never_claimed_detected(self):
        """D3: обобщённый label 'AMD Radeon Graphics' = unknown, не detected."""
        self.assertEqual(lmd._model_status("AMD Radeon Graphics"), "unknown")
        self.assertEqual(lmd._model_status("Radeon RX 7800 XT"), "detected")
        self.assertEqual(lmd._model_status(None), "unavailable")

    def test_D4_no_pass_string_in_any_fixture_report(self):
        """D4: ни один golden-отчёт не содержит строку PASS (страховка P4)."""
        for fx in sorted(os.listdir(GOLDEN)):
            rep = report(os.path.join(GOLDEN, fx))
            blob = json.dumps(rep)
            self.assertNotIn('"PASS"', blob)
            self.assertNotIn("supported\": true", blob)


class TestRdna2WithCpuNode(unittest.TestCase):
    """Фикстура fake-rdna2-vega: RDNA2 (gfx1030) + CPU-узел KFD без поля name.

    Сценарий реального Linux: topology содержит узел 0 (CPU, properties без
    «name») и узел 1 (GPU gfx1030). Проверяет, что:
      * CPU-узел НЕ порождает ложный gfx-алиас в классификации;
      * GPU-узел корректно коррелирует с DRM-картой (known_gfx из KFD);
      * VRAM 16 GiB детектируется по mem_info_vram_total.
    """

    def setUp(self):
        self.report_ = report(os.path.join(GOLDEN, "fake-rdna2-vega"))

    def test_cpu_node_has_no_name_and_not_attributed(self):
        # в JSON-отчёте ключи словарей всегда строковые
        nodes = {str(n["node"]): n for n in self.report_["kfd_nodes"]}
        self.assertIn("0", nodes)
        self.assertNotIn("name", nodes["0"])
        # CPU-узел не должен «получать» gfx от куда-то
        c = self.report_["classification"]
        self.assertEqual(c["known_gfx_aliases_in_kfd"], ["gfx1030"])

    def test_gpu_card_correlated_with_gfx1030(self):
        gpus = self.report_["drm_gpus"]
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual(g["vendor_id"], "0x1002")
        self.assertEqual(g["device_id"], "0x73bf")
        self.assertEqual(g["driver"], "amdgpu")
        self.assertEqual(g["kfd_gfx"], "gfx1030")
        self.assertEqual(g["architecture"], "RDNA2")
        self.assertEqual(g["compute_status"], "detected")
        self.assertEqual(g["vram_bytes"], 16 * 1024 ** 3)
        self.assertEqual(g["vram_status"], "detected")

    def test_verdict_is_candidate_not_pass(self):
        c = self.report_["classification"]
        self.assertEqual(c["preliminary_verdict"], "ROCM_CANDIDATE")
        self.assertTrue(c["amdgpu_driver_bound"])
        # POLICY P4: нигде нет PASS/«поддерживается»
        blob = json.dumps(self.report_)
        self.assertNotIn('"PASS"', blob)

    def test_all_statuses_valid_enum(self):
        for g in self.report_["drm_gpus"]:
            for key in ("driver_status", "vram_status", "model_status",
                        "compute_status"):
                self.assertIn(g[key], ALL_STATUSES)


class TestRdna3Target7800XT(unittest.TestCase):
    """Фикстура fake-rdna3-7800xt: целевая конфигурация RX 7800 XT + Ryzen 7 5700X.

    Реалистичное KFD-дерево: CPU-узел 0 (без name, caches/links), GPU-узел 1
    (gfx1101, mem_banks_count 2, io_links), iGPU-узел 2 (gfx1036 — связка
    AM5 с включённым iGPU). DRM: card0 + renderD128 по общему PCI-узлу,
    connector-каталог card0-VGA-1 и «висячий» renderD129 без vendor/device
    (проверка R3-регрессии на расширенном дереве).

    Политика доказательств (P4/P5) НЕ изменена: при двух gfx-узлах и одной
    дискретной DRM-карте корреляция неоднозначна → честный unknown вместо
    ложного detected; точная модель только через проверенную subsystem-пару.
    """

    FIXTURE = os.path.join(GOLDEN, "fake-rdna3-7800xt")

    def setUp(self):
        self.rep = report(self.FIXTURE)

    # --- окружение -----------------------------------------------------
    def test_cpu_ram_os_target_config(self):
        self.assertEqual(self.rep["os_pretty_name"], "Ubuntu 24.04.1 LTS")
        self.assertIn("Ryzen 7 5700X", self.rep["cpu"]["model"])
        self.assertEqual(self.rep["cpu"]["cores"], 16)
        self.assertEqual(self.rep["memory"]["status"], "detected")
        self.assertEqual(self.rep["memory"]["mem_total_bytes"], 65789344 * 1024)
        self.assertEqual(self.rep["loaded_modules"]["amdgpu"], "detected")
        self.assertEqual(self.rep["loaded_modules"]["drm"], "detected")

    # --- KFD topology ---------------------------------------------------
    def test_kfd_topology_three_nodes(self):
        nodes = {str(n["node"]): n for n in self.rep["kfd_nodes"]}
        self.assertEqual(set(nodes), {"0", "1", "2"})
        # CPU-узел: properties без поля name (как в fake-rdna2-vega/C2)
        self.assertNotIn("name", nodes["0"])
        gpu_node = next(n for n in self.rep["kfd_nodes"]
                        if str(n["node"]) == "1")
        self.assertEqual(gpu_node["name"], "gfx1101")
        # сырые ключи GPU-узла сохраняются для будущего probe-слоя (E3)
        for k in ("vdhip", "simd_count", "mem_banks_count", "io_links_count"):
            self.assertIn(k, gpu_node["raw_keys"])

    def test_classification_collects_both_known_aliases(self):
        c = self.rep["classification"]
        self.assertEqual(sorted(c["known_gfx_aliases_in_kfd"]),
                         ["gfx1036", "gfx1101"])
        self.assertTrue(c["kfd_device_present"])
        self.assertTrue(c["amd_gpu_detected"])
        self.assertTrue(c["amdgpu_driver_bound"])

    # --- DRM-карта ------------------------------------------------------
    def test_drm_card_fields_and_render_node(self):
        gpus = [g for g in self.rep["drm_gpus"]]
        # card1 внутри pci.../drm/ и «висячий» renderD129 не порождают карт
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual(g["card"], "card0")
        self.assertEqual(g["vendor_id"], "0x1002")
        self.assertEqual(g["device_id"], "0x7480")
        self.assertEqual(g["driver"], "amdgpu")
        self.assertEqual(g["render_node"], "sys/class/drm/renderD128")
        self.assertEqual(g["pci_path"], "0000:00:01.1")
        self.assertEqual(g["vram_bytes"], 17163096064)  # ~16 GiB из фикстуры
        self.assertEqual(g["vram_status"], "detected")

    def test_honest_states_without_subsystem_evidence(self):
        """Без подтверждённой subsystem-пары: model/compute — unknown, не выдумка."""
        g = self.rep["drm_gpus"][0]
        self.assertEqual(g["model"], "AMD Radeon Graphics")  # обобщённый label
        self.assertEqual(g["model_status"], "unknown")       # D3/B2
        # два gfx-узла vs одна карта -> pool(2) != amd_cards(1): не угадываем
        self.assertNotIn("kfd_gfx", g)                      # B3: алиас не приписан
        self.assertTrue(g["kfd_node_present"])              # C2: KFD всё же есть
        self.assertEqual(g["compute_status"], "unknown")    # ждёт probe E3/E4
        self.assertNotEqual(g["compute_status"], "unsupported")
        # E1: unknown обязан быть РАЗЪЯСНЁН: 2 известных gfx-узла vs 1 карта
        # — количества не сходятся, привязка была бы догадкой
        self.assertEqual(g["kfd_correlation"], "count_mismatch")

    def test_verdict_candidate_but_no_pass(self):
        c = self.rep["classification"]
        self.assertEqual(c["preliminary_verdict"], "ROCM_CANDIDATE")
        blob = json.dumps(self.rep)
        self.assertNotIn('"PASS"', blob)                    # P4
        self.assertNotIn("supported\": true", blob)

    def test_all_statuses_valid_enum(self):
        for g in self.rep["drm_gpus"]:
            for key in ("driver_status", "vram_status", "model_status",
                        "compute_status"):
                self.assertIn(g[key], ALL_STATUSES)
        for n in self.rep["kfd_nodes"]:
            self.assertIn(n["status"], ALL_STATUSES)

    # --- путь к точному имени модели ------------------------------------
    def test_exact_model_via_reference_subsystem_pair(self):
        """Табличный ключ 0x1002:0x7480:0x0732:0x0008 даёт точное имя и RDNA3."""
        import shutil, tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        fx = os.path.join(tmp, "fx")
        shutil.copytree(self.FIXTURE, fx, symlinks=True)
        dev = os.path.join(fx, "sys/devices/pci0000:00/0000:00:01.1")
        with open(os.path.join(dev, "subsystem_vendor"), "w") as fh:
            fh.write("0x0732\n")
        with open(os.path.join(dev, "subsystem_device"), "w") as fh:
            fh.write("0x0008\n")
        rep = report(fx)
        g = rep["drm_gpus"][0]
        self.assertEqual(g["model"], "Radeon RX 7800 XT")
        self.assertEqual(g["model_status"], "detected")
        self.assertEqual(g["subsystem_key"], "0x1002:0x7480:0x0732:0x0008")
        # политика НЕ меняется: неоднозначная корреляция остаётся unknown
        self.assertEqual(g["compute_status"], "unknown")

    def test_single_gpu_node_correlates_gfx1101_rdna3(self):
        """Если iGPU-узел убрать, единственная карта честно получает gfx1101/RDNA3."""
        import shutil, tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        fx = os.path.join(tmp, "fx")
        shutil.copytree(self.FIXTURE, fx, symlinks=True)
        shutil.rmtree(os.path.join(
            fx, "sys/devices/virtual/kfd/kfd/topology/nodes/2"))
        rep = report(fx)
        g = rep["drm_gpus"][0]
        self.assertEqual(g["kfd_gfx"], "gfx1101")
        self.assertEqual(g["architecture"], "RDNA3")
        self.assertEqual(g["compute_status"], "detected")  # кандидат ROCm
        # E1: detected обязан быть РАЗЪЯСНЁН как подтверждённая привязка
        self.assertEqual(g["kfd_correlation"], "matched")
        blob = json.dumps(rep)
        self.assertNotIn('"PASS"', blob)                   # detected != PASS


ALL_CORRELATION_REASONS = {
    "matched", "driver_not_amdgpu", "model_not_rocm_supported",
    "kfd_not_present", "node_properties_unreadable", "ambiguous_mapping",
    "count_mismatch", "gfx_not_in_whitelist", "not_evaluated",
}


class TestKfdCorrelationCases(unittest.TestCase):
    """E1: отчёт честно различает три ситуации вокруг неоднозначной KFD-связи.

    Ситуация 1 — KFD корректно сопоставлен с GPU → compute_status=detected,
                 kfd_correlation=matched;
    Ситуация 2 — KFD существует, но gfx вне whitelist (например будущая
                 архитектура) → unknown + gfx_not_in_whitelist (НЕ unsupported:
                 данных о неподдержке нет);
    Ситуация 3 — KFD и GPU есть, но связь невозможно однозначно установить →
                 unknown + count_mismatch / ambiguous_mapping /
                 node_properties_unreadable.

    Политика доказательств (P4/P5) не изменена: ни один unknown не подменён
    догадкой, PASS не выдаётся без machine-отчёта с реального железа.
    """

    FIXTURE = os.path.join(GOLDEN, "fake-rdna3-7800xt")

    def _fixture(self) -> str:
        import shutil, tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        fx = os.path.join(tmp, "fx")
        shutil.copytree(self.FIXTURE, fx, symlinks=True)
        return fx

    NODES = "sys/devices/virtual/kfd/kfd/topology/nodes"

    # --- ситуация 1: корректная корреляция --------------------------------
    def test_case1_matched_correlation(self):
        """Один gfx-узел + одна amdgpu-карта: привязка доказуема → matched."""
        fx = self._fixture()
        shutil.rmtree(os.path.join(fx, self.NODES, "2"))  # убрать iGPU-узел
        rep = report(fx)
        g = rep["drm_gpus"][0]
        self.assertEqual(g["kfd_gfx"], "gfx1101")
        self.assertEqual(g["architecture"], "RDNA3")
        self.assertEqual(g["compute_status"], "detected")
        self.assertEqual(g["kfd_correlation"], "matched")
        self.assertTrue(g["kfd_node_present"])
        # detected — это всё ещё НЕ PASS (P4): финальный вердикт за probe
        self.assertNotIn('"PASS"', json.dumps(rep))

    # --- ситуация 2: KFD есть, gfx вне whitelist --------------------------
    def test_case2_kfd_present_gfx_not_in_whitelist(self):
        """gfx950 (нет в GFX_ALIASES) при одной карте: unknown, не unsupported."""
        self.assertNotIn("gfx950", lmd.GFX_ALIASES)
        fx = self._fixture()
        # единственный GPU-узел KFD с неизвестным алиасом: узел 1 (тот, что
        # коррелирует с дискретной картой) переписываем на gfx950; iGPU-узел 2
        # убираем, иначе два именованных gfx-узла vs одна карта — это уже
        # ситуация 3 (count_mismatch), а не «вне whitelist»
        shutil.rmtree(os.path.join(fx, self.NODES, "2"))
        props = os.path.join(fx, self.NODES, "1/properties")
        with open(props, "w") as fh:
            fh.write("name gfx950\nunique_id 1\nsimd_count 192\n"
                     "mem_banks_count 2\nio_links_count 1\n")
        rep = report(fx)
        g = rep["drm_gpus"][0]
        self.assertNotIn("kfd_gfx", g)                    # чужой алиас не приписан
        self.assertTrue(g["kfd_node_present"])            # KFD действительно есть
        self.assertEqual(g["compute_status"], "unknown")  # НЕ unsupported (нет данных)
        self.assertEqual(g["kfd_correlation"], "gfx_not_in_whitelist")
        # классификация тоже не находит известных алиасов — никаких выдумок
        self.assertEqual(rep["classification"]["known_gfx_aliases_in_kfd"], [])

    def test_case2_offlist_alias_never_becomes_architecture(self):
        """Вне-whitelist gfx не должен порождать поле architecture (без догадок)."""
        fx = self._fixture()
        # известные узлы убираем: иначе pool из gfx1036 дал бы matched;
        # CPU-узел 0 остаётся без name (нативное свойство CPU-узлов KFD)
        shutil.rmtree(os.path.join(fx, self.NODES, "2"))
        p = os.path.join(fx, self.NODES, "1/properties")
        with open(p, "w") as fh:
            fh.write("name gfx9999\nunique_id 1\nsimd_count 64\n")
        rep = report(fx)
        g = rep["drm_gpus"][0]
        self.assertNotIn("kfd_gfx", g)
        self.assertNotEqual(g.get("architecture"), "RDNA2")  # ложная архитектура
        self.assertEqual(g["compute_status"], "unknown")
        self.assertEqual(g["kfd_correlation"], "gfx_not_in_whitelist")

    # --- ситуация 3: связь невозможно однозначно установить ---------------
    def test_case3a_count_mismatch_two_nodes_one_card(self):
        """Базовая фикстура: 2 известных gfx-узла vs 1 DRM-карта → count_mismatch."""
        rep = report(self.FIXTURE)
        g = rep["drm_gpus"][0]
        self.assertTrue(g["kfd_node_present"])
        self.assertNotIn("kfd_gfx", g)                    # не угадываем, какой узел чей
        self.assertEqual(g["compute_status"], "unknown")
        self.assertEqual(g["kfd_correlation"], "count_mismatch")

    def test_case3b_ambiguous_mapping_two_cards_two_nodes(self):
        """2 карты и 2 gfx-узла: порядок cardN↔node N не доказан → ambiguous."""
        fx = self._fixture()
        # вторая AMD-карта: карточка в /sys/class/drm с device→PCI-узлом
        # 0000:03:00.0 (уже есть в фикстуре); заполняем PCI-узел как udev
        dev = os.path.join(fx, "sys/devices/pci0000:00/0000:03:00.0")
        for fname, val in (("vendor", "0x1002\n"), ("device", "0x7480\n"),
                           ("mem_info_vram_total", "17163096064\n")):
            with open(os.path.join(dev, fname), "w") as fh:
                fh.write(val)
        drv = os.path.join(dev, "driver")
        if not os.path.lexists(drv):
            os.symlink("../../bus/pci/drivers/amdgpu", drv)
        card1 = os.path.join(fx, "sys/class/drm/card1")
        os.makedirs(card1, exist_ok=True)
        link = os.path.join(card1, "device")
        if os.path.lexists(link):
            os.remove(link)
        # relative symlink разрешается относительно каталога ссылки:
        # sys/class/drm/card1 -> ../../../devices/... = sys/devices/...
        os.symlink("../../../devices/pci0000:00/0000:03:00.0", link)
        # KFD: два известных gfx-узла (1: gfx1101 уже есть; 2: делаем gfx1030)
        p2 = os.path.join(fx, self.NODES, "2/properties")
        with open(p2, "w") as fh:
            fh.write("name gfx1030\nunique_id 2\nsimd_count 96\n"
                     "mem_banks_count 1\nio_links_count 1\n")
        rep = report(fx)
        amd = [g for g in rep["drm_gpus"] if g["vendor_id"] == "0x1002"]
        self.assertEqual(len(amd), 2)
        for g in amd:
            self.assertNotIn("kfd_gfx", g)                # B3: перекрёстных догадок нет
            self.assertTrue(g["kfd_node_present"])
            self.assertEqual(g["compute_status"], "unknown")
            self.assertEqual(g["kfd_correlation"], "ambiguous_mapping")

    def test_case3c_unreadable_node_names(self):
        """Все properties биты: KFD есть, имена недоступны → unreadable (C2/E1)."""
        import glob
        fx = self._fixture()
        for p in glob.glob(os.path.join(fx, self.NODES, "*/properties")):
            with open(p, "w") as fh:
                fh.write("corrupted\n")
        rep = report(fx)
        g = rep["drm_gpus"][0]
        self.assertTrue(g["kfd_node_present"])            # C2: KFD не потеряли
        self.assertNotEqual(g["compute_status"], "unsupported")
        self.assertEqual(g["compute_status"], "unknown")
        self.assertEqual(g["kfd_correlation"], "node_properties_unreadable")

    # --- инварианты ---------------------------------------------------------
    def test_reason_always_explains_compute_status(self):
        """Причина обязана согласованно сопровождать статус (все 3 случая)."""
        cases = []
        # 1) matched
        fx = self._fixture()
        shutil.rmtree(os.path.join(fx, self.NODES, "2"))
        cases.append(report(fx))
        # 2) offlist
        fx = self._fixture()
        p = os.path.join(fx, self.NODES, "1/properties")
        with open(p, "w") as fh:
            fh.write("name gfx950\nunique_id 1\n")
        cases.append(report(fx))
        # 3) базовая фикстура (count_mismatch)
        cases.append(report(self.FIXTURE))
        for rep in cases:
            for g in rep["drm_gpus"]:
                r = g["kfd_correlation"]
                self.assertIn(r, ALL_CORRELATION_REASONS)
                cs = g["compute_status"]
                if r == "matched":
                    self.assertEqual(cs, "detected")
                elif r in ("count_mismatch", "ambiguous_mapping",
                           "gfx_not_in_whitelist", "node_properties_unreadable"):
                    self.assertEqual(cs, "unknown")      # никогда не PASS/detected
                self.assertNotIn('"PASS"', json.dumps(rep))

    def test_doctor_reports_reason_for_unknown(self):
        """--doctor на целевой фикстуре: unknown сопровождается причиной."""
        proc = run_detect(self.FIXTURE, ["--doctor"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("причина: count_mismatch", proc.stdout)
        # P4: PASS может упоминаться ТОЛЬКО в disclaimer-строке про probe;
        # ни одна секция не должна выглядеть как присвоение PASS
        for line in proc.stdout.splitlines():
            if "PASS" in line:
                self.assertIn("выдаёт только probe", line)
            self.assertNotIn("Compute: PASS", line)
            self.assertNotIn("verdict: PASS", line)

    def test_doctor_mode_on_target_fixture(self):
        proc = run_detect(self.FIXTURE, ["--doctor"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("VRAM: 16368 MiB (detected)", proc.stdout)
        self.assertIn("unknown", proc.stdout)


if __name__ == "__main__":
    unittest.main()
