# Probe-слой E3: точная привязка GPU к KFD — минимальный дизайн

Дата: 2026-10-19. Статус: **дизайн, реализация не начата**. Ни ioctl, ни
обращение к устройствам в этом документе не выполнялись и пока не кодятся.
Код, схема, тесты и политика P4/P5 этим документом **не меняются**.

Документ продолжает серию: ARCHITECTURE.md (этапы E0–E9), POLICY.md (P1–P5),
COMPATIBILITY.md (матрица статусов). Термины `detected/unknown/unsupported/
error/unavailable` — из scripts/localmind-detect.py (STATUS_*).

---

## 1. Проблема: что уже доказано и где предел sysfs-корреляции

Детектор v0.2 (read-only, /sys + /proc + lspci + lsmod) умеет:

| Данные | Источник в коде | Статус |
|---|---|---|
| DRM-карты: cardN, vendor/device ID, driver-symlink, PCI-путь, renderD по общему PCI-узлу | `detect_drm_gpus()` | факт |
| Модель карты по subsystem-паре (hwdata pci.ids) | `AMD_SUBSYSTEM_NAMES`, D2 | факт (для известных пар) |
| VRAM: mem_info_vram_total (+ null-валидность схемы) | `detect_drm_gpus()` | факт/unknown честно |
| KFD topology: узлы, name(gfx), cpu_count, raw_keys | `detect_kfd_nodes()` | факт |
| mem_banks/io_links/caches фикстур golden (size/width/type/to_node) | файлы на диске | **не читаются кодом** (см. §3) |
| Классификация + diagnostic-поле `kfd_correlation` с причиной | `correlate_kfd_to_gpus()`, E1 | факт |
| Сводка для doctor | `summarize_kfd_correlation()`, E7 | факт |

Предел (зафиксирован кодом и тестами E1): без обращения к ядру связь
`cardN/renderD M ↔ KFD node N` доказывается только эвристиками. Три ситуации
различаются честно, но две из них остаются `unknown`:

1. **matched** — единственный gfx-узел ↔ единственная AMD-карта → `detected`
   (это всё ещё НЕ PASS по поддержке, только ROCm-кандидат; P4).
2. **gfx_not_in_whitelist** — KFD есть, gfx вне GFX_ALIASES → unknown.
3. **ambiguous_mapping / count_mismatch** — карт и gfx-узлов не сходится
   (например, целевая fake-rdna3-7800xt: dGPU gfx1101 + iGPU gfx1036 при
   одной видимой DRM-карте) → unknown. Подменить догадкой запрещено.

Именно ситуация 3 на реальном Ryzen+Radeon (iGPU активен в BIOS/CRAT) —
главная цель E3: превратить «неоднозначно» в «доказано» или остаться
честным unknown, если доказательств нет.

## 2. Каких доказательств не хватает

Чтобы утверждать «карта X = узел Y», нужен хотя бы один **канонический
идентификатор**, общий для DRM-объекта и KFD-узла:

- **PCI BDF (bus:device.function)** — самый прямой мост: DRM-устройство уже
  знает `pci_path` (`0000:03:00.0` в фикстуре); KFD GPU-узел описывает тот
  же PCI через поле `local_id` properties (формат `<domain>:<bus>:<dev>.<func>`
  либо bus-номер) и/или io_link к CPU-узлу. Сейчас детектор `local_id`
  **не сохраняет** (properties парсятся, но в узел кладываются только
  name/cpu_count/raw_keys) — это дыра, закрываемая в E3-lite без ioctl (§4.0).
- **drm_render_minor** — в свойствах KFD-узла (amdgpu выставляет номер
  renderD этой карты). Если читается — прямая связка dev↔node без привилегий.
- **ioctl AMDKFD_IOC_GET_MEMORY_TOPOLOGY / GET_PROCESS_INFO / crat** —
  канонический путь rocminfo/ROCr runtime: полный CRAT-дамп с типами
  процессора (GPU/CPU), PCIe-атомами и ссылками. Даёт ответ даже когда
  sysfs-представление обрезано (старые ядра, lockdown).

Отдельно E3 обязан получить то, чего sysfs-детектор принципиально не знает:

- работает ли открытый `/dev/kfd` + `AMDKFD_IOC_ACQUIRE_VM`/queue-создание
  (живность compute-стека, а не наличие файлов);
- версию KFD-IOCTL (`AMDKFD_IOC_GET_VERSION`);
- доступность карты из render-node (для будущего Vulkan-пути E4 это вход).

## 3. Интерфейсы ядра, требуемые для каждого шага

| Шаг | Интерфейс | Файл/узел | Права | Побочные эффекты |
|---|---|---|---|---|
| 3.0 дочитать sysfs (local_id, drm_render_minor, mem_banks, links) | обычный read | `/sys/devices/virtual/kfd/kfd/topology/nodes/N/{properties,mem_banks/*/properties,links/*/properties}` | нет (world-readable) | нет |
| 3.1 версия KFD | `ioctl(fd, AMDKFD_IOC_GET_VERSION)` = IOIOR('K', 0x00, uint32) | `/dev/kfd` (обычно crw-rw-rw-, группа render/video) | открытие O_RDWR может требовать членства в группе `render`/`video`; на Ubuntu — в группе `render` | нет (только чтение версии) |
| 3.2 дамп CRAT-топологии | `ioctl(fd, AMDKFD_IOC_GET_MEMORY_TOPOLOGY)` = IOIOR('K', 0x0A, struct{ptr,size}) | `/dev/kfd` | как 3.1 | нет; буфер аллоцируется в userspace |
| 3.3 (опционально, ПОЗЖЕ) проверка живости compute | `AMDKFD_IOC_CREATE_QUEUE` с tiny-queue | `/dev/kfd` | те же | создаёт очередь в GPU — **выход за read-only**, в минимальный E3 НЕ входит |
| 3.4 render-узел | stat/chgrp сверка `/dev/dri/renderD*` с drm_render_minor из 3.0 | `/dev/dri/*` | чтение stat — без прав | нет |

Константы ioctl: брать из upstream-заголовков ROCr kernel-headers
(`drivers/gpu/drm/amd/include/linux/kfd_ioctl.h`, лицензия GPL-2.0 —
используется только ABI-константная таблица, как делает сам rocminfo;
собственный Python-код лицензии не наследует, но константы обязаны быть
сверены с версией ядра стенда — см. §6).

**Права доступа (реальность Ubuntu LTS):** `/dev/kfd` имеет группу `render`
(udev-правило amdgpu). Пользователь без группы `render` получит EACCES —
E3 обязан классифицировать это как отдельный честный исход
`permission_denied` (unknown), а НЕ как «KFD сломан» и НЕ как FAIL.
Никакого setuid/sudo-эскалации каркас не предлагает.

## 4. Минимальная архитектура E3 (по частям, каждая съёмная)

Принцип: **E3-lite сначала без ioctl** — максимум пользы от уже разрешённого
чтения; ioctl добавляется вторым шагом и только при наличии железа.

### 4.0 E3-lite: дополнить sysfs-корреляцию (без новых привилегий)
- `detect_kfd_nodes()` сохраняет распарсенные `local_id`, `drm_render_minor`,
  `unique_id` (сейчас теряются в raw_keys) + списки mem_banks (size/width) и
  links (type/from_node/to_node/weight) — данные УЖЕ есть в golden-фикстурах.
- Новый чистый алгоритм `correlate_by_pci_or_minor()`:
  карта↔узел, если `pci_path == local_id` ИЛИ `renderD M == drm_render_minor`.
  Совпадение одного канонического поля → `matched` (evidence=`pci_bdf` или
  `render_minor`). Поля противоречат → новый диагноз `conflict` (unknown).
  Ничего не совпало → прежние диагнозы E1 (никаких ослаблений!).
- Порядок вызова: каноническая корреляция ДО эвристической; эвристика
  применяется только к картам/узлам, не имеющим канонического доказательства.
- Схема: additive-поля (`kfd_evidence`, `local_id`, ...) — старые отчёты
  остаются валидны (как сделано с kfd_correlation_summary в E7).

### 4.1 E3-ioctl: подтверждение через /dev/kfd (только на стенде)
Модуль `probe-kfd` (отдельный исполняемый файл, НЕ часть детектора по
умолчанию — детектор остаётся строго read-only без открытия устройств):

```
lm probe kfd [--json]            # явный запуск человеком/CI со стенда
  ├─ open(/dev/kfd, O_RDWR)      # EACCES/ENOENT → честный исход, стоп
  ├─ GET_VERSION                 # версия KFD-ABI в отчёт
  ├─ GET_MEMORY_TOPOLOGY         # CRAT-дамп → узлы GPU с PCI-атомами
  ├─ сопоставить CRAT-GPU ↔ DRM-карты ↔ render-миноры (двойное доказательство)
  └─ machine-report tests/golden/real-<host>-<gpu>.json (P5)
```

Выход — JSON поверх той же схемы (additive `probe_e3`-секция), verdict:
`PASS/FAIL/NOT_TESTED` по P4 здесь НЕ выдаётся — E3 доказывает **привязку и
живность интерфейса**; поддержку модели доказывает прогон rocminfo/HIP-smoke
(E3 full) и llama.cpp (E5) — их вердикты идут в COMPATIBILITY.md.

### 4.2 Read-only гарантии E3
- открытые файлы: только `O_RDONLY` для sysfs; `/dev/kfd` — исключение,
  требуется O_RDWR для ioctl-класса READ (так устроен KFD ABI); никаких
  write()/ioctl класса WRITE/CREATE;
- CREATE_QUEUE/ACQUIRE_VM запрещены в минимальном E3 (создают состояние GPU);
- ничего не устанавливается, udev/sysctl/модули не трогаются;
- каждый шаг под timeout; SIGINT-safe (fd закрывается finally);
- вывод — stdout/файл golden-отчёта; временные файлы не пишутся.

## 5. Обработка ошибок (полная, без «проглотила и молчит»)

| Исход | Код | Значение для consumer |
|---|---|---|
| ok | 0 | привязка доказана (evidence указан) |
| no_device | 3 | /dev/kfd отсутствует → NOT_TESTED (не FAIL!) |
| permission_denied | 4 | EACCES → unknown, подсказка про группу render; ничего не чиним |
| ioctl_unsupported | 5 | ENOTTY/EINVAL → старый/иной KFD ABI → unknown |
| topology_conflict | 6 | CRAT противоречит sysfs → unknown + оба наблюдения в отчёт |
| collect_error | 7 | таймаут/OSError → error (как STATUS_ERROR в детекторе) |

Инварианты: любой ненулевой исход **не может** породить `detected`/PASS там,
где раньше был unknown; unknown никогда не заменяется догадкой; при
противоречии источников побеждает «меньше знаешь — крепче спишь» (conflict).

## 6. Тестируемость без железа

- E3-lite полностью тестируется на существующих golden-фикстурах: нужно
  добавить `local_id`/`drm_render_minor` в properties фикстур (fake-rdna3-7800xt
  уже содержит drm_controller_id; формат local_id фиксируется по upstream-CRAT)
  и построить пару «фикстура с согласованными ID» / «фикстура с конфликтом».
- ioctl-слой: unit-тесты только на мок-слое (подстановка вместо реального
  fd, как LM_FAKE_SYS для sysfs); интеграционный прогон — исключительно на
  стенде E2, результат — golden real-отчёт (P5).
- CI без GPU: линковка, schema-валидность, моки; реальные verdicts в
  COMPATIBILITY появляются только из real-отчётов.

## 7. Ограничения: что НЕЛЬЗЯ проверить без реального RX 7800 XT

1. Фактический формат/наличие `local_id` и `drm_render_minor` в properties
   GPU-узла на ядре/ROCm стенда (в песочнице — только синтетика фикстур).
2. Поведение ioctl GET_VERSION/GET_MEMORY_TOPOLOGY: размеры CRAT, флаги,
   возвраты на конкретной версии amdgpu (ядро HWE vs OEM).
3. Реальная группа/права `/dev/kfd` после установки драйвера из backports.
4. Видимость iGPU gfx1036 одновременно с dGPU (зависит от BIOS Above-4G,
   iGPU-mode) — влияет на то, останется ли count_mismatch или исчезнет.
5. Присутствие `label` в DRM (в фикстуре model_status=unknown) и точное
   marketing-имя в lspci на живом железе.
6. Любые заявления PASS/FAIL в COMPATIBILITY.md — до первого real-отчёта
   строка RX 7800 XT обязана оставаться NOT_TESTED (P4).

## 8. Связь с этапами и следующий шаг

E3 (этот документ) замыкает разрыв detect→probe: E1 даёт причины unknown,
E7 lm doctor их показывает, E3-lite превращает часть count_mismatch/
ambiguous в доказанный matched, E3-ioctl — проверяет живность на стенде.
Реализация начинается с §4.0 (чистые функции + фикстуры + тесты, без
устройства), §4.1 — только после появления железа E2.

Открытые вопросы (не решать кодом заранее):
- сохранять ли полный CRAT-дамп в golden-отчёт (объём) или хеш + резюме;
- нужен ли отдельный schema_version для probe-отчётов (аддитивно vs мажор).
