# Матрица совместимости LocalMind OS

Формат: одна строка = одно заявление. Заполняется **только** на основе
machine-generated отчётов `localmind-detect` + probe с реального железа
(принцип 4, POLICY P4). Пустой статус = NOT_TESTED.

Легенда: PASS / FAIL / NOT_TESTED.

| GPU (chip) | Arch | Driver/amdgpu | ROCm verdict | Vulkan verdict | llama.cpp ROCm | llama.cpp Vulkan | Ollama GPU | Отчёт (tests/golden/) | Дата |
|---|---|---|---|---|---|---|---|---|---|
| Radeon RX 7900 XTX (gfx1100) | RDNA3 | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | — | — |
| Radeon RX 7800 XT (gfx1101) | RDNA3 | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | — | — |
| Radeon RX 6800 (gfx1030) | RDNA2 | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | — | — |
| Radeon RX 6600 (gfx1032) | RDNA2 | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | — | — |
| Intel/AMD iGPU (generic) | — | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_TESTED | — | — |

Замечания:
- Официальный список ROCm поддержки десктопных Radeon ограничен; для многих
  SKU единственный доказуемый путь — Vulkan (llama.cpp ggml-vulkan).
- Статус ROCm для gfx110x в upstream меняется по релизам — проверять по
  release notes конкретной версии ROCm перед каждым заявлением.
- Ни одна строка выше пока не подтверждена: среда разработки (Debian 12,
  без GPU) не позволяет выполнять тесты. Это не FAIL, это отсутствие данных.
