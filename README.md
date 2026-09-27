# Analytical Performance Model of a Small CNN

Домашняя работа по курсу Efficient Models.

Цель — построить простую аналитическую модель стоимости forward pass для небольшой CNN и проверить её на реальном GPU.

Модель предсказывает:

- `FLOPs(S, B)` — число операций;
- `Memory(S, B)` — пиковую память;
- `Latency(S, B, θ)` — время одного forward pass;
- `Energy(S, B, θE)` — энергию одного forward pass.

Здесь `S` — размер изображения, `B` — batch size.

## Model

Используется последовательная CNN из задания:

```text
Conv7x7 s2 3→32
MaxPool 3x3 s2
Conv5x5 32→64
Conv3x3 s2 64→128
Conv1x1 128→256
Conv3x3 s2 256→256
Conv1x1 256→512
GlobalAvgPool
Linear 512→256
Linear 256→100
```

Все измерения выполняются в FP32, `eval()` и `torch.inference_mode()`.

## Analytical equations

Соглашение:

```text
1 MAC = 2 FLOPs
```

Итоговая формула FLOPs:

```text
FLOPs(S, B) = B * (17712 * S^2 + 313344)
```

Упрощённая модель пиковой памяти:

```text
Memory(S, B) = 4,161,296 + 44 * B * S^2 bytes
```

Оценка logical bytes moved:

```text
Bytes(S, B) = 4 * [1,040,324 + B * (91 * S^2 + 2148)]
```

Модель latency:

```text
Latency = t0 + max(
    Bytes / effective_bandwidth,
    FLOPs / effective_compute_rate
)
```

Модель energy:

```text
Energy = e0 + e_flop * GFLOPs + e_memory * GB_moved
```

Параметры latency и energy калибруются по реальным измерениям GPU.

## Measurement protocol

Используется сетка из 132 конфигураций `(S, B)`.

Для каждой точки:

- создаётся случайный FP32 tensor;
- выполняются warm-up проходы;
- latency измеряется как median времени forward pass;
- memory измеряется через `torch.cuda.max_memory_allocated()`;
- energy оценивается по whole-GPU power через NVML;
- OOM ловится и записывается отдельно.

25% точек не участвуют в fitting и используются только для validation.

## Results

Эксперимент был выполнен на NVIDIA Tesla T4.

Все 132 конфигурации из заданной сетки поместились в память, поэтому OOM внутри этой сетки не наблюдался.

Предварительно по графикам видно, что модель хорошо ловит общий тренд, но на больших нагрузках сильнее ошибается, особенно по памяти.

Validation median absolute percentage error:

- Latency: ~15.5%
- Energy: ~34.0%
- Memory: ~54.6%

Основная причина расхождения по памяти — упрощённая аналитическая модель не учитывает внутренние CUDA/cuDNN buffers, workspace и поведение allocator.

По latency модель выделила в основном launch-bound и compute-bound режимы. Memory-bound область на этой сетке получилась очень небольшой, что показывает ограничение простой whole-network модели.

## Repository structure

```text
hw1/
├── README.md
├── hw1_handwritten.pdf
├── models.py
├── equations.py
├── measure.py
├── calibrate.py
├── make_readme.py
└── results/
    ├── measurements.csv
    ├── metadata.json
    ├── theta.json
    └── figures/
```

## Reproduction

Установить зависимости:

```bash
pip install torch numpy pandas scipy matplotlib nvidia-ml-py
```

Запустить измерения:

```bash
python measure.py
```

Откалибровать параметры и построить графики:

```bash
python calibrate.py
```

## Files

- `hw1_handwritten.pdf` — ручной вывод формул;
- `measurements.csv` — все измерения;
- `theta.json` — fitted parameters и validation metrics;
- `results/figures/` — графики predicted vs measured.
