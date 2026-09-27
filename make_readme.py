
import json
from pathlib import Path

import pandas as pd


results = Path("results")

df = pd.read_csv(
    results / "measurements.csv"
)

with open(
    results / "theta.json",
    "r",
    encoding="utf-8",
) as f:
    theta = json.load(f)

with open(
    results / "metadata.json",
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


ok_count = int(
    (df["status"] == "OK").sum()
)

oom_count = int(
    (df["status"] == "OOM").sum()
)


latency_metrics = theta[
    "metrics"
]["latency_validation"]

memory_metrics = theta[
    "metrics"
]["memory_validation"]


if "energy_validation" in theta["metrics"]:
    energy_metrics = theta[
        "metrics"
    ]["energy_validation"]

    energy_error = (
        f'{energy_metrics["median_ape_percent"]:.2f}%'
    )
else:
    energy_error = "not available"


if oom_count == 0:
    oom_text = (
        "No OOM configurations were observed "
        "for the required grid."
    )
else:
    oom_text = (
        f"{oom_count} configurations produced OOM."
    )


if "predicted_regime" in df.columns:
    regime_counts = (
        df["predicted_regime"]
        .value_counts()
        .to_dict()
    )
else:
    regime_counts = {}


text = f"""# HW1 — Analytical Performance Model of a Small CNN

## Hardware and software

- GPU: {metadata["gpu"]}
- GPU memory: {metadata["gpu_total_memory_bytes"] / 1024**3:.2f} GiB
- PyTorch: {metadata["torch"]}
- CUDA: {metadata["cuda"]}
- cuDNN: {metadata["cudnn"]}
- Python: {metadata["python"]}
- Precision: FP32
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.backends.cuda.matmul.allow_tf32 = False`

## Model

The experiment uses the sequential CNN from the assignment:

`Conv7x7 s2 3→32 → ReLU → MaxPool →`
`Conv5x5 32→64 → ReLU →`
`Conv3x3 s2 64→128 → ReLU →`
`Conv1x1 128→256 → ReLU →`
`Conv3x3 s2 256→256 → ReLU →`
`Conv1x1 256→512 → ReLU →`
`GlobalAvgPool → Linear 512→256 → ReLU → Linear 256→100`.

The model runs in `eval()` mode and under `torch.inference_mode()`.

Random FP32 tensors are used as inputs.

## Analytical equations

FLOPs:

```text
FLOPs(S, B) = B * (17712 * S^2 + 313344)
