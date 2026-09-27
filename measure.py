
import gc
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import SmallCNN
from equations import flops, memory, bytes_moved


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "measurements.csv"


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)


BASE_S = [32, 64, 128, 224, 256, 384, 512]

# Randomly sampled once with seed=42 and then fixed
# so that the experiment is reproducible.
EXTRA_S = [80, 272, 352, 400]

BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

EXTRA_B = [29, 56, 179]


S_VALUES = sorted(BASE_S + EXTRA_S)
B_VALUES = sorted(BASE_B + EXTRA_B)


try:
    import pynvml

    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_power_w():
        return pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE) / 1000.0

    POWER_AVAILABLE = True

except Exception as exc:
    print("Power measurement unavailable:", exc)

    POWER_AVAILABLE = False

    def get_power_w():
        return np.nan


def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def warmup(model, x, repetitions=3):
    with torch.inference_mode():
        for _ in range(repetitions):
            model(x)

    torch.cuda.synchronize()


def one_timed_forward(model, x):
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        y = model(x)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    del y

    return elapsed


def measure_latency(model, x):
    pilot = one_timed_forward(model, x)

    repeats = int(np.clip(0.5 / max(pilot, 1e-6), 7, 30))

    times = [
        one_timed_forward(model, x)
        for _ in range(repeats)
    ]

    return float(np.median(times))


def measure_memory(model, x):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        y = model(x)

    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()

    del y

    return int(peak)


def measure_energy(model, x, latency_s):
    if not POWER_AVAILABLE:
        return np.nan

    target_duration = 1.0

    repetitions = int(
        np.clip(
            target_duration / max(latency_s, 1e-6),
            10,
            5000,
        )
    )

    samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            try:
                samples.append(
                    (time.perf_counter(), get_power_w())
                )
            except Exception:
                pass

            time.sleep(0.02)

    thread = threading.Thread(target=sampler)
    thread.start()

    time.sleep(0.05)

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(repetitions):
            model(x)

    torch.cuda.synchronize()

    end = time.perf_counter()

    stop_event.set()
    thread.join()

    valid_power = [
        power
        for timestamp, power in samples
        if start <= timestamp <= end
        and np.isfinite(power)
    ]

    if not valid_power:
        return np.nan

    average_power = float(np.mean(valid_power))

    total_energy = average_power * (end - start)

    return total_energy / repetitions


def measure_configuration(model, S, B):
    x = torch.randn(
        B,
        3,
        S,
        S,
        device="cuda",
        dtype=torch.float32,
    )

    warmup(model, x)

    memory_bytes = measure_memory(model, x)

    latency_s = measure_latency(model, x)

    energy_j = measure_energy(
        model,
        x,
        latency_s,
    )

    del x

    return latency_s, memory_bytes, energy_j


def create_validation_split(configurations):
    rng = np.random.default_rng(2026)

    n_validation = round(len(configurations) * 0.25)

    validation_indices = set(
        rng.choice(
            len(configurations),
            size=n_validation,
            replace=False,
        ).tolist()
    )

    return [
        i in validation_indices
        for i in range(len(configurations))
    ]


def main():
    model = SmallCNN().cuda().eval()

    gpu_properties = torch.cuda.get_device_properties(0)

    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": gpu_properties.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "S_values": S_VALUES,
        "B_values": B_VALUES,
        "power_measurement_available": POWER_AVAILABLE,
    }

    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    configurations = [
        (S, B)
        for S in S_VALUES
        for B in B_VALUES
    ]

    validation_flags = create_validation_split(configurations)

    print("GPU:", metadata["gpu"])
    print("Image sizes:", S_VALUES)
    print("Batch sizes:", B_VALUES)
    print("Configurations:", len(configurations))
    print()

    rows = []

    for index, ((S, B), is_validation) in enumerate(
        zip(configurations, validation_flags)
    ):
        cleanup_cuda()

        print(
            f"[{index + 1:3d}/{len(configurations)}] "
            f"S={S:3d}, B={B:3d}",
            end="  ",
            flush=True,
        )

        try:
            latency_s, memory_bytes, energy_j = measure_configuration(
                model,
                S,
                B,
            )

            status = "OK"

            print(
                f"{latency_s * 1000:8.3f} ms | "
                f"{memory_bytes / 1024**2:8.1f} MiB | "
                f"{energy_j:8.4f} J"
            )

        except torch.cuda.OutOfMemoryError:
            latency_s = np.nan
            memory_bytes = np.nan
            energy_j = np.nan
            status = "OOM"

            print("OOM")

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                latency_s = np.nan
                memory_bytes = np.nan
                energy_j = np.nan
                status = "OOM"

                print("OOM")
            else:
                raise

        rows.append(
            {
                "S": S,
                "B": B,
                "latency_s": latency_s,
                "memory_bytes": memory_bytes,
                "energy_j": energy_j,
                "status": status,
                "is_validation": is_validation,
                "predicted_flops": float(flops(S, B)),
                "predicted_memory_bytes": float(memory(S, B)),
                "predicted_bytes_moved": float(bytes_moved(S, B)),
            }
        )

        pd.DataFrame(rows).to_csv(
            CSV_PATH,
            index=False,
        )

        cleanup_cuda()

    print()
    print("Saved:", CSV_PATH)


if __name__ == "__main__":
    main()
