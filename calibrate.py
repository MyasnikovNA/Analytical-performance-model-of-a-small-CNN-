
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import least_squares, nnls

from equations import (
    flops,
    memory,
    bytes_moved,
    latency,
    energy,
)


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "measurements.csv"


def fit_latency(df):
    train = df[
        (df["status"] == "OK")
        & (~df["is_validation"])
    ]

    S = train["S"].to_numpy()
    B = train["B"].to_numpy()
    y = train["latency_s"].to_numpy()

    F = flops(S, B) / 1e9
    Q = bytes_moved(S, B) / 1e9

    initial = np.array([
        5e-5,
        200.0,
        3000.0,
    ])

    lower = np.array([
        1e-7,
        1.0,
        10.0,
    ])

    upper = np.array([
        0.1,
        5000.0,
        100000.0,
    ])

    def prediction(log_parameters):
        t0, bandwidth, compute_rate = np.exp(log_parameters)

        return t0 + np.maximum(
            Q / bandwidth,
            F / compute_rate,
        )

    def residuals(log_parameters):
        predicted = prediction(log_parameters)

        return np.log(predicted) - np.log(y)

    result = least_squares(
        residuals,
        np.log(initial),
        bounds=(
            np.log(lower),
            np.log(upper),
        ),
    )

    t0, bandwidth, compute_rate = np.exp(result.x)

    return {
        "t0_s": float(t0),
        "bandwidth_GBs": float(bandwidth),
        "compute_GFLOPs": float(compute_rate),
    }


def fit_energy(df):
    train = df[
        (df["status"] == "OK")
        & (~df["is_validation"])
        & (df["energy_j"].notna())
    ]

    if len(train) < 3:
        return None

    S = train["S"].to_numpy()
    B = train["B"].to_numpy()

    F = flops(S, B) / 1e9
    Q = bytes_moved(S, B) / 1e9

    y = train["energy_j"].to_numpy()

    X = np.column_stack([
        np.ones_like(F),
        F,
        Q,
    ])

    coefficients, _ = nnls(X, y)

    return {
        "e0_j": float(coefficients[0]),
        "j_per_gflop": float(coefficients[1]),
        "j_per_gbyte": float(coefficients[2]),
    }


def error_metrics(real, predicted):
    real = np.asarray(real)
    predicted = np.asarray(predicted)

    absolute_percentage_error = (
        np.abs(predicted - real)
        / np.maximum(np.abs(real), 1e-12)
        * 100.0
    )

    return {
        "median_ape_percent": float(
            np.median(absolute_percentage_error)
        ),
        "mean_ape_percent": float(
            np.mean(absolute_percentage_error)
        ),
        "rmse": float(
            np.sqrt(np.mean((predicted - real) ** 2))
        ),
    }


def add_predictions(df, theta_latency, theta_energy):
    S = df["S"].to_numpy()
    B = df["B"].to_numpy()

    df["flops_prediction"] = flops(S, B)
    df["memory_prediction_bytes"] = memory(S, B)

    df["latency_prediction_s"] = latency(
        S,
        B,
        theta_latency,
    )

    if theta_energy is not None:
        df["energy_prediction_j"] = energy(
            S,
            B,
            theta_energy,
        )
    else:
        df["energy_prediction_j"] = np.nan

    Q = bytes_moved(S, B) / 1e9
    F = flops(S, B) / 1e9

    launch_term = np.full(
        len(df),
        theta_latency["t0_s"],
    )

    memory_term = (
        Q / theta_latency["bandwidth_GBs"]
    )

    compute_term = (
        F / theta_latency["compute_GFLOPs"]
    )

    terms = np.column_stack([
        launch_term,
        memory_term,
        compute_term,
    ])

    names = np.array([
        "launch-bound",
        "memory-bound",
        "compute-bound",
    ])

    df["predicted_regime"] = names[
        np.argmax(terms, axis=1)
    ]

    return df


def create_surface_plot(
    df,
    prediction_function,
    measured_column,
    filename,
    zlabel,
    scale=1.0,
):
    S_grid = np.linspace(
        df["S"].min(),
        df["S"].max(),
        60,
    )

    B_grid = np.linspace(
        df["B"].min(),
        df["B"].max(),
        60,
    )

    SS, BB = np.meshgrid(
        S_grid,
        B_grid,
    )

    ZZ = prediction_function(SS, BB) / scale

    fig = plt.figure(figsize=(9, 6))

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        SS,
        BB,
        ZZ,
        alpha=0.35,
    )

    if measured_column is not None:
        measured = df[
            (df["status"] == "OK")
            & (df[measured_column].notna())
        ]

        ax.scatter(
            measured["S"],
            measured["B"],
            measured[measured_column] / scale,
            s=18,
            label="measured",
        )

        ax.legend()

    else:
        ax.scatter(
            df["S"],
            df["B"],
            prediction_function(
                df["S"].to_numpy(),
                df["B"].to_numpy(),
            ) / scale,
            s=18,
            label="grid points",
        )

        ax.legend()

    ax.set_xlabel("Image size S")
    ax.set_ylabel("Batch size B")
    ax.set_zlabel(zlabel)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def create_parity_plot(
    real,
    predicted,
    filename,
    label,
):
    real = np.asarray(real)
    predicted = np.asarray(predicted)

    low = min(real.min(), predicted.min())
    high = max(real.max(), predicted.max())

    plt.figure(figsize=(6, 6))

    plt.scatter(
        real,
        predicted,
        s=20,
    )

    plt.plot(
        [low, high],
        [low, high],
        linestyle="--",
    )

    plt.xlabel(f"Measured {label}")
    plt.ylabel(f"Predicted {label}")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)

    theta_latency = fit_latency(df)
    theta_energy = fit_energy(df)

    print("Latency theta:")
    print(theta_latency)

    print()
    print("Energy theta:")
    print(theta_energy)

    df = add_predictions(
        df,
        theta_latency,
        theta_energy,
    )

    metrics = {}

    for split_name, validation_flag in [
        ("calibration", False),
        ("validation", True),
    ]:
        subset = df[
            (df["status"] == "OK")
            & (df["is_validation"] == validation_flag)
        ]

        metrics[f"latency_{split_name}"] = error_metrics(
            subset["latency_s"],
            subset["latency_prediction_s"],
        )

        if (
            theta_energy is not None
            and subset["energy_j"].notna().any()
        ):
            energy_subset = subset[
                subset["energy_j"].notna()
            ]

            metrics[f"energy_{split_name}"] = error_metrics(
                energy_subset["energy_j"],
                energy_subset["energy_prediction_j"],
            )

        metrics[f"memory_{split_name}"] = error_metrics(
            subset["memory_bytes"],
            subset["memory_prediction_bytes"],
        )

    theta = {
        "latency": theta_latency,
        "energy": theta_energy,
        "metrics": metrics,
    }

    with open(RESULTS_DIR / "theta.json", "w") as f:
        json.dump(theta, f, indent=2)

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    create_surface_plot(
        df,
        lambda S, B: flops(S, B),
        None,
        "flops_surface.png",
        "FLOPs (GFLOPs)",
        1e9,
    )

    create_surface_plot(
        df,
        lambda S, B: memory(S, B),
        "memory_bytes",
        "memory_surface.png",
        "Memory (MiB)",
        1024**2,
    )

    create_surface_plot(
        df,
        lambda S, B: latency(
            S,
            B,
            theta_latency,
        ),
        "latency_s",
        "latency_surface.png",
        "Latency (ms)",
        1e-3,
    )

    if theta_energy is not None:
        create_surface_plot(
            df,
            lambda S, B: energy(
                S,
                B,
                theta_energy,
            ),
            "energy_j",
            "energy_surface.png",
            "Energy (J)",
            1.0,
        )

    ok = df[df["status"] == "OK"]

    create_parity_plot(
        ok["memory_bytes"] / 1024**2,
        ok["memory_prediction_bytes"] / 1024**2,
        "memory_parity.png",
        "memory (MiB)",
    )

    create_parity_plot(
        ok["latency_s"] * 1000,
        ok["latency_prediction_s"] * 1000,
        "latency_parity.png",
        "latency (ms)",
    )

    if theta_energy is not None:
        energy_ok = ok[ok["energy_j"].notna()]

        create_parity_plot(
            energy_ok["energy_j"],
            energy_ok["energy_prediction_j"],
            "energy_parity.png",
            "energy (J)",
        )

    print()
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print()
    print("Regimes:")
    print(df["predicted_regime"].value_counts())

    print()
    print("Saved theta.json and figures.")


if __name__ == "__main__":
    main()
