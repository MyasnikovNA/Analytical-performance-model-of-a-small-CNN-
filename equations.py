
import numpy as np


PARAMETER_BYTES = 4_161_296.0


def flops(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    return B * (17712.0 * S**2 + 313344.0)


def memory(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    peak_activation_bytes = 44.0 * B * S**2

    return PARAMETER_BYTES + peak_activation_bytes


def bytes_moved(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    elements = 1_040_324.0 + B * (91.0 * S**2 + 2148.0)

    return 4.0 * elements


def latency(image_size, batch, theta):
    F_gflop = flops(image_size, batch) / 1e9
    Q_gb = bytes_moved(image_size, batch) / 1e9

    t0 = theta["t0_s"]
    bandwidth = theta["bandwidth_GBs"]
    compute_rate = theta["compute_GFLOPs"]

    return t0 + np.maximum(
        Q_gb / bandwidth,
        F_gflop / compute_rate,
    )


def energy(image_size, batch, theta_energy):
    F_gflop = flops(image_size, batch) / 1e9
    Q_gb = bytes_moved(image_size, batch) / 1e9

    return (
        theta_energy["e0_j"]
        + theta_energy["j_per_gflop"] * F_gflop
        + theta_energy["j_per_gbyte"] * Q_gb
    )
