from __future__ import annotations

import numpy as np

from core import DatasetSpec, RegressionDataset, Vector


def true_almost_linear(x: Vector) -> Vector:
    return 2.0 * x - 1.0 + 0.1 * np.sin(4.0 * x)


def true_nonlinear(x: Vector) -> Vector:
    return 0.5 * x**3 - x + np.sin(3.0 * x) + 2.0 * np.exp(-10.0 * (x - 1.0) ** 2)


DATASET_SPECS = [
    DatasetSpec(
        name="almost_linear",
        title="Almost linear dependence",
        formula="f_true(x) = 2x - 1 + 0.1 sin(4x)",
        x_min=-3.0,
        x_max=3.0,
        m=150,
        sigma=0.25,
        seed=202601,
        f_true=true_almost_linear,
    ),
    DatasetSpec(
        name="nonlinear",
        title="Strongly nonlinear dependence",
        formula="f_true(x) = 0.5x^3 - x + sin(3x) + 2 exp(-10(x - 1)^2)",
        x_min=-2.0,
        x_max=2.0,
        m=150,
        sigma=0.30,
        seed=202602,
        f_true=true_nonlinear,
    ),
]


def generate_dataset(spec: DatasetSpec) -> RegressionDataset:
    rng = np.random.default_rng(spec.seed)
    x = np.linspace(spec.x_min, spec.x_max, spec.m, dtype=np.float64)
    y_true = spec.f_true(x)
    noise = rng.normal(0.0, spec.sigma, size=spec.m)
    y = y_true + noise
    return RegressionDataset(spec=spec, x=x, y=y, y_true=y_true)


def default_datasets() -> list[RegressionDataset]:
    return [generate_dataset(spec) for spec in DATASET_SPECS]
