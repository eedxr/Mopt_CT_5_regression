from __future__ import annotations

import numpy as np

from core import FeatureInfo, Matrix, Vector


def normalize_x(x: Vector) -> tuple[Vector, float, float]:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= 0.0:
        raise ValueError("Cannot normalize a constant feature")
    return (x - mean) / std, mean, std


def apply_normalization(x: Vector, mean: float, std: float) -> Vector:
    if std <= 0.0:
        raise ValueError("std must be positive")
    return (x - mean) / std


def polynomial_features(z: Vector, degree: int) -> Matrix:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return np.column_stack([z**power for power in range(degree + 1)]).astype(np.float64)


def build_design_matrix(
    x: Vector,
    degree: int,
    normalized: bool = True,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[Matrix, FeatureInfo]:
    if normalized:
        if mean is None or std is None:
            z, fitted_mean, fitted_std = normalize_x(x)
        else:
            fitted_mean = float(mean)
            fitted_std = float(std)
            z = apply_normalization(x, fitted_mean, fitted_std)
    else:
        fitted_mean = 0.0 if mean is None else float(mean)
        fitted_std = 1.0 if std is None else float(std)
        z = x.astype(np.float64, copy=False)

    return polynomial_features(z, degree), FeatureInfo(
        degree=degree,
        normalized=normalized,
        mean=fitted_mean,
        std=fitted_std,
    )


def transform_with_info(x: Vector, info: FeatureInfo) -> Matrix:
    return build_design_matrix(
        x,
        degree=info.degree,
        normalized=info.normalized,
        mean=info.mean,
        std=info.std,
    )[0]


def predict(Phi: Matrix, w: Vector) -> Vector:
    return Phi @ w
