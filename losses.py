from __future__ import annotations

import numpy as np

from core import Matrix, Vector


L1_EPS = 1e-8


def regularization_name(lambda1: float, lambda2: float) -> str:
    has_l1 = lambda1 > 0.0
    has_l2 = lambda2 > 0.0
    if has_l1 and has_l2:
        return "ElasticNet"
    if has_l1:
        return "L1"
    if has_l2:
        return "L2"
    return "none"


def residuals(Phi: Matrix, y: Vector, w: Vector) -> Vector:
    return Phi @ w - y


def risk(Phi: Matrix, y: Vector, w: Vector) -> float:
    r = residuals(Phi, y, w)
    return float(np.mean(r**2))


def loss_components(
    Phi: Matrix,
    y: Vector,
    w: Vector,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    reg_intercept: bool = False,
) -> tuple[float, float, float, float]:
    empirical_risk = risk(Phi, y, w)
    w_reg = w if reg_intercept else w[1:]
    l1 = float(lambda1 * np.sum(np.abs(w_reg)))
    l2 = float(lambda2 * np.sum(w_reg**2))
    return empirical_risk + l1 + l2, empirical_risk, l1, l2


def gradient(
    Phi: Matrix,
    y: Vector,
    w: Vector,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    reg_intercept: bool = False,
) -> Vector:
    m = len(y)
    r = residuals(Phi, y, w)
    grad = 2.0 / m * (Phi.T @ r)

    if lambda1 > 0.0 or lambda2 > 0.0:
        grad_reg = np.zeros_like(w)
        start = 0 if reg_intercept else 1
        if lambda1 > 0.0:
            grad_reg[start:] += lambda1 * w[start:] / np.sqrt(w[start:] ** 2 + L1_EPS)
        if lambda2 > 0.0:
            grad_reg[start:] += 2.0 * lambda2 * w[start:]
        grad = grad + grad_reg

    return grad


def l2_regularization_matrix(n_features: int, reg_intercept: bool = False) -> Matrix:
    matrix = np.eye(n_features, dtype=np.float64)
    if not reg_intercept and n_features:
        matrix[0, 0] = 0.0
    return matrix
