from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from core import Matrix, OptimizationResult, Vector
from losses import gradient, l2_regularization_matrix, loss_components, regularization_name


def analytic_linear_1d(
    x_feature: Vector,
    y: Vector,
    Phi: Matrix,
    dataset: str,
    normalized: bool,
    condition_number: float | None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
) -> OptimizationResult:
    if Phi.shape[1] != 2:
        raise ValueError("analytic_linear_1d expects a degree-1 design matrix")

    start = perf_counter()
    x_mean = float(np.mean(x_feature))
    y_mean = float(np.mean(y))
    denominator = float(np.sum((x_feature - x_mean) ** 2))
    if denominator <= 0.0:
        raise ValueError("Cannot fit a line to a constant feature")
    a = float(np.sum((x_feature - x_mean) * (y - y_mean)) / denominator)
    b = y_mean - a * x_mean
    w = np.array([b, a], dtype=np.float64)
    elapsed = perf_counter() - start

    history = _empty_history()
    _append_history(history, Phi, y, w, lambda1, lambda2, 0, 0, elapsed)
    loss, risk, l1, l2 = loss_components(Phi, y, w, lambda1, lambda2)
    return OptimizationResult(
        method="analytic",
        dataset=dataset,
        degree=1,
        w=w,
        history=history,
        loss=loss,
        risk=risk,
        l1=l1,
        l2=l2,
        iterations=1,
        epochs=0,
        grad_evals=0,
        time_seconds=elapsed,
        status="closed_form",
        lambda1=lambda1,
        lambda2=lambda2,
        regularization=regularization_name(lambda1, lambda2),
        normalized=normalized,
        condition_number=condition_number,
    )


def sgd(
    Phi: Matrix,
    y: Vector,
    w0: Vector,
    dataset: str,
    degree: int,
    epochs: int = 300,
    alpha0: float = 0.02,
    decay: float = 0.02,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    seed: int = 0,
    normalized: bool = True,
    condition_number: float | None = None,
    max_grad_norm: float = 100.0,
) -> OptimizationResult:
    return minibatch_gd(
        Phi=Phi,
        y=y,
        w0=w0,
        dataset=dataset,
        degree=degree,
        batch_size=1,
        epochs=epochs,
        alpha0=alpha0,
        decay=decay,
        lambda1=lambda1,
        lambda2=lambda2,
        seed=seed,
        method="sgd",
        normalized=normalized,
        condition_number=condition_number,
        max_grad_norm=max_grad_norm,
    )


def minibatch_gd(
    Phi: Matrix,
    y: Vector,
    w0: Vector,
    dataset: str,
    degree: int,
    batch_size: int,
    epochs: int = 300,
    alpha0: float = 0.02,
    decay: float = 0.02,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    seed: int = 0,
    method: str = "mini_batch",
    normalized: bool = True,
    condition_number: float | None = None,
    tol: float = 1e-9,
    max_grad_norm: float = 100.0,
) -> OptimizationResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    m = len(y)
    batch_size = min(int(batch_size), m)
    rng = np.random.default_rng(seed)
    w = w0.astype(np.float64, copy=True)
    history = _empty_history()
    start = perf_counter()
    grad_evals = 0
    clipped_updates = 0
    status = "max_epochs_reached"

    _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, 0, 0.0)

    for epoch in range(1, epochs + 1):
        alpha = alpha0 / (1.0 + decay * (epoch - 1))
        indices = rng.permutation(m)
        for first in range(0, m, batch_size):
            batch = indices[first:first + batch_size]
            grad = gradient(Phi[batch], y[batch], w, lambda1=lambda1, lambda2=lambda2)
            grad_evals += 1
            if not np.all(np.isfinite(grad)):
                status = "nonfinite_gradient"
                elapsed = perf_counter() - start
                return _finish(
                    method,
                    dataset,
                    degree,
                    w,
                    Phi,
                    y,
                    history,
                    epoch,
                    grad_evals,
                    elapsed,
                    status,
                    batch_size,
                    lambda1,
                    lambda2,
                    normalized,
                    condition_number,
                    {"alpha0": alpha0, "decay": decay, "clipped_updates": clipped_updates},
                )
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
                clipped_updates += 1
            w = w - alpha * grad
            if not np.all(np.isfinite(w)):
                status = "nonfinite_weights"
                elapsed = perf_counter() - start
                return _finish(
                    method,
                    dataset,
                    degree,
                    w,
                    Phi,
                    y,
                    history,
                    epoch,
                    grad_evals,
                    elapsed,
                    status,
                    batch_size,
                    lambda1,
                    lambda2,
                    normalized,
                    condition_number,
                    {"alpha0": alpha0, "decay": decay, "clipped_updates": clipped_updates},
                )

        elapsed = perf_counter() - start
        _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, epoch, elapsed)
        full_grad = gradient(Phi, y, w, lambda1=lambda1, lambda2=lambda2)
        if float(np.linalg.norm(full_grad)) <= tol:
            status = "converged"
            break

    elapsed = perf_counter() - start
    return _finish(
        method,
        dataset,
        degree,
        w,
        Phi,
        y,
        history,
        epoch,
        grad_evals,
        elapsed,
        status,
        batch_size,
        lambda1,
        lambda2,
        normalized,
        condition_number,
        {"alpha0": alpha0, "decay": decay, "clipped_updates": clipped_updates, "max_grad_norm": max_grad_norm},
    )


def gauss_newton(
    Phi: Matrix,
    y: Vector,
    w0: Vector,
    dataset: str,
    degree: int,
    max_iter: int = 30,
    tol: float = 1e-12,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    normalized: bool = True,
    condition_number: float | None = None,
) -> OptimizationResult:
    if lambda1 > 0.0:
        raise ValueError("Gauss-Newton implementation supports no L1 term")

    w = w0.astype(np.float64, copy=True)
    history = _empty_history()
    start = perf_counter()
    grad_evals = 0
    status = "max_iter_reached"
    _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, 0, 0.0)

    m, n = Phi.shape
    reg = l2_regularization_matrix(n)

    for iteration in range(1, max_iter + 1):
        r = Phi @ w - y
        a = Phi.T @ Phi + m * lambda2 * reg
        b = -Phi.T @ r - m * lambda2 * (reg @ w)
        grad_evals += 1
        try:
            h = np.linalg.solve(a, b)
            linear_solver = "solve"
        except np.linalg.LinAlgError:
            h = np.linalg.lstsq(a, b, rcond=None)[0]
            linear_solver = "lstsq"

        if not np.all(np.isfinite(h)):
            status = "nonfinite_step"
            break
        w = w + h
        elapsed = perf_counter() - start
        _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, iteration, elapsed)

        if float(np.linalg.norm(h)) <= tol * (1.0 + float(np.linalg.norm(w))):
            status = "converged"
            break

    elapsed = perf_counter() - start
    return _finish(
        "gauss_newton",
        dataset,
        degree,
        w,
        Phi,
        y,
        history,
        iteration if "iteration" in locals() else 0,
        grad_evals,
        elapsed,
        status,
        None,
        lambda1,
        lambda2,
        normalized,
        condition_number,
        {"linear_solver": linear_solver if "linear_solver" in locals() else ""},
    )


def levenberg_marquardt(
    Phi: Matrix,
    y: Vector,
    w0: Vector,
    dataset: str,
    degree: int,
    max_iter: int = 60,
    tol: float = 1e-12,
    mu0: float = 1e-2,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    normalized: bool = True,
    condition_number: float | None = None,
) -> OptimizationResult:
    if lambda1 > 0.0:
        raise ValueError("Levenberg-Marquardt implementation supports no L1 term")
    if mu0 <= 0.0:
        raise ValueError("mu0 must be positive")

    w = w0.astype(np.float64, copy=True)
    history = _empty_history()
    start = perf_counter()
    grad_evals = 0
    mu = float(mu0)
    accepted = 0
    rejected = 0
    status = "max_iter_reached"
    _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, 0, 0.0)

    m, n = Phi.shape
    identity = np.eye(n, dtype=np.float64)
    reg = l2_regularization_matrix(n)

    for iteration in range(1, max_iter + 1):
        old_loss = loss_components(Phi, y, w, lambda1=lambda1, lambda2=lambda2)[0]
        r = Phi @ w - y
        a = Phi.T @ Phi + m * lambda2 * reg + mu * identity
        b = -Phi.T @ r - m * lambda2 * (reg @ w)
        grad_evals += 1
        try:
            h = np.linalg.solve(a, b)
            linear_solver = "solve"
        except np.linalg.LinAlgError:
            h = np.linalg.lstsq(a, b, rcond=None)[0]
            linear_solver = "lstsq"

        if not np.all(np.isfinite(h)):
            status = "nonfinite_step"
            break

        candidate = w + h
        new_loss = loss_components(Phi, y, candidate, lambda1=lambda1, lambda2=lambda2)[0]
        step_norm = float(np.linalg.norm(h))
        if np.isfinite(new_loss) and new_loss <= old_loss:
            w = candidate
            mu = max(mu / 2.0, 1e-14)
            accepted += 1
            elapsed = perf_counter() - start
            _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, iteration, elapsed)
            if step_norm <= tol * (1.0 + float(np.linalg.norm(w))):
                status = "converged"
                break
        elif step_norm <= tol * (1.0 + float(np.linalg.norm(w))):
            status = "converged"
            break
        else:
            mu = min(mu * 2.0, 1e14)
            rejected += 1
            elapsed = perf_counter() - start
            _append_history(history, Phi, y, w, lambda1, lambda2, grad_evals, iteration, elapsed)

        if mu >= 1e14:
            status = "damping_too_large"
            break

    elapsed = perf_counter() - start
    return _finish(
        "levenberg_marquardt",
        dataset,
        degree,
        w,
        Phi,
        y,
        history,
        iteration if "iteration" in locals() else 0,
        grad_evals,
        elapsed,
        status,
        None,
        lambda1,
        lambda2,
        normalized,
        condition_number,
        {
            "mu0": mu0,
            "mu_final": mu,
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "linear_solver": linear_solver if "linear_solver" in locals() else "",
        },
    )


def _empty_history() -> dict[str, list[float]]:
    return {
        "loss": [],
        "risk": [],
        "l1": [],
        "l2": [],
        "grad_evals": [],
        "epochs": [],
        "time": [],
    }


def _append_history(
    history: dict[str, list[float]],
    Phi: Matrix,
    y: Vector,
    w: Vector,
    lambda1: float,
    lambda2: float,
    grad_evals: int,
    epoch: int,
    elapsed: float,
) -> None:
    loss, risk, l1, l2 = loss_components(Phi, y, w, lambda1=lambda1, lambda2=lambda2)
    history["loss"].append(loss)
    history["risk"].append(risk)
    history["l1"].append(l1)
    history["l2"].append(l2)
    history["grad_evals"].append(float(grad_evals))
    history["epochs"].append(float(epoch))
    history["time"].append(float(elapsed))


def _finish(
    method: str,
    dataset: str,
    degree: int,
    w: Vector,
    Phi: Matrix,
    y: Vector,
    history: dict[str, list[float]],
    iteration_or_epoch: int,
    grad_evals: int,
    elapsed: float,
    status: str,
    batch_size: int | None,
    lambda1: float,
    lambda2: float,
    normalized: bool,
    condition_number: float | None,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    loss, risk, l1, l2 = loss_components(Phi, y, w, lambda1=lambda1, lambda2=lambda2)
    return OptimizationResult(
        method=method,
        dataset=dataset,
        degree=degree,
        w=w.copy(),
        history=history,
        loss=loss,
        risk=risk,
        l1=l1,
        l2=l2,
        iterations=int(iteration_or_epoch),
        epochs=int(iteration_or_epoch) if batch_size is not None else 0,
        grad_evals=int(grad_evals),
        time_seconds=float(elapsed),
        status=status,
        batch_size=batch_size,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        regularization=regularization_name(lambda1, lambda2),
        normalized=normalized,
        condition_number=condition_number,
        metadata=dict(metadata or {}),
    )
