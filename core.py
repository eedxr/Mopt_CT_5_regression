from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt


type Vector = npt.NDArray[np.float64]
type Matrix = npt.NDArray[np.float64]
type ScalarFunction = Callable[[Vector], Vector]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    title: str
    formula: str
    x_min: float
    x_max: float
    m: int
    sigma: float
    seed: int
    f_true: ScalarFunction

    @property
    def noise_variance(self) -> float:
        return self.sigma**2


@dataclass(frozen=True, slots=True)
class RegressionDataset:
    spec: DatasetSpec
    x: Vector
    y: Vector
    y_true: Vector

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def title(self) -> str:
        return self.spec.title


@dataclass(frozen=True, slots=True)
class FeatureInfo:
    degree: int
    normalized: bool
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    method: str
    dataset: str
    degree: int
    w: Vector
    history: dict[str, list[float]]
    loss: float
    risk: float
    l1: float
    l2: float
    iterations: int
    epochs: int
    grad_evals: int
    time_seconds: float
    status: str
    batch_size: int | None = None
    lambda1: float = 0.0
    lambda2: float = 0.0
    regularization: str = "none"
    normalized: bool = True
    condition_number: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def as_vector(x: npt.ArrayLike) -> Vector:
    return np.asarray(x, dtype=np.float64)


def weights_to_string(w: Vector) -> str:
    return "[" + ", ".join(f"{value:.10g}" for value in np.asarray(w, dtype=float)) + "]"


def result_to_row(result: OptimizationResult, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment": "",
        "dataset": result.dataset,
        "method": result.method,
        "degree": result.degree,
        "regularization": result.regularization,
        "lambda1": result.lambda1,
        "lambda2": result.lambda2,
        "batch_size": "" if result.batch_size is None else result.batch_size,
        "normalized": result.normalized,
        "condition_number": result.condition_number,
        "loss": result.loss,
        "risk": result.risk,
        "l1": result.l1,
        "l2": result.l2,
        "iterations": result.iterations,
        "epochs": result.epochs,
        "grad_evals": result.grad_evals,
        "time_seconds": result.time_seconds,
        "status": result.status,
        "converged": result.converged,
        "weights": weights_to_string(result.w),
    }
    row.update(result.metadata)
    row.update(extra)
    return row
