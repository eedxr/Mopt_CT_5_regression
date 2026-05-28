from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import FeatureInfo, OptimizationResult, RegressionDataset, result_to_row
from data import default_datasets
from features import build_design_matrix, normalize_x, transform_with_info
from losses import regularization_name
from optimizers import (
    analytic_linear_1d,
    gauss_newton,
    levenberg_marquardt,
    minibatch_gd,
    sgd,
)
from plotting import (
    plot_batch_comparison,
    plot_coefficients,
    plot_dataset,
    plot_fit,
    plot_history_terms,
    plot_loss,
    plot_multiple_fits,
    plot_optimizer_comparison,
    save_results_table,
    slugify,
)


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

DEGREES = [1, 2, 3, 4, 5]
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 150]


def start_experiments() -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = default_datasets()
    for dataset in datasets:
        plot_dataset(dataset, FIGURES_DIR / "datasets" / f"{dataset.name}.png")

    frames = [
        save_dataset_description(datasets),
        run_base_experiments(datasets),
        run_batch_size_experiment(datasets),
        run_regularization_experiment(datasets),
        run_optimizer_comparison(datasets),
    ]
    summary = pd.concat(frames[1:], ignore_index=True)
    save_results_table(summary, TABLES_DIR / "summary.csv")
    return summary


def save_dataset_description(datasets: list[RegressionDataset]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        spec = dataset.spec
        rows.append(
            {
                "dataset": spec.name,
                "title": spec.title,
                "f_true": spec.formula,
                "x_min": spec.x_min,
                "x_max": spec.x_max,
                "m": spec.m,
                "sigma": spec.sigma,
                "noise_variance": spec.noise_variance,
                "seed": spec.seed,
            }
        )
    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "datasets.csv")
    return df


def run_base_experiments(datasets: list[RegressionDataset]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    epochs = 360
    max_iter = 35

    for dataset in datasets:
        for degree in DEGREES:
            Phi, info = build_design_matrix(dataset.x, degree=degree, normalized=True)
            condition = _condition_number(Phi)
            z = (dataset.x - info.mean) / info.std
            w0 = np.zeros(degree + 1, dtype=np.float64)
            results: list[OptimizationResult] = []

            if degree == 1:
                results.append(
                    analytic_linear_1d(
                        z,
                        dataset.y,
                        Phi,
                        dataset=dataset.name,
                        normalized=True,
                        condition_number=condition,
                    )
                )

            results.extend(
                [
                    sgd(
                        Phi,
                        dataset.y,
                        w0,
                        dataset=dataset.name,
                        degree=degree,
                        epochs=epochs,
                        alpha0=0.018,
                        decay=0.02,
                        seed=_seed(dataset.name, degree, 1),
                        normalized=True,
                        condition_number=condition,
                    ),
                    minibatch_gd(
                        Phi,
                        dataset.y,
                        w0,
                        dataset=dataset.name,
                        degree=degree,
                        batch_size=32,
                        epochs=epochs,
                        alpha0=0.025,
                        decay=0.02,
                        seed=_seed(dataset.name, degree, 2),
                        normalized=True,
                        condition_number=condition,
                    ),
                    gauss_newton(
                        Phi,
                        dataset.y,
                        w0,
                        dataset=dataset.name,
                        degree=degree,
                        max_iter=max_iter,
                        normalized=True,
                        condition_number=condition,
                    ),
                    levenberg_marquardt(
                        Phi,
                        dataset.y,
                        w0,
                        dataset=dataset.name,
                        degree=degree,
                        max_iter=2 * max_iter,
                        mu0=1e-2,
                        normalized=True,
                        condition_number=condition,
                    ),
                ]
            )

            for result in results:
                rows.append(result_to_row(result, experiment="base_models"))
                _save_fit_and_loss(dataset, info, result, "base_models")

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "base_models.csv")
    return df


def run_batch_size_experiment(datasets: list[RegressionDataset]) -> pd.DataFrame:
    dataset = _dataset_by_name(datasets, "nonlinear")
    degree = 5
    epochs = 480
    Phi, info = build_design_matrix(dataset.x, degree=degree, normalized=True)
    condition = _condition_number(Phi)
    w0 = np.zeros(degree + 1, dtype=np.float64)
    rows: list[dict[str, object]] = []
    histories: dict[int, dict[str, list[float]]] = {}

    for batch_size in BATCH_SIZES:
        result = minibatch_gd(
            Phi,
            dataset.y,
            w0,
            dataset=dataset.name,
            degree=degree,
            batch_size=min(batch_size, len(dataset.y)),
            epochs=epochs,
            alpha0=0.022,
            decay=0.018,
            seed=_seed("batch", batch_size, 0),
            normalized=True,
            condition_number=condition,
        )
        rows.append(result_to_row(result, experiment="batch_size"))
        _save_history(result, "batch_size", f"batch_{min(batch_size, len(dataset.y))}")
        histories[min(batch_size, len(dataset.y))] = result.history

    plot_batch_comparison(
        histories,
        "Batch size influence, nonlinear dataset, degree 5",
        FIGURES_DIR / "batch_size" / "loss_by_epoch.png",
        x_key="epochs",
    )
    plot_batch_comparison(
        histories,
        "Batch size influence by gradient evaluations",
        FIGURES_DIR / "batch_size" / "loss_by_grad_evals.png",
        x_key="grad_evals",
    )

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "batch_size.csv")
    return df


def run_regularization_experiment(datasets: list[RegressionDataset]) -> pd.DataFrame:
    dataset = _dataset_by_name(datasets, "nonlinear")
    degree = 10
    epochs = 1000
    Phi, info = build_design_matrix(dataset.x, degree=degree, normalized=True)
    condition = _condition_number(Phi)
    w0 = np.zeros(degree + 1, dtype=np.float64)

    regularizations = [
        ("none", 0.0, 0.0),
        ("L1_1e-4", 1e-4, 0.0),
        ("L1_1e-3", 1e-3, 0.0),
        ("L1_1e-2", 1e-2, 0.0),
        ("L2_1e-4", 0.0, 1e-4),
        ("L2_1e-3", 0.0, 1e-3),
        ("L2_1e-2", 0.0, 1e-2),
        ("ElasticNet_1e-3", 1e-3, 1e-3),
        ("L1_1e-1", 1e-1, 0.0),
        ("L2_1e-1", 0.0, 1e-1),
        ("ElasticNet_1e-1", 1e-1, 1e-1),
    ]
    rows: list[dict[str, object]] = []
    selected_predictions: dict[str, np.ndarray] = {}
    selected_weights: dict[str, np.ndarray] = {}

    x_grid = np.linspace(dataset.spec.x_min, dataset.spec.x_max, 500, dtype=np.float64)
    Phi_grid = transform_with_info(x_grid, info)
    y_true_grid = dataset.spec.f_true(x_grid)

    for label, lambda1, lambda2 in regularizations:
        result = minibatch_gd(
            Phi,
            dataset.y,
            w0,
            dataset=dataset.name,
            degree=degree,
            batch_size=32,
            epochs=epochs,
            alpha0=0.002,
            decay=0.02,
            lambda1=lambda1,
            lambda2=lambda2,
            seed=_seed("regularization", int(lambda1 * 1e6 + lambda2 * 1e6), 0),
            normalized=True,
            condition_number=condition,
            max_grad_norm=1000.0,
        )
        rows.append(result_to_row(result, experiment="regularization", reg_label=label))
        _save_history(result, "regularization", label)
        plot_history_terms(
            result.history,
            f"Regularization terms: {label}",
            FIGURES_DIR / "regularization" / f"terms_{slugify(label)}.png",
        )

        if label in {"none", "L1_1e-2", "L2_1e-2", "ElasticNet_1e-3", "ElasticNet_1e-1"}:
            selected_predictions[label] = Phi_grid @ result.w
            selected_weights[label] = result.w

    plot_multiple_fits(
        dataset.x,
        dataset.y,
        x_grid,
        y_true_grid,
        selected_predictions,
        "Regularization influence, nonlinear dataset, degree 10",
        FIGURES_DIR / "regularization" / "fits_selected.png",
    )
    plot_coefficients(
        selected_weights,
        "Polynomial coefficients before and after regularization",
        FIGURES_DIR / "regularization" / "coefficients_selected.png",
    )

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "regularization.csv")
    return df


def run_optimizer_comparison(datasets: list[RegressionDataset]) -> pd.DataFrame:
    dataset = _dataset_by_name(datasets, "nonlinear")
    rows: list[dict[str, object]] = []
    rows.extend(_run_optimizer_block(dataset, degree=5, normalized=True, tag="optimizer_comparison"))
    rows.extend(_run_optimizer_block(dataset, degree=10, normalized=False, tag="ill_conditioned_gn_lm"))
    df = pd.DataFrame(rows)
    save_results_table(df[df["experiment"] == "optimizer_comparison"], TABLES_DIR / "optimizer_comparison.csv")
    save_results_table(df[df["experiment"] == "ill_conditioned_gn_lm"], TABLES_DIR / "ill_conditioned_gn_lm.csv")
    return df


def summarize_to_console(summary: pd.DataFrame) -> None:
    columns = [
        "experiment",
        "dataset",
        "method",
        "degree",
        "regularization",
        "lambda1",
        "lambda2",
        "batch_size",
        "normalized",
        "condition_number",
        "loss",
        "risk",
        "iterations",
        "epochs",
        "grad_evals",
        "time_seconds",
        "status",
    ]
    existing = [column for column in columns if column in summary.columns]
    print(summary[existing].to_string(index=False))


def _run_optimizer_block(
    dataset: RegressionDataset,
    degree: int,
    normalized: bool,
    tag: str,
) -> list[dict[str, object]]:
    Phi, info = build_design_matrix(dataset.x, degree=degree, normalized=normalized)
    condition = _condition_number(Phi)
    w0 = np.zeros(degree + 1, dtype=np.float64)
    epochs = 450
    max_iter = 40
    histories: dict[str, dict[str, list[float]]] = {}
    rows: list[dict[str, object]] = []

    if tag == "ill_conditioned_gn_lm":
        results = [
            gauss_newton(
                Phi,
                dataset.y,
                w0,
                dataset=dataset.name,
                degree=degree,
                max_iter=max_iter,
                normalized=normalized,
                condition_number=condition,
            ),
            levenberg_marquardt(
                Phi,
                dataset.y,
                w0,
                dataset=dataset.name,
                degree=degree,
                max_iter=2 * max_iter,
                mu0=1e-1,
                normalized=normalized,
                condition_number=condition,
            ),
        ]
    else:
        results = [
            sgd(
                Phi,
                dataset.y,
                w0,
                dataset=dataset.name,
                degree=degree,
                epochs=epochs,
                alpha0=0.018,
                decay=0.02,
                seed=_seed(tag, 1, 0),
                normalized=normalized,
                condition_number=condition,
            ),
            minibatch_gd(
                Phi,
                dataset.y,
                w0,
                dataset=dataset.name,
                degree=degree,
                batch_size=32,
                epochs=epochs,
                alpha0=0.025,
                decay=0.02,
                seed=_seed(tag, 2, 0),
                normalized=normalized,
                condition_number=condition,
            ),
            gauss_newton(
                Phi,
                dataset.y,
                w0,
                dataset=dataset.name,
                degree=degree,
                max_iter=max_iter,
                normalized=normalized,
                condition_number=condition,
            ),
            levenberg_marquardt(
                Phi,
                dataset.y,
                w0,
                dataset=dataset.name,
                degree=degree,
                max_iter=2 * max_iter,
                mu0=1e-2,
                normalized=normalized,
                condition_number=condition,
            ),
        ]

    for result in results:
        rows.append(result_to_row(result, experiment=tag))
        _save_history(result, tag)
        histories[result.method] = result.history

    plot_optimizer_comparison(
        histories,
        f"{tag}: degree {degree}, normalized={normalized}",
        FIGURES_DIR / tag / f"loss_degree_{degree}_normalized_{normalized}.png",
        x_key="grad_evals",
    )

    return rows


def _save_fit_and_loss(
    dataset: RegressionDataset,
    info: FeatureInfo,
    result: OptimizationResult,
    experiment: str,
) -> None:
    x_grid = np.linspace(dataset.spec.x_min, dataset.spec.x_max, 500, dtype=np.float64)
    Phi_grid = transform_with_info(x_grid, info)
    y_pred_grid = Phi_grid @ result.w
    y_true_grid = dataset.spec.f_true(x_grid)
    base = (
        FIGURES_DIR
        / experiment
        / dataset.name
        / f"degree_{result.degree}"
        / slugify(result.method)
    )
    plot_fit(
        dataset.x,
        dataset.y,
        x_grid,
        y_true_grid,
        y_pred_grid,
        f"{dataset.title}: {result.method}, degree {result.degree}",
        base / "fit.png",
    )
    plot_loss(
        result.history,
        f"Loss: {dataset.name}, {result.method}, degree {result.degree}",
        base / "loss.png",
        x_key="epochs",
    )
    _save_history(result, experiment)


def _save_history(
    result: OptimizationResult,
    experiment: str,
    label: str = "",
) -> None:
    history = result.history
    if not history.get("loss"):
        return
    frame = pd.DataFrame(history)
    frame.insert(0, "method", result.method)
    frame.insert(0, "dataset", result.dataset)
    frame.insert(0, "degree", result.degree)
    frame.insert(0, "regularization", result.regularization)
    frame.insert(0, "lambda1", result.lambda1)
    frame.insert(0, "lambda2", result.lambda2)
    if result.batch_size is not None:
        frame.insert(0, "batch_size", result.batch_size)
    name_parts = [
        result.dataset,
        result.method,
        f"degree_{result.degree}",
        result.regularization,
        f"l1_{result.lambda1:g}",
        f"l2_{result.lambda2:g}",
    ]
    if result.batch_size is not None:
        name_parts.append(f"batch_{result.batch_size}")
    if label:
        name_parts.append(label)
    output_path = TABLES_DIR / "histories" / experiment / f"{slugify('_'.join(name_parts))}.csv"
    save_results_table(frame, output_path)


def _dataset_by_name(datasets: list[RegressionDataset], name: str) -> RegressionDataset:
    for dataset in datasets:
        if dataset.name == name:
            return dataset
    raise KeyError(name)


def _condition_number(Phi: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(Phi.T @ Phi))
    except np.linalg.LinAlgError:
        return float("inf")


def _seed(*parts: object) -> int:
    text = "_".join(str(part) for part in parts)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return 1000 + value


__all__ = [
    "PROJECT_DIR",
    "RESULTS_DIR",
    "TABLES_DIR",
    "FIGURES_DIR",
    "DEGREES",
    "BATCH_SIZES",
    "start_experiments",
    "summarize_to_console",
]
