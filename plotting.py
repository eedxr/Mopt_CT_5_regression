from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import RegressionDataset, Vector


plt.style.use("seaborn-v0_8-whitegrid")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
    return slug.strip("_") or "item"


def save_results_table(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dataset(dataset: RegressionDataset, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(dataset.x, dataset.y, s=18, alpha=0.68, label="noisy observations")
    ax.plot(dataset.x, dataset.y_true, linewidth=2.2, label="true function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(dataset.title)
    ax.legend(fontsize=8)
    save_plot(fig, output_path)


def plot_fit(
    x: Vector,
    y: Vector,
    x_grid: Vector,
    y_true_grid: Vector,
    y_pred_grid: Vector,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(x, y, s=18, alpha=0.62, label="noisy observations")
    ax.plot(x_grid, y_true_grid, linewidth=2.1, label="true function")
    ax.plot(x_grid, y_pred_grid, linewidth=2.1, label="model prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(fontsize=8)
    save_plot(fig, output_path)


def plot_multiple_fits(
    x: Vector,
    y: Vector,
    x_grid: Vector,
    y_true_grid: Vector,
    predictions: Mapping[str, Vector],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.scatter(x, y, s=16, alpha=0.55, label="noisy observations")
    ax.plot(x_grid, y_true_grid, linewidth=2.3, color="black", label="true function")
    for label, y_pred in predictions.items():
        ax.plot(x_grid, y_pred, linewidth=1.8, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(fontsize=7, ncols=2)
    save_plot(fig, output_path)


def plot_loss(
    history: dict[str, list[float]],
    title: str,
    output_path: Path,
    x_key: str = "epochs",
    y_key: str = "loss",
    log_y: bool = True,
) -> None:
    if not history.get(y_key):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.asarray(history.get(x_key, range(len(history[y_key]))), dtype=np.float64)
    y = np.asarray(history[y_key], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        plt.close(fig)
        return
    x = x[finite]
    y = y[finite]
    ax.plot(x, y, linewidth=1.8)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    if log_y and np.all(y > 0.0):
        ax.set_yscale("log")
    save_plot(fig, output_path)


def plot_history_terms(
    history: dict[str, list[float]],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.asarray(history.get("epochs", range(len(history["loss"]))), dtype=np.float64)
    for key in ("loss", "risk", "l1", "l2"):
        values = np.asarray(history.get(key, []), dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(values)
        if values.size and finite.any() and np.any(values[finite] > 0.0):
            ax.plot(x[finite], values[finite], linewidth=1.7, label=key)
    ax.set_xlabel("epochs")
    ax.set_ylabel("value")
    ax.set_title(title)
    positive = [
        value
        for key in ("loss", "risk", "l1", "l2")
        for value in history.get(key, [])
        if np.isfinite(value) and value > 0.0
    ]
    if positive:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    save_plot(fig, output_path)


def plot_batch_comparison(
    histories: Mapping[int, dict[str, list[float]]],
    title: str,
    output_path: Path,
    x_key: str = "epochs",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for batch_size, history in histories.items():
        x = np.asarray(history[x_key], dtype=np.float64)
        y = np.asarray(history["loss"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            ax.plot(x[finite], y[finite], linewidth=1.55, label=f"B={batch_size}")
    ax.set_xlabel(x_key)
    ax.set_ylabel("loss")
    ax.set_title(title)
    if all(np.all(np.asarray(history["loss"])[np.isfinite(history["loss"])] > 0.0) for history in histories.values()):
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncols=2)
    save_plot(fig, output_path)


def plot_optimizer_comparison(
    histories: Mapping[str, dict[str, list[float]]],
    title: str,
    output_path: Path,
    x_key: str = "grad_evals",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, history in histories.items():
        x = np.asarray(history[x_key], dtype=np.float64)
        y = np.asarray(history["loss"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            ax.plot(x[finite], y[finite], linewidth=1.7, marker="o" if len(y) < 12 else None, label=label)
    ax.set_xlabel(x_key)
    ax.set_ylabel("loss")
    ax.set_title(title)
    if all(np.all(np.asarray(history["loss"])[np.isfinite(history["loss"])] > 0.0) for history in histories.values()):
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    save_plot(fig, output_path)


def plot_coefficients(
    weights: Mapping[str, Vector],
    title: str,
    output_path: Path,
) -> None:
    labels = list(weights.keys())
    max_len = max(len(w) for w in weights.values())
    x = np.arange(max_len)
    width = 0.78 / max(1, len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for index, label in enumerate(labels):
        padded = np.full(max_len, np.nan)
        padded[: len(weights[label])] = weights[label]
        offset = (index - (len(labels) - 1) / 2.0) * width
        ax.bar(x + offset, padded, width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("coefficient index")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    save_plot(fig, output_path)
