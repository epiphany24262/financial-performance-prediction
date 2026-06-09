from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sector_distribution(df: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    counts = df["sector"].fillna("Missing").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("图1  行业板块样本分布")
    ax.set_xlabel("sector")
    ax.set_ylabel("样本数")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def plot_industry_top20(df: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    counts = df["industry"].fillna("Missing").value_counts().head(20).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    counts.plot(kind="barh", ax=ax, color="#F58518")
    ax.set_title("图2  行业 Top20 样本分布")
    ax.set_xlabel("样本数")
    ax.set_ylabel("industry")
    _save(fig, path)


def plot_missing_top20(missing_rates: pd.Series, path: Path) -> None:
    set_plot_style()
    top = missing_rates.sort_values(ascending=False).head(20).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    top.plot(kind="barh", ax=ax, color="#E45756")
    ax.set_title("图3  缺失率最高的20个字段")
    ax.set_xlabel("缺失率")
    ax.set_ylabel("字段")
    ax.set_xlim(0, min(1.0, max(0.05, float(top.max()) * 1.05)))
    _save(fig, path)


def plot_missing_by_quarter(missing_by_quarter: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(missing_by_quarter["quarter"], missing_by_quarter["avg_missing_rate"], marker="o", color="#72B7B2")
    ax.set_title("图4  各季度平均缺失率")
    ax.set_xlabel("季度")
    ax.set_ylabel("平均缺失率")
    ax.set_ylim(0, min(1.0, max(0.05, float(missing_by_quarter["avg_missing_rate"].max()) * 1.2)))
    _save(fig, path)


def plot_target_distributions(target_df: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    cols = list(target_df.columns)
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10))
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        values = np.sign(target_df[col].astype(float)) * np.log1p(np.abs(target_df[col].astype(float)))
        ax.hist(values.dropna(), bins=40, color="#54A24B", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel("signed log1p")
        ax.set_ylabel("count")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("图5  9个目标的 signed-log 分布", y=1.01, fontsize=13)
    _save(fig, path)


def plot_lag_correlation_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("图6  历史季度与目标的平均绝对相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    _save(fig, path)


def plot_accounting_identity_error(errors: pd.Series, path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(errors.dropna(), bins=60, color="#B279A2", alpha=0.85)
    ax.set_title("图7  会计恒等式相对误差分布")
    ax.set_xlabel("(assets - liabilities - equity) / assets")
    ax.set_ylabel("count")
    _save(fig, path)


def plot_target_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("图8  目标变量相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, path)
