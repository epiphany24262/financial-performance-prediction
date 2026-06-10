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


def _display_experiment_name(name: str) -> str:
    mapping = {
        "B0": "B0 均值",
        "B1": "B1 近一期",
        "B2": "B2 上年同期",
        "B3": "B3 趋势外推",
        "B4": "B4 规则融合",
        "M1_ridge_history_raw": "M1 Ridge",
        "M2_hgb_history_raw": "M2 HGB",
        "M3a_catboost_direct_history_raw": "M3a 直模-历史",
        "M3b_catboost_direct_history_industry": "M3b 直模-行业",
        "M3c_catboost_direct_history_metadata": "M3c 直模-元数据",
        "M3d_catboost_direct_history_metadata_engineered": "M3d 直模-工程",
        "M4_catboost_residual_history_metadata_engineered": "M4 残差",
        "M6_oof_blend": "M6 融合",
        "M7_accounting_postprocess": "M7 会计后处理",
    }
    return mapping.get(name, name)


def _display_target_name(name: str) -> str:
    mapping = {
        "Q0_TOTAL_ASSETS": "总资产",
        "Q0_TOTAL_LIABILITIES": "总负债",
        "Q0_TOTAL_STOCKHOLDERS_EQUITY": "股东权益",
        "Q0_GROSS_PROFIT": "毛利",
        "Q0_COST_OF_REVENUES": "营业成本",
        "Q0_REVENUES": "营业收入",
        "Q0_OPERATING_INCOME": "营业利润",
        "Q0_OPERATING_EXPENSES": "营业费用",
        "Q0_EBITDA": "EBITDA",
    }
    return mapping.get(name, name)


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


def plot_model_comparison(scores: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    ordered = scores.sort_values("mean_r2", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#BAB0AC" if value < ordered["mean_r2"].max() else "#4C78A8" for value in ordered["mean_r2"]]
    labels = [_display_experiment_name(item) for item in ordered["experiment_id"]]
    ax.barh(labels, ordered["mean_r2"], color=colors)
    ax.set_title("模型 OOF 平均 R2 对比")
    ax.set_xlabel("OOF 平均 R2")
    ax.set_ylabel("实验")
    for idx, value in enumerate(ordered["mean_r2"]):
        ax.text(value, idx, f" {value:.3f}", va="center", fontsize=8)
    _save(fig, path)


def plot_target_score_heatmap(scores: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    target_cols = [col for col in scores.columns if col.startswith("r2_Q0_")]
    matrix = scores.set_index("experiment_id")[target_cols]
    matrix.columns = [_display_target_name(col.removeprefix("r2_")) for col in matrix.columns]
    matrix.index = [_display_experiment_name(name) for name in matrix.index]
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.45 * len(matrix))))
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("各目标 OOF R2 热力图")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    _save(fig, path)


def plot_oof_scatter(oof: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    targets = [col.removeprefix("actual_") for col in oof.columns if col.startswith("actual_Q0_")]
    ncols = 3
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 11))
    axes = np.asarray(axes).reshape(-1)
    for ax, target in zip(axes, targets):
        actual = oof[f"actual_{target}"].astype(float)
        pred = oof[f"pred_{target}"].astype(float)
        actual_log = np.sign(actual) * np.log1p(np.abs(actual))
        pred_log = np.sign(pred) * np.log1p(np.abs(pred))
        ax.scatter(actual_log, pred_log, s=9, alpha=0.45, color="#4C78A8", edgecolors="none")
        lower = np.nanmin([actual_log.min(), pred_log.min()])
        upper = np.nanmax([actual_log.max(), pred_log.max()])
        ax.plot([lower, upper], [lower, upper], color="#E45756", linewidth=1)
        ax.set_title(_display_target_name(target))
        ax.set_xlabel("真实值 signed log1p")
        ax.set_ylabel("预测值 signed log1p")
    for ax in axes[len(targets):]:
        ax.axis("off")
    fig.suptitle("OOF 真实值与预测值散点图", y=1.01, fontsize=13)
    _save(fig, path)


def plot_residual_distribution(oof: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    residuals = []
    labels = []
    for col in oof.columns:
        if not col.startswith("actual_Q0_"):
            continue
        target = col.removeprefix("actual_")
        residual = oof[f"pred_{target}"].astype(float) - oof[f"actual_{target}"].astype(float)
        scale = np.nanmedian(np.abs(oof[f"actual_{target}"].astype(float))) or 1.0
        residuals.append((residual / scale).replace([np.inf, -np.inf], np.nan).dropna())
        labels.append(_display_target_name(target))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.boxplot(residuals, labels=labels, showfliers=False, vert=False)
    ax.set_title("各目标 OOF 残差分布")
    ax.set_xlabel("残差 / 目标真实值中位绝对值")
    ax.set_ylabel("目标")
    _save(fig, path)


def plot_blend_weights(blend_scores: pd.DataFrame, path: Path) -> None:
    set_plot_style()
    members = blend_scores["members"].iloc[0].split(",")
    weights = pd.DataFrame(
        [list(map(float, row.split(","))) for row in blend_scores["weights"]],
        index=[_display_target_name(target) for target in blend_scores["target"]],
        columns=members,
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bottom = np.zeros(len(weights))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for idx, member in enumerate(members):
        ax.bar(weights.index, weights[member], bottom=bottom, label=member, color=colors[idx % len(colors)])
        bottom += weights[member].values
    ax.set_title("逐目标 OOF 融合权重")
    ax.set_xlabel("目标")
    ax.set_ylabel("权重")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper right")
    _save(fig, path)
