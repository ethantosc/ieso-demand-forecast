#!/usr/bin/env python3
"""Static figures for the reports. Every figure has its numbers in findings_auto.md too.

Palette: three categorical slots that pass the colour-blind checks as a set,
gray for context, ink-coloured text. Lines are direct-labelled, so identity
never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
CONTEXT = "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def ladder(summary: pl.DataFrame, labels: dict[str, str], highlight: str | None,
           threshold: float | None, path: Path, title: str) -> Path:
    """Horizontal bars, one per model, sorted by MAE. Only the best Track A model is coloured."""
    rows = summary.sort("MAE", descending=True)
    fig, ax = plt.subplots(figsize=(9, 0.45 * rows.height + 1.4))
    names = [labels.get(m, m) for m in rows["model"]]
    colours = [SERIES[0] if m == highlight else CONTEXT for m in rows["model"]]
    hatch = ["//" if m.startswith("lgbm_B") else "" for m in rows["model"]]
    bars = ax.barh(names, rows["MAE"], color=colours, height=0.62, edgecolor=SURFACE, linewidth=2)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    for y, v in enumerate(rows["MAE"]):
        ax.text(v, y, f"  {v:,.0f}", va="center", color=INK_2, fontsize=9)
    if threshold is not None:
        ax.axvline(threshold, color=INK, lw=1, ls="--")
        ax.text(threshold, rows.height - 0.35, " seasonal-naive threshold", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel("MAE (MW), lower is better")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, rows["MAE"].max() * 1.15)
    ax.set_title(title)
    return _save(fig, path)


def colour_for(model: str) -> str:
    """One colour per model, the same in every figure - never by position in a list."""
    if model.startswith("lgbm_A"):
        return SERIES[0]
    if model == "seasonal_naive":
        return SERIES[1]
    if model == "ieso_official":
        return SERIES[2]
    return INK_2


def _spread(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so no two are closer than `gap`, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous = None
    for i in order:
        placed[i] = values[i] if previous is None else max(values[i], previous + gap)
        previous = placed[i]
    shift = (sum(placed) - sum(values)) / len(values)   # recentre on the true ends
    return [p - shift for p in placed]


def lines_by(table: pl.DataFrame, x: str, models: list[str], labels: dict[str, str],
             path: Path, title: str, xlabel: str, ylabel: str = "MAE (MW)") -> Path:
    """Up to three models as direct-labelled lines (MAE by hour, by day, ...)."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ends = []
    for model in list(dict.fromkeys(models))[:3]:
        part = table.filter(pl.col("model") == model).sort(x)
        if part.is_empty():
            continue
        ax.plot(part[x], part["value"], color=colour_for(model), lw=2)
        ends.append((model, part[x][-1], part["value"][-1]))
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)
    if ends:
        low, high = ax.get_ylim()
        label_y = _spread([e[2] for e in ends], gap=0.06 * (high - low))
        for (model, x_end, _), y in zip(ends, label_y):
            ax.annotate(f" {labels.get(model, model)}", (x_end, y), color=INK_2, fontsize=8.5,
                        va="center", annotation_clip=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save(fig, path)


def residuals(daily: pl.DataFrame, acf: list[float], errors: pl.Series, label: str, path: Path) -> Path:
    """Three diagnostics for one model: bias over time, error autocorrelation, error spread."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.2, 1, 1.2]})
    ax = axes[0]
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(daily["delivery_date"], daily["me_30d"], color=SERIES[0], lw=2)
    ax.set_title("Bias: 30-day rolling ME")
    ax.set_ylabel("MW (forecast - actual)")
    ax = axes[1]
    lags = list(range(1, len(acf) + 1))
    ax.bar(lags, acf, color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    band = 1.96 / max(daily.height, 1) ** 0.5
    ax.axhspan(-band, band, color=GRID, zorder=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title("Daily mean error: ACF")
    ax.set_xlabel("lag (days)")
    ax.set_xticks(lags)
    ax = axes[2]
    ax.hist(errors.drop_nulls().to_list(), bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title("Hourly errors")
    ax.set_xlabel("MW")
    fig.suptitle(f"Residual diagnostics - {label}", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, path)
