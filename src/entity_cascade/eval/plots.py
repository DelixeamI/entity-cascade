"""PNG charts for the report. Thin marks, recessive grid, text in ink (never series colour)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from entity_cascade.eval.modes import CurvePoint
from entity_cascade.eval.report import EvalResults

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1-5 in their fixed order (reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FONT = ["Segoe UI", "DejaVu Sans"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
        }
    )


def _clean(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _pct_axis(ax: Axes, axis: str = "both") -> None:
    from matplotlib.ticker import PercentFormatter

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def plot_curves(
    curves: Mapping[str, Mapping[str, Sequence[CurvePoint]]],
    results: EvalResults,
    path: Path,
) -> None:
    """Auto-resolved share vs precision, per mode, val and test side by side."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    target = results.target_precision
    for ax, split in zip(axes, ("val", "test"), strict=True):
        _clean(ax)
        for color, (mode, by_split) in zip(SERIES, curves.items(), strict=False):
            pts = by_split[split]
            ax.plot(
                [p.coverage for p in pts],
                [p.precision for p in pts],
                color=color,
                linewidth=2,
                label=mode,
            )
        ax.axhline(target, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.99, target + 0.004, f"target {target:.0%}", color=INK2, fontsize=9, ha="right")
        zb = next(z for z in results.zone_blocks if z.split == split)
        auto = next(z for z in zb.zones if z.zone == "auto_match")
        if auto.rows:
            ax.plot(
                [auto.share],
                [auto.top1_correct],
                marker="o",
                markersize=9,
                markerfacecolor=INK,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                linestyle="none",
                label="calibrated auto_match zone",
                zorder=5,
            )
        ax.set_title(f"{split}: share of rows resolved automatically vs precision")
        ax.set_xlabel("share of all rows resolved automatically")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.5, 1.01)
        _pct_axis(ax)
    axes[0].set_ylabel("precision of the automatic answers")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.text(
        0.01,
        0.005,
        "RRF scores are heavily tied, so the hybrid (RRF) curve starts at the size of the top tie.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(results: EvalResults, path: Path) -> None:
    """Target precision sweep: what val promised vs what test delivered."""
    _style()
    rows = [t for t in results.tradeoff if t.feasible]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _clean(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for t in rows:
        ax.plot(
            [t.val_auto_share, t.test_auto_share],
            [t.val_auto_precision, t.test_auto_precision],
            color=AXIS,
            linewidth=1,
            zorder=1,
        )
    ax.plot(
        [t.val_auto_share for t in rows],
        [t.val_auto_precision for t in rows],
        marker="o",
        markersize=8,
        markerfacecolor=SURFACE,
        markeredgecolor=SERIES[0],
        markeredgewidth=2,
        linestyle="none",
        label="val (where thresholds were fitted)",
        zorder=3,
    )
    ax.plot(
        [t.test_auto_share for t in rows],
        [t.test_auto_precision for t in rows],
        marker="o",
        markersize=8,
        markerfacecolor=SERIES[1],
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        linestyle="none",
        label="test (held out)",
        zorder=4,
    )
    for t in rows:
        ax.annotate(
            t.label.replace(" (Wilson 95% bound)", " W"),
            (t.test_auto_share, t.test_auto_precision),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            color=INK2,
        )
    ax.set_xlabel("share of rows resolved automatically")
    ax.set_ylabel("precision of the automatic answers")
    ax.set_title("Target precision vs automation (labels: target; W = Wilson bound)")
    _pct_axis(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zones(results: EvalResults, path: Path) -> None:
    """Zone shares for val and test, and test after the LLM if it was run."""
    _style()
    bars: list[tuple[str, list[tuple[str, float]]]] = []
    for zb in results.zone_blocks:
        bars.append((f"{zb.split}, by zone", [(z.zone, z.share) for z in zb.zones]))
    llm_row = next((s for s in results.systems if s.name == "cascade + LLM arbiter"), None)
    if llm_row is not None and results.llm is not None and not results.llm.stand_in:
        m = llm_row.metrics
        bars.append(
            (
                "test, who answers",
                [
                    ("rules (auto_match + no_match)", m.automation_rate - m.llm_share),
                    ("LLM arbiter", m.llm_share),
                    ("human", m.human_share),
                ],
            )
        )
    names = ["auto_match", "llm_arbiter", "human_review", "no_match"]
    colors = {n: SERIES[i] for i, n in enumerate(names)}
    colors.update({"rules (auto_match + no_match)": SERIES[0], "LLM arbiter": SERIES[1]})
    colors["human"] = SERIES[2]
    fig, ax = plt.subplots(figsize=(8.5, 1.4 + 0.9 * len(bars)))
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    seen: dict[str, bool] = {}
    for y, (_, parts) in enumerate(reversed(bars)):
        left = 0.0
        for name, share in parts:
            if share != share or share <= 0:
                continue
            ax.barh(
                y,
                share,
                left=left,
                color=colors[name],
                edgecolor=SURFACE,
                linewidth=2,
                height=0.6,
                label=name if name not in seen else None,
            )
            seen[name] = True
            if share >= 0.06:
                ax.text(
                    left + share / 2,
                    y,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    color="#0b0b0b",
                    fontsize=9,
                )
            left += share
    ax.set_yticks(range(len(bars)), [b[0] for b in reversed(bars)], color=INK2)
    ax.set_xlim(0, 1)
    _pct_axis(ax, "x")
    ax.set_title("Where rows go", loc="left", color=INK, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(
    results: EvalResults,
    curves: Mapping[str, Mapping[str, Sequence[CurvePoint]]],
    out_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(curves, results, out_dir / "curve_automation_precision.png")
    plot_tradeoff(results, out_dir / "tradeoff_target_precision.png")
    plot_zones(results, out_dir / "zones.png")
    return [
        "curve_automation_precision.png",
        "tradeoff_target_precision.png",
        "zones.png",
    ]
