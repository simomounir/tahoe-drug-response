"""Case study figures as inline SVG (phase 5a spec C7).

Drawn with matplotlib using sentinel colours that are then replaced by the page's CSS custom properties, so the
figures follow light/dark mode: series → `var(--series-n)`, text and axes → `currentColor`, grid → `var(--grid)`.
Every data mark carries a native `<title>` tooltip. Palette: dataviz reference categorical slots 1–3, validated in
both modes; the light-mode contrast warning is relieved by marker shapes, direct labels and the data tables on the page.
"""
from __future__ import annotations

import io
import re
from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SPLIT_ORDER = ["random", "unseen_cell_line", "unseen_drug", "both_unseen"]
SPLIT_LABELS = {"random": "random", "unseen_cell_line": "new\ncell line", "unseen_drug": "new\ndrug", "both_unseen": "new drug +\nnew line"}
MODEL_SLOT = {"global_mean": 1, "ridge": 2, "neural": 3}  # colour follows the model on every figure
MODEL_MARKER = {"global_mean": "s", "ridge": "D", "neural": "^"}
MODEL_LABEL = {"global_mean": "per-dose mean", "ridge": "ridge", "neural": "neural"}

_SERIES = {slot: f"#0100{slot:02x}" for slot in range(1, 4)}  # sentinels, never shown
_TEXT, _GRID = "#010a0a", "#010b0b"


def _style():
    plt.rcParams.update({
        "svg.fonttype": "none", "font.size": 10, "font.family": "sans-serif",
        "text.color": _TEXT, "axes.labelcolor": _TEXT, "axes.edgecolor": _GRID, "xtick.color": _TEXT, "ytick.color": _TEXT,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def _to_svg(fig, tips: dict[str, str]) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    for slot, hexcode in _SERIES.items():
        svg = svg.replace(hexcode, f"var(--series-{slot})")
    svg = svg.replace(_TEXT, "currentColor").replace(_GRID, "var(--grid)")
    svg = re.sub(r"font-family:[^;\"]*", "font-family: inherit", svg)
    svg = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', r'<svg\1 width="100%" role="img"', svg, count=1)
    for gid, text in tips.items():
        svg = svg.replace(f'<g id="{gid}">', f'<g id="{gid}"><title>{escape(text)}</title>', 1)
    return svg


def degradation_svg(rows: list[dict]) -> str:
    """Median de_pearson with 95% intervals per model across the splits, easiest to hardest."""
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(SPLIT_ORDER))
    tips = {}
    for model, slot in MODEL_SLOT.items():
        by = {r["split"]: r for r in rows if r["model"] == model}
        y = np.array([by[s]["estimate"] for s in SPLIT_ORDER])
        colour = _SERIES[slot]
        ax.plot(x, y, color=colour, linewidth=2, zorder=2)
        for i, s in enumerate(SPLIT_ORDER):
            r = by[s]
            gid = f"deg-{model}-{s}"
            ax.errorbar([x[i]], [r["estimate"]], yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt=MODEL_MARKER[model],
                        color=colour, markersize=7, capsize=3, linewidth=1.5, zorder=3, gid=gid)
            tips[gid] = f"{MODEL_LABEL[model]} · {s}: {r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        # Direct label at the easiest split, where the models are furthest apart (they converge on the hardest one).
        ax.annotate(MODEL_LABEL[model], (x[0], y[0]), xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
                    color=colour, fontsize=10)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_xlim(-1.1, len(x) - 0.6)
    ax.set_ylabel("median DE-gene Pearson")
    ax.grid(axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], color=_SERIES[s], marker=MODEL_MARKER[m], linewidth=2, label=MODEL_LABEL[m])
                       for m, s in MODEL_SLOT.items()], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return _to_svg(fig, tips)


def failure_case_svg(genes: list[str], truth: np.ndarray, preds: dict[str, np.ndarray], max_genes: int = 20) -> str:
    """Dot plot: one row per DE gene (largest |measured| first, up to `max_genes`), measured vs each model's prediction."""
    _style()
    order = np.argsort(-np.abs(truth), kind="stable")[:max_genes]
    order = order[np.argsort(-truth[order], kind="stable")]  # up-regulated at the top
    n = len(order)
    fig, ax = plt.subplots(figsize=(7.0, 0.28 * n + 1.2))
    y = np.arange(n)[::-1]
    tips = {}
    ax.axvline(0, color=_GRID, linewidth=1)
    for model, slot in MODEL_SLOT.items():
        if model not in preds:
            continue
        for row, gi in zip(y, order):
            gid = f"fc-{model}-{gi}"
            ax.plot([preds[model][gi]], [row], MODEL_MARKER[model], color=_SERIES[slot], markersize=6, gid=gid, zorder=3)
            tips[gid] = f"{genes[gi]} · {MODEL_LABEL[model]}: {preds[model][gi]:+.2f}"
    for row, gi in zip(y, order):
        gid = f"fc-measured-{gi}"
        ax.plot([truth[gi]], [row], "o", color=_TEXT, markersize=8, gid=gid, zorder=4)
        tips[gid] = f"{genes[gi]} · measured: {truth[gi]:+.2f}"
    ax.set_yticks(y, [genes[gi] for gi in order])
    ax.set_xlabel("logFC vs DMSO")
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=_TEXT, marker="o", linestyle="", markersize=8, label="measured")]
    handles += [plt.Line2D([], [], color=_SERIES[s], marker=MODEL_MARKER[m], linestyle="", label=MODEL_LABEL[m])
                for m, s in MODEL_SLOT.items() if m in preds]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    return _to_svg(fig, tips)
