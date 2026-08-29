#!/usr/bin/env python3
"""Build scaling visualizations (paper figures + dashboard JSON).

Reads `supplementary/scaling_core_summary.json` and writes:

  EMNLP_Word_Ladder_MARCH/latex/fig8_scaling_curves.{pdf,png}
  EMNLP_Word_Ladder_MARCH/latex/fig9_family_heatmap.{pdf,png}
  supplementary/visualizations/scaling_data.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "supplementary" / "scaling_core_summary.json"
VIZ_DIR = ROOT / "supplementary" / "visualizations"
PAPER_DIR = ROOT / "EMNLP_Word_Ladder_MARCH" / "latex"

# Family palette (Mistral and Ministral share a color since they're the same vendor).
FAMILY_COLOR = {
    "Qwen3": "#1f77b4",       # blue
    "Ministral": "#d62728",   # red
    "Mistral": "#d62728",     # same as Ministral
    "Gemma3": "#9467bd",      # purple
    "Llama": "#2ca02c",       # green
}
FAMILY_DISPLAY = {
    "Qwen3": "Qwen3",
    "Ministral": "Mistral AI",
    "Mistral": "Mistral AI",
    "Gemma3": "Gemma 3",
    "Llama": "Llama",
}

ENVS = ["Word Ladder", "Alloy", "GB1"]
# Default "best of pair" prompt per env (matches main-text reporting choice).
DEFAULT_PROMPT = {"Word Ladder": "Scaffold", "Alloy": "Self-check", "GB1": "Self-check"}


def _load() -> list:
    return json.load(open(SUMMARY))["rows"]


def _save(fig, stem: str):
    pdf = PAPER_DIR / f"{stem}.pdf"
    png = PAPER_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {pdf} and {png}")


def fig8_scaling_curves(rows):
    """Scaling curve: SR vs model size on log-x, per family, three env panels."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ai, env in enumerate(ENVS):
        ax = axes[ai]
        # Group cells by display-family
        by_family = {}
        for r in rows:
            if r["env"] != env or r["prompt"] != DEFAULT_PROMPT[env]:
                continue
            fam_key = FAMILY_DISPLAY[r["family"]]
            by_family.setdefault(fam_key, []).append(r)
        for fam, frows in sorted(by_family.items()):
            xs = [r["size_b"] for r in sorted(frows, key=lambda x: x["size_b"])]
            ys = [100 * r["SR"] for r in sorted(frows, key=lambda x: x["size_b"])]
            color = FAMILY_COLOR.get(
                next((k for k, v in FAMILY_DISPLAY.items() if v == fam), None), "#888"
            )
            ax.plot(xs, ys, marker="o", color=color, label=fam, linewidth=1.8, markersize=7)
            # annotate each marker with size
            for x, y in zip(xs, ys):
                ax.annotate(f"{x}B", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
        ax.set_xscale("log")
        ax.set_xticks([3, 8, 14, 32, 70])
        ax.set_xticklabels(["3", "8", "14", "32", "70"])
        ax.set_xlabel("Model size (B params, log)", fontsize=9)
        if ai == 0:
            ax.set_ylabel("Success rate (%)", fontsize=9)
        ax.set_title(f"{env}\n({DEFAULT_PROMPT[env]} prompt)", fontsize=10)
        ax.set_ylim(0, 70)
        ax.grid(linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ai == 2:
            ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Cross-family scaling: Word Ladder scales, Alloy and GB1 stay structurally hard", fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig, "fig8_scaling_curves")


def fig9_family_heatmap(rows):
    """Cross-family heatmap: 13 models grouped by family/size, columns = envs."""
    # Build ordered model list: by family display order, then size
    family_order = ["Qwen3", "Ministral", "Mistral", "Gemma3", "Llama"]
    model_keys = []
    seen = set()
    for fam in family_order:
        fam_rows = [r for r in rows if r["family"] == fam]
        for m in sorted({r["model"] for r in fam_rows}, key=lambda mm: next(r["size_b"] for r in fam_rows if r["model"] == mm)):
            if m in seen:
                continue
            seen.add(m)
            model_keys.append((fam, m))

    matrix = np.full((len(model_keys), len(ENVS)), np.nan)
    labels = []
    for i, (fam, m) in enumerate(model_keys):
        size = next((r["size_b"] for r in rows if r["model"] == m), "?")
        labels.append(f"{FAMILY_DISPLAY[fam]} {m.split('-')[-1] if False else size}B")
        for j, env in enumerate(ENVS):
            cell = next((r for r in rows if r["model"] == m and r["env"] == env and r["prompt"] == DEFAULT_PROMPT[env]), None)
            if cell is not None:
                matrix[i, j] = cell["SR"]

    # Better labels: family + size, prefixed with short family
    short_fam = {"Qwen3": "Qwen3", "Ministral": "Mistral", "Mistral": "Mistral", "Gemma3": "Gemma", "Llama": "Llama"}
    labels = []
    for fam, m in model_keys:
        size = next((r["size_b"] for r in rows if r["model"] == m), "?")
        labels.append(f"{short_fam[fam]} {size}B")

    fig, ax = plt.subplots(figsize=(5.4, 0.32 * len(model_keys) + 1.3))
    cmap = LinearSegmentedColormap.from_list("blues", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.75, aspect="auto")
    for i in range(len(model_keys)):
        for j in range(len(ENVS)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.42 else "black"
                ax.text(j, i, f"{int(round(100 * v))}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([f"{e}\n({DEFAULT_PROMPT[e]})" for e in ENVS], fontsize=9)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(labels, fontsize=9)
    # Color-code y-tick labels by family
    for i, (fam, _) in enumerate(model_keys):
        ax.get_yticklabels()[i].set_color(FAMILY_COLOR.get(fam, "black"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Success rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    # Family group separators
    boundaries = []
    last_fam = None
    for i, (fam, _) in enumerate(model_keys):
        if last_fam is not None and fam != last_fam:
            boundaries.append(i - 0.5)
        last_fam = fam
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=2)
        ax.axhline(b, color="black", linewidth=0.5)
    ax.set_title("Cross-family scaling matrix (50 seeds per cell)", fontsize=10.5)
    _save(fig, "fig9_family_heatmap")


def write_dashboard_scaling(rows):
    out = VIZ_DIR / "scaling_data.json"
    payload = {
        "envs": ENVS,
        "default_prompt": DEFAULT_PROMPT,
        "families": ["Qwen3", "Ministral", "Mistral", "Gemma3", "Llama"],
        "family_color": FAMILY_COLOR,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


def fig10_main_discrimination(rows):
    """Headline main-paper figure: 13 models × 3 envs, each panel ranked by SR.
    Best per panel boxed. Family-colored bars. Reviewer sees in 5 seconds
    that no model wins all three columns."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    for ai, env in enumerate(ENVS):
        ax = axes[ai]
        cells = [r for r in rows if r["env"] == env and r["prompt"] == DEFAULT_PROMPT[env]]
        cells.sort(key=lambda r: -r["SR"])
        labels = []
        srs = []
        colors = []
        for r in cells:
            family_short = {"Qwen3": "Qwen3", "Ministral": "Mistral", "Mistral": "Mistral",
                            "Gemma3": "Gemma", "Llama": "Llama"}[r["family"]]
            labels.append(f"{family_short} {r['size_b']}B")
            srs.append(100 * r["SR"])
            colors.append(FAMILY_COLOR[r["family"]])
        y = np.arange(len(cells))
        bars = ax.barh(y, srs, color=colors, edgecolor="white", linewidth=0.5)
        # Outline the top bar
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(2.0)
        # Annotate values
        for bar, v in zip(bars, srs):
            ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(round(v))}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        # Color y-tick text by family for instant visual grouping
        for ti, r in enumerate(cells):
            ax.get_yticklabels()[ti].set_color(FAMILY_COLOR[r["family"]])
        ax.set_xlim(0, max(60, max(srs) + 12))
        ax.set_xlabel("Success rate (%)", fontsize=9)
        ax.set_title(f"{env}  ({DEFAULT_PROMPT[env]} prompt)", fontsize=10)
        # Annotate the winner above the top bar
        winner = cells[0]
        ax.text(0.5, -0.45, f"best: {labels[0]}",
                transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold",
                color=FAMILY_COLOR[winner["family"]])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
    # Custom family legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in ["Qwen3", "Ministral", "Gemma3", "Llama"]]
    fig.legend(handles, ["Qwen3", "Mistral AI (Ministral / Mistral-Small)", "Gemma 3", "Llama"],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    fig.suptitle("Different environments select different best models — the benchmark discriminates", fontsize=12, y=1.1)
    fig.tight_layout()
    _save(fig, "fig5_main_discrimination")


def main():
    rows = _load()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig8_scaling_curves(rows)
    fig9_family_heatmap(rows)
    fig10_main_discrimination(rows)
    write_dashboard_scaling(rows)


if __name__ == "__main__":
    main()
