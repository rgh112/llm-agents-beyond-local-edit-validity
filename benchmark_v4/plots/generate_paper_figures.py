#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Figure 1: Main prompt grouped bar chart
Figure 2: History ablation line plot
Figure 3: Failure composition stacked bars
Figure 4: GB1 operating-point calibration plot
"""
import json
import sys, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path("results_v4/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──
with open("results_v4/main_prompt_20260311_223450/main_prompt_summary.json") as f:
    main_data = json.load(f)

with open("results_v4/gb1_main_rerun_20260312_132622/gb1_main_rerun_summary.json") as f:
    gb1_rerun = json.load(f)

with open("results_v4/history_ablation_20260312_101111/history_ablation_summary.json") as f:
    history_data = json.load(f)

with open("results_v4/gb1_calibration_appendix_20260312_115344/calibration_appendix_summary.json") as f:
    calib_data = json.load(f)

# Replace GB1 results in main_data with rerun
main_results = [r for r in main_data["results"] if r["env"] != "gb1_sequence"]
main_results.extend(gb1_rerun["results"])


# ═══════════════════════════════════════════════════════════
# Figure 1: Main Prompt Grouped Bar Chart
# ═══════════════════════════════════════════════════════════
def fig1_main_prompt():
    prompts = ["zero_shot", "few_shot_format", "few_shot_strategy",
               "scaffold", "self_check_generic", "self_check"]
    prompt_labels = ["Zero-shot", "Few-shot\n(format)", "Few-shot\n(strategy)",
                     "Scaffold", "Self-check\n(generic)", "Self-check\n(task)"]
    envs = ["word_ladder", "alloy", "gb1_sequence"]
    env_labels = ["Word Ladder", "Alloy", "GB1 landscape"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Build SR lookup
    sr = {}
    for r in main_results:
        sr[(r["env"], r["prompt"])] = r["SR"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(prompts))
    width = 0.25

    for i, (env, label, color) in enumerate(zip(envs, env_labels, colors)):
        vals = [sr.get((env, p), 0) * 100 for p in prompts]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.85)
        # Value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Prompt Condition")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Effect of Prompt Interventions Across Environments")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_labels)
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotations for key findings
    ax.annotate("no broad\nformat-only lift",
                xy=(1, 3), fontsize=8, color="gray", ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_main_prompt.pdf")
    fig.savefig(OUT_DIR / "fig1_main_prompt.png")
    print(f"  Figure 1 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Figure 2: History Ablation Line Plot
# ═══════════════════════════════════════════════════════════
def fig2_history_ablation():
    h_results = history_data["results"]

    # Build lookup
    sr = {}
    for r in h_results:
        sr[(r["env"], r["prompt"], r["memory"])] = r["SR"]

    mem_levels = ["state_only", "window_1", "window_3"]
    mem_labels = ["M0\n(state only)", "M1\n(last 1)", "M3\n(last 3)"]

    lines = [
        ("word_ladder", "zero_shot", "WL / zero-shot", "#4C72B0", "--", "o"),
        ("word_ladder", "scaffold", "WL / scaffold", "#4C72B0", "-", "s"),
        ("alloy", "zero_shot", "Alloy / zero-shot", "#DD8452", "--", "o"),
        ("alloy", "self_check", "Alloy / self-check", "#DD8452", "-", "s"),
        ("gb1_sequence", "zero_shot", "GB1 / zero-shot", "#55A868", "--", "o"),
        ("gb1_sequence", "self_check", "GB1 / self-check", "#55A868", "-", "s"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(mem_levels))

    for env, prompt, label, color, ls, marker in lines:
        vals = [sr.get((env, prompt, m), 0) * 100 for m in mem_levels]
        ax.plot(x, vals, label=label, color=color, linestyle=ls, marker=marker,
                markersize=7, linewidth=2)

    ax.set_xlabel("Memory Condition")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Effect of Edit History Access on Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(mem_labels)
    ax.set_ylim(-2, 42)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotation arrows for key effects
    ax.annotate("GB1: history\ncatastrophic\n(-30%p)",
                xy=(1, 5), xytext=(1.5, 18),
                arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.5),
                fontsize=8, color="#55A868", ha="center")

    ax.annotate("Alloy: history\nhelpful (+12%p)",
                xy=(2, 25), xytext=(2.3, 35),
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.5),
                fontsize=8, color="#DD8452", ha="center")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_history_ablation.pdf")
    fig.savefig(OUT_DIR / "fig2_history_ablation.png")
    print(f"  Figure 2 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Figure 3: Failure Composition Stacked Bars
# ═══════════════════════════════════════════════════════════
def fig3_failure_composition():
    # Use main results: one representative prompt per env
    # WL: scaffold (best), Alloy: self_check (best), GB1: self_check (best)
    # Plus zero_shot for comparison
    configs = [
        ("word_ladder", "zero_shot", "WL\nzero-shot"),
        ("word_ladder", "scaffold", "WL\nscaffold"),
        ("alloy", "zero_shot", "Alloy\nzero-shot"),
        ("alloy", "self_check", "Alloy\nself-check"),
        ("gb1_sequence", "zero_shot", "GB1\nzero-shot"),
        ("gb1_sequence", "self_check", "GB1\nself-check"),
    ]

    # Failure categories
    surface_keys = {"MALFORMED_ACTION", "INVALID_POSITION", "INVALID_VALUE",
                    "INVALID_WORD", "ILLEGAL_EDIT", "REPEATED_EXACT_EDIT"}
    budget_keys = {"BUDGET_UNAWARE_ACTION"}
    tradeoff_keys = {"OBJECTIVE_TRADEOFF_FAILURE"}
    trap_keys = {"LOCAL_OPTIMUM_TRAP"}
    premature_keys = {"PREMATURE_FINALIZE"}
    other_structural = {"HARD_CONSTRAINT_VIOLATION", "RECOVERY_COST_EXPLOSION",
                        "GLOBAL_FEASIBILITY_LOSS", "OSCILLATION"}

    categories = [
        ("Surface\n(format)", surface_keys, "#AEC7E8"),
        ("Budget\nunaware", budget_keys, "#FFBB78"),
        ("Objective\ntradeoff", tradeoff_keys, "#FF7F0E"),
        ("Local optimum\ntrap", trap_keys, "#D62728"),
        ("Premature\nfinalize", premature_keys, "#9467BD"),
        ("Other\nstructural", other_structural, "#8C564B"),
    ]

    # Build lookup
    event_lookup = {}
    for r in main_results:
        event_lookup[(r["env"], r["prompt"])] = r.get("all_events", {})

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    bar_width = 0.6

    bottoms = np.zeros(len(configs))

    for cat_label, keys, color in categories:
        vals = []
        for env, prompt, _ in configs:
            evts = event_lookup.get((env, prompt), {})
            count = sum(evts.get(k, 0) for k in keys)
            vals.append(count)
        vals = np.array(vals, dtype=float)
        ax.bar(x, vals, bar_width, bottom=bottoms, label=cat_label, color=color, alpha=0.85)
        bottoms += vals

    ax.set_xlabel("Environment / Prompt")
    ax.set_ylabel("Total Event Count (50 episodes)")
    ax.set_title("Failure Mode Decomposition by Environment and Prompt")
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in configs], fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_failure_composition.pdf")
    fig.savefig(OUT_DIR / "fig3_failure_composition.png")
    print(f"  Figure 3 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Figure 4: GB1 Operating-Point Calibration
# ═══════════════════════════════════════════════════════════
def fig4_gb1_calibration():
    c_results = calib_data["results"]

    # Build lookup
    sr = {}
    for r in c_results:
        sr[(r["policy"], r["fitness_target"])] = r["SR"]

    targets = [4.7, 4.8, 5.0]

    policies = [
        ("additive_greedy", "Additive greedy\n(obs-level)", "#AEC7E8", "--", "^"),
        ("obs_additive_greedy", "Obs heuristic", "#C7C7C7", ":", "v"),
        ("zero_shot", "LLM zero-shot", "#4C72B0", "-", "o"),
        ("few_shot_strategy", "LLM few-shot\n(strategy)", "#DD8452", "-", "s"),
        ("privileged_noisy_oracle", "Noisy oracle\n(privileged)", "#55A868", "--", "D"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(targets))

    for policy, label, color, ls, marker in policies:
        vals = [sr.get((policy, ft), 0) * 100 for ft in targets]
        ax.plot(x, vals, label=label, color=color, linestyle=ls, marker=marker,
                markersize=8, linewidth=2)

    # Highlight chosen operating point
    ax.axvline(x=2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(2.05, 58, "chosen\nthreshold", fontsize=8, color="gray", va="bottom")

    ax.set_xlabel("Fitness Target Threshold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("GB1 Operating Point Calibration")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in targets])
    ax.set_ylim(-3, 65)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Shaded regions
    ax.axhspan(0, 5, alpha=0.05, color="red")
    ax.text(0.05, 2, "trivial (all fail)", fontsize=7, color="red", alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_gb1_calibration.pdf")
    fig.savefig(OUT_DIR / "fig4_gb1_calibration.png")
    print(f"  Figure 4 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Generate all
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_main_prompt()
    fig2_history_ablation()
    fig3_failure_composition()
    fig4_gb1_calibration()
    print(f"\nAll figures saved to {OUT_DIR}/")
