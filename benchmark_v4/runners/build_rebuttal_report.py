#!/usr/bin/env python3
"""Combine Qwen baseline + cross-model results into a single rebuttal report.

Inputs:
  - Qwen summary table (from analyze_summaries.py)
  - Cross-model summary (cross_model_summary.json)
  - Cross-model rebuttal table (from analyze_rebuttal.py on its raw, optional)

Output:
  rebuttal_report.md with:
    - Table R1: cross-model SR comparison (with CIs) by env × prompt × model
    - Table R2: local-validity-conditioned (only for cells with raw analysis)
    - Table R3: surface vs structural failure decomposition

Example:
    python -m benchmark_v4.runners.build_rebuttal_report \
        --qwen-summary results_v4/rebuttal/summary_table_qwen_summary_baseline.json \
        --cross-model results_v4/cross_model_r1_main_*/cross_model_summary.json \
        --cross-rebuttal results_v4/rebuttal/rebuttal_table_cross_model.json \
        --out results_v4/rebuttal/rebuttal_report.md
"""
import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SURFACE_EVENTS = {
    "MALFORMED_ACTION", "INVALID_POSITION", "INVALID_VALUE",
    "INVALID_WORD", "ILLEGAL_EDIT", "REPEATED_EXACT_EDIT",
}
STRUCTURAL_EVENTS = {
    "BUDGET_UNAWARE_ACTION", "LOCAL_OPTIMUM_TRAP",
    "OBJECTIVE_TRADEOFF_FAILURE", "PREMATURE_FINALIZE",
    "HARD_CONSTRAINT_VIOLATION", "RECOVERY_COST_EXPLOSION",
    "GLOBAL_FEASIBILITY_LOSS", "OSCILLATION",
}

ENV_STRUCTURAL_PROMPT = {
    "word_ladder": "scaffold",
    "alloy": "self_check",
    "gb1_sequence": "self_check",
}


def bootstrap_ci(successes, n, n_boot=2000, alpha=0.05, seed=42):
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    data = [1] * successes + [0] * (n - successes)
    boots = []
    for _ in range(n_boot):
        s = sum(rng.choice(data) for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def short_model(name):
    return name.split("/")[-1] if "/" in name else name


def collect_qwen_rows(path):
    """Returns rows in the same shape as cross-model rows."""
    d = json.load(open(path))
    rows = []
    for r in d["rows"]:
        rows.append({
            "model": r["model"],
            "env": r["env"],
            "prompt": r["prompt"],
            "memory": r["memory"],
            "n": r["n"],
            "n_success": r["n_success"],
            "SR": r["SR"],
            "SR_ci95": r["SR_ci95"],
            "surface_failure_count": r["surface_failure_count"],
            "structural_failure_count": r["structural_failure_count"],
            "terminal_failures": r["terminal_failures"],
            "local_valid_edit_rate": None,
            "SR_given_surface_clean": None,
            "SR_given_surface_clean_ci95": None,
            "n_surface_clean_episodes": None,
            "top_residual_structural": None,
            "source": "qwen_summary",
        })
    return rows


def collect_cross_rows(path, raw_rebuttal_path=None):
    d = json.load(open(path))
    rows = []
    raw_rows = {}
    if raw_rebuttal_path and Path(raw_rebuttal_path).exists():
        rd = json.load(open(raw_rebuttal_path))
        for r in rd.get("rows", []):
            # raw episode logs use short model name; index by short name
            raw_rows[(r["env"], short_model(r["model"]), r["prompt"], r["memory"])] = r
    for r in d.get("results", []):
        n = int(r["n_total"])
        n_succ = int(r["n_success"])
        lo, hi = bootstrap_ci(n_succ, n)
        key = (r["env"], short_model(r["model"]), r["prompt"], r["memory"])
        rr = raw_rows.get(key, {})
        rows.append({
            "model": r["model"],
            "env": r["env"],
            "prompt": r["prompt"],
            "memory": r["memory"],
            "n": n,
            "n_success": n_succ,
            "SR": r["SR"],
            "SR_ci95": [lo, hi],
            "surface_failure_count": r.get("surface_failure_count", 0),
            "structural_failure_count": r.get("structural_failure_count", 0),
            "terminal_failures": r.get("terminal_failures", {}),
            "local_valid_edit_rate":
                r.get("local_valid_edit_rate")
                if r.get("local_valid_edit_rate") is not None
                else rr.get("local_valid_edit_rate"),
            "SR_given_surface_clean":
                r.get("SR_given_surface_clean")
                if r.get("SR_given_surface_clean") is not None
                else rr.get("SR_given_surface_clean"),
            "SR_given_surface_clean_ci95": rr.get("SR_given_surface_clean_ci95"),
            "n_surface_clean_episodes":
                r.get("n_surface_clean_episodes")
                if r.get("n_surface_clean_episodes") is not None
                else rr.get("n_surface_clean_episodes"),
            "top_residual_structural": rr.get("top_residual_structural"),
            "source": "cross_model",
        })
    return rows


def fmt_pct(x):
    return f"{x:.0%}" if x is not None else "—"


def fmt_ci(ci):
    if ci is None:
        return "—"
    lo, hi = ci
    return f"[{lo:.0%}, {hi:.0%}]"


def build_table_r1(rows):
    """Cross-model SR comparison: env × prompt × model."""
    out = ["## Table R1 — Cross-model SR comparison\n"]
    out.append(
        "Each cell aggregates 50 episodes (`state_only` memory). "
        "**95% CI**: episode-level bootstrap percentile (2,000 resamples). "
        "Structural prompt = `scaffold` for word_ladder, `self_check` for alloy and gb1_sequence.\n"
    )
    by_env = defaultdict(list)
    for r in rows:
        if r["prompt"] not in {"zero_shot", "few_shot_format",
                               ENV_STRUCTURAL_PROMPT[r["env"]]}:
            continue
        by_env[r["env"]].append(r)

    for env in ["word_ladder", "alloy", "gb1_sequence"]:
        if env not in by_env:
            continue
        out.append(f"\n### {env}\n")
        out.append(
            "| Model | zero_shot | few_shot_format | "
            f"{ENV_STRUCTURAL_PROMPT[env]} (structural) |"
        )
        out.append("|---|---|---|---|")
        # group by model
        by_model = defaultdict(dict)
        for r in by_env[env]:
            by_model[r["model"]][r["prompt"]] = r
        for model in sorted(by_model):
            cells = []
            for p in ["zero_shot", "few_shot_format", ENV_STRUCTURAL_PROMPT[env]]:
                r = by_model[model].get(p)
                if r is None:
                    cells.append("—")
                else:
                    cells.append(
                        f"{r['SR']:.0%} {fmt_ci(r['SR_ci95'])} "
                        f"({r['n_success']}/{r['n']})"
                    )
            out.append(f"| {short_model(model)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def build_table_r2(rows):
    """Local-validity-conditioned analysis for cells with that data."""
    out = ["## Table R2 — Local validity vs trajectory-level success\n"]
    out.append(
        "Tests whether failures are *structural* rather than *schema-following*. "
        "**LocalValid** = fraction of edit attempts the env accepted as valid. "
        "**SurfClean** = episodes with no surface-failure event. "
        "**SR\\|SurfClean** = success rate restricted to surface-clean episodes. "
        "If `SR\\|SurfClean` ≪ 100% with high `LocalValid`, models produce well-formed valid edits "
        "yet still fail the delayed global objective.\n"
    )
    eligible = [r for r in rows if r.get("local_valid_edit_rate") is not None]
    if not eligible:
        out.append("\n_Per-episode raw logs not available for local-validity analysis._\n")
        return "\n".join(out) + "\n"

    by_env = defaultdict(list)
    for r in eligible:
        if r["prompt"] not in {"zero_shot", "few_shot_format",
                               ENV_STRUCTURAL_PROMPT[r["env"]]}:
            continue
        by_env[r["env"]].append(r)

    for env in ["word_ladder", "alloy", "gb1_sequence"]:
        if env not in by_env:
            continue
        out.append(f"\n### {env}\n")
        out.append(
            "| Model | Prompt | LocalValid | SurfClean / N | SR\\|SurfClean | Top residual structural |"
        )
        out.append("|---|---|---:|---:|---|---|")
        for r in sorted(by_env[env], key=lambda x: (x["model"], x["prompt"])):
            sr_clean = r["SR_given_surface_clean"]
            sr_clean_str = "—"
            if sr_clean is not None:
                ci_str = fmt_ci(r.get("SR_given_surface_clean_ci95"))
                sr_clean_str = f"{sr_clean:.0%} {ci_str}"
            top = r.get("top_residual_structural") or "—"
            out.append(
                f"| {short_model(r['model'])} | {r['prompt']} | "
                f"{fmt_pct(r['local_valid_edit_rate'])} | "
                f"{r.get('n_surface_clean_episodes', 0)}/{r['n']} | "
                f"{sr_clean_str} | {top} |"
            )
    return "\n".join(out) + "\n"


def build_table_r3(rows):
    """Surface vs structural failure decomposition."""
    out = ["## Table R3 — Surface vs structural failure decomposition\n"]
    out.append(
        "Total event counts across all episodes in the cell. "
        "*Surface*: malformed action / invalid position / invalid value / invalid word / "
        "illegal edit / repeated edit. "
        "*Structural*: budget-unaware action / local-optimum trap / "
        "objective-tradeoff failure / premature finalize / hard-constraint violation / "
        "recovery-cost explosion / global-feasibility loss / oscillation.\n"
    )
    by_env = defaultdict(list)
    for r in rows:
        if r["prompt"] not in {"zero_shot", "few_shot_format",
                               ENV_STRUCTURAL_PROMPT[r["env"]]}:
            continue
        by_env[r["env"]].append(r)

    for env in ["word_ladder", "alloy", "gb1_sequence"]:
        if env not in by_env:
            continue
        out.append(f"\n### {env}\n")
        out.append("| Model | Prompt | SR | Surface | Structural | Surf:Struct | Top terminal failures |")
        out.append("|---|---|---:|---:|---:|---|---|")
        for r in sorted(by_env[env], key=lambda x: (x["model"], x["prompt"])):
            surf = r["surface_failure_count"]
            struct = r["structural_failure_count"]
            ratio = "—"
            if surf + struct > 0:
                if struct == 0:
                    ratio = "all-surface"
                elif surf == 0:
                    ratio = "all-struct"
                else:
                    ratio = f"{surf}/{struct} ≈ {surf/struct:.1f}"
            top = ", ".join(
                f"{k}:{v}" for k, v in
                sorted(r["terminal_failures"].items(), key=lambda x: -x[1])[:3]
            ) or "—"
            out.append(
                f"| {short_model(r['model'])} | {r['prompt']} | {r['SR']:.0%} | "
                f"{surf} | {struct} | {ratio} | {top} |"
            )
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-summary", required=True,
                    help="Path to summary_table_qwen_summary_baseline.json (output of analyze_summaries.py).")
    ap.add_argument("--cross-model", required=True,
                    help="Glob pattern or path to cross_model_summary.json.")
    ap.add_argument("--cross-rebuttal", default=None,
                    help="Optional rebuttal_table_*.json from analyze_rebuttal.py over the cross-model raw logs.")
    ap.add_argument("--out", required=True,
                    help="Output markdown path.")
    args = ap.parse_args()

    qwen_rows = collect_qwen_rows(args.qwen_summary)
    cross_paths = sorted(glob.glob(args.cross_model)) if "*" in args.cross_model else [args.cross_model]
    cross_rows = []
    for cp in cross_paths:
        cross_rows.extend(collect_cross_rows(cp, args.cross_rebuttal))

    all_rows = qwen_rows + cross_rows

    # Build report
    lines = []
    lines.append("# Rebuttal evidence: cross-model sanity check + bootstrap CIs + local-validity conditioning\n")
    lines.append("This report aggregates evidence to address reviewer concerns:\n")
    lines.append("1. **Single-model evidence** — extended to multiple model families via OpenRouter.\n")
    lines.append("2. **Aggregation/stability** — episode-level bootstrap 95% CIs added to all main success rates.\n")
    lines.append("3. **Format vs structural failure** — surface-clean conditioning isolates trajectory-level construction failure from schema-following errors.\n")
    lines.append("\nAll numbers are restricted to `state_only` memory and three prompts per environment "
                 "(zero_shot, few_shot_format, env-specific structural).\n")

    n_qwen = len(qwen_rows)
    n_cross = len(cross_rows)
    cross_models = sorted({r["model"] for r in cross_rows})
    lines.append(f"\n_Sources: Qwen baseline ({n_qwen} cells, summary JSONs) + cross-model sweep "
                 f"({n_cross} cells across {len(cross_models)} additional models: {', '.join(cross_models)})._\n")

    lines.append("\n" + build_table_r1(all_rows))
    lines.append("\n" + build_table_r2(all_rows))
    lines.append("\n" + build_table_r3(all_rows))

    lines.append("\n## Suggested rebuttal language\n")
    lines.append(
        "> We agree that single-model evidence does not establish a universal claim. "
        "We added a cross-model sanity check on the same three environments under three "
        "prompt conditions per environment. The qualitative diagnostic patterns we report "
        "in the paper recur across model families: (i) format-only exemplars do not "
        "consistently improve, and frequently match or under-perform, zero-shot baselines; "
        "(ii) structural-event counts dominate surface-event counts in alloy and gb1_sequence "
        "across all models; (iii) where per-episode logs are available, success rate "
        "conditioned on surface-clean trajectories remains substantially below 100%, "
        "indicating that models produce locally valid edits yet still fail the delayed "
        "global objective. We also added 95% bootstrap CIs to all main success-rate "
        "tables to clarify the stability of the reported effects; many quantitative "
        "differences across prompts have overlapping intervals, and we will frame them "
        "as qualitative diagnostic patterns rather than significant separations.\n"
    )

    md = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"Wrote {args.out}")
    print(f"  qwen rows: {n_qwen}; cross rows: {n_cross}; cross models: {cross_models}")


if __name__ == "__main__":
    main()
