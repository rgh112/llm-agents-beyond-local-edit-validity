#!/usr/bin/env python3
"""Paired ablation analysis from raw episode logs.

This analyzer compares intervention/control conditions on matched
env-model-seed units, which is stronger than comparing aggregate rates.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_episodes(pattern):
    episodes = []
    for path in glob.glob(pattern, recursive=True):
        if not path.endswith(".json"):
            continue
        try:
            with open(path) as f:
                ep = json.load(f)
        except Exception:
            continue
        if "success" not in ep or "steps" not in ep:
            continue
        ep["_path"] = path
        ep["controller"] = _controller_from_regime(ep.get("regime", ""))
        ep["sensitivity_setting"] = (ep.get("final_metrics") or {}).get("sensitivity_setting", "")
        episodes.append(ep)
    return episodes


def _controller_from_regime(regime):
    if ":" in regime:
        return regime.split(":", 1)[1].split(":", 1)[0]
    return ""


def unit_key(ep, fields):
    return tuple(str(ep.get(f, "")) for f in fields)


def cond_key(ep, varied):
    return str(ep.get(varied, ""))


def bootstrap_mean(vals, n_boot=5000, seed=0):
    vals = np.array(vals, dtype=float)
    if len(vals) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(episodes, *, varied, control, treatments, match_fields, label):
    by_unit = defaultdict(dict)
    for ep in episodes:
        by_unit[unit_key(ep, match_fields)][cond_key(ep, varied)] = ep

    rows = []
    for treatment in treatments:
        diffs = []
        clean_diffs = []
        struct_diffs = []
        n_control_success = 0
        n_treat_success = 0
        for conds in by_unit.values():
            if control not in conds or treatment not in conds:
                continue
            c = conds[control]
            t = conds[treatment]
            cs = 1.0 if c.get("success") else 0.0
            ts = 1.0 if t.get("success") else 0.0
            diffs.append(ts - cs)
            n_control_success += int(cs)
            n_treat_success += int(ts)
            clean_diffs.append(_surface_clean_success(t) - _surface_clean_success(c))
            struct_diffs.append(_structural_events(t) - _structural_events(c))
        mean, lo, hi = bootstrap_mean(diffs)
        clean_mean, clean_lo, clean_hi = bootstrap_mean(clean_diffs)
        struct_mean, struct_lo, struct_hi = bootstrap_mean(struct_diffs)
        rows.append({
            "ablation": f"{label}:{treatment}-vs-{control}",
            "matched_n": len(diffs),
            "control_success": n_control_success,
            "treatment_success": n_treat_success,
            "delta_sr": mean,
            "delta_sr_low": lo,
            "delta_sr_high": hi,
            "delta_surface_clean_success": clean_mean,
            "delta_surface_clean_success_low": clean_lo,
            "delta_surface_clean_success_high": clean_hi,
            "delta_structural_events": struct_mean,
            "delta_structural_events_low": struct_lo,
            "delta_structural_events_high": struct_hi,
        })
    return rows


SURFACE = {
    "MALFORMED_ACTION", "INVALID_POSITION", "INVALID_VALUE",
    "INVALID_WORD", "ILLEGAL_EDIT", "REPEATED_EXACT_EDIT",
}
STRUCTURAL = {
    "BUDGET_UNAWARE_ACTION", "LOCAL_OPTIMUM_TRAP",
    "OBJECTIVE_TRADEOFF_FAILURE", "PREMATURE_FINALIZE",
    "HARD_CONSTRAINT_VIOLATION", "RECOVERY_COST_EXPLOSION",
    "GLOBAL_FEASIBILITY_LOSS", "OSCILLATION",
}


def _events(ep):
    out = []
    for step in ep.get("steps", []):
        out.extend(str(e) for e in step.get("events", []))
    return out


def _surface_clean_success(ep):
    events = set(_events(ep))
    return 1.0 if ep.get("success") and not (events & SURFACE) else 0.0


def _structural_events(ep):
    return sum(1 for e in _events(ep) if e in STRUCTURAL)


def fmt(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_glob", help="Glob such as 'results_v4/run/raw/*.json'.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    episodes = load_episodes(args.raw_glob)
    rows = []
    rows.extend(paired_table(
        episodes,
        varied="prompt_condition",
        control="zero_shot",
        treatments=["few_shot_format", "scaffold", "self_check", "self_check_generic", "few_shot_strategy"],
        match_fields=["env_name", "model", "memory_condition", "controller", "sensitivity_setting", "seed"],
        label="prompt",
    ))
    rows.extend(paired_table(
        episodes,
        varied="memory_condition",
        control="state_only",
        treatments=["window_1", "window_3", "full_history", "summary", "best_state"],
        match_fields=["env_name", "model", "prompt_condition", "controller", "seed"],
        label="memory",
    ))
    rows.extend(paired_table(
        episodes,
        varied="controller",
        control="greedy",
        treatments=["greedy_sampled", "self_consistency", "beam", "loop_avoidant", "independent_restart"],
        match_fields=["env_name", "model", "prompt_condition", "memory_condition", "seed"],
        label="controller",
    ))
    rows.extend(paired_table(
        episodes,
        varied="sensitivity_setting",
        control="default",
        treatments=[
            "heavy_scaling_off", "heavy_scaling_strong", "recovery_off",
            "recovery_strict", "uts_relaxed", "uts_strict",
            "density_relaxed", "density_strict", "noise_off", "noise_high",
            "easy_combined", "hard_combined", "additive_evaluator",
            "threshold_4p7", "threshold_4p8", "threshold_5p2",
            "stability_gate_off", "stability_gate_strict",
        ],
        match_fields=["env_name", "model", "prompt_condition", "memory_condition", "controller", "seed"],
        label="sensitivity",
    ))

    headers = [
        "ablation", "matched N", "control success", "treatment success",
        "Delta SR", "Delta SR 95% CI", "Delta surface-clean success",
        "Delta surface-clean 95% CI", "Delta structural events",
        "Delta structural 95% CI",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        if r["matched_n"] == 0:
            continue
        lines.append("| " + " | ".join([
            r["ablation"],
            str(r["matched_n"]),
            str(r["control_success"]),
            str(r["treatment_success"]),
            fmt(r["delta_sr"]),
            f"[{fmt(r['delta_sr_low'])}, {fmt(r['delta_sr_high'])}]",
            fmt(r["delta_surface_clean_success"]),
            f"[{fmt(r['delta_surface_clean_success_low'])}, {fmt(r['delta_surface_clean_success_high'])}]",
            f"{r['delta_structural_events']:.2f}" if r["delta_structural_events"] is not None else "-",
            (
                f"[{r['delta_structural_events_low']:.2f}, {r['delta_structural_events_high']:.2f}]"
                if r["delta_structural_events_low"] is not None else "-"
            ),
        ]) + " |")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output} from {len(episodes)} raw episodes")


if __name__ == "__main__":
    main()
