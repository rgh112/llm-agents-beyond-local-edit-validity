#!/usr/bin/env python3
"""Effect-size tables for ablation experiments.

The regular summary tables report each condition independently. This analyzer
builds reviewer-facing delta tables: intervention minus control, with
bootstrap CIs from episode-level counts when available.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def ci_diff(a_success, a_total, b_success, b_total, n_boot=5000, seed=0):
    if a_total <= 0 or b_total <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    a = rng.binomial(a_total, a_success / a_total, size=n_boot) / a_total
    b = rng.binomial(b_total, b_success / b_total, size=n_boot) / b_total
    d = b - a
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pct(x):
    if x is None:
        return "-"
    return f"{100 * float(x):.0f}%"


def load_rows(pattern):
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)
        for row in data.get("results", []):
            r = dict(row)
            r["_source"] = path
            rows.append(r)
    return rows


def key_for(row, fields):
    return tuple(str(row.get(f, "")) for f in fields)


def index_rows(rows, fields):
    out = {}
    for row in rows:
        out[key_for(row, fields)] = row
    return out


def add_delta(lines, label, control, treatment):
    if not control or not treatment:
        return
    dsr = float(treatment.get("SR", 0)) - float(control.get("SR", 0))
    lo, hi = ci_diff(
        int(control.get("n_success", 0)),
        int(control.get("n_total", 0)),
        int(treatment.get("n_success", 0)),
        int(treatment.get("n_total", 0)),
    )
    dlv = _diff(treatment.get("local_valid_edit_rate"), control.get("local_valid_edit_rate"))
    dclean = _diff(treatment.get("SR_given_surface_clean"), control.get("SR_given_surface_clean"))
    dstruct = _diff_rate(
        treatment.get("structural_failure_count"), treatment.get("n_total"),
        control.get("structural_failure_count"), control.get("n_total"),
    )
    lines.append("| " + " | ".join([
        label,
        str(control.get("env", "")),
        str(control.get("model", "")),
        str(control.get("prompt", "")),
        str(control.get("memory", "")),
        str(control.get("controller", "")),
        str(control.get("sensitivity_setting", "")),
        str(treatment.get("prompt", "")),
        str(treatment.get("memory", "")),
        str(treatment.get("controller", "")),
        str(treatment.get("sensitivity_setting", "")),
        f"{control.get('n_success', 0)}/{control.get('n_total', 0)}",
        f"{treatment.get('n_success', 0)}/{treatment.get('n_total', 0)}",
        pct(dsr),
        f"[{pct(lo)}, {pct(hi)}]",
        pct(dlv),
        pct(dclean),
        pct(dstruct),
    ]) + " |")


def _diff(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _diff_rate(a_count, a_total, b_count, b_total):
    if a_count is None or b_count is None or not a_total or not b_total:
        return None
    return float(a_count) / float(a_total) - float(b_count) / float(b_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="Glob for *_summary.json files.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_rows(args.summary)
    headers = [
        "ablation", "env", "model", "control prompt", "control memory",
        "control controller", "control setting", "treat prompt",
        "treat memory", "treat controller", "treat setting", "control SR",
        "treat SR", "Delta SR", "Delta SR 95% CI", "Delta LocalValid",
        "Delta SR given SurfaceClean", "Delta structural events/ep",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    # Prompt ablations: few-shot-format and structural prompt against zero-shot.
    prompt_index = index_rows(
        rows,
        ["env", "model", "memory", "controller", "sensitivity_setting", "prompt"],
    )
    for row in rows:
        prompt = row.get("prompt")
        if prompt in {"few_shot_format", "scaffold", "self_check", "self_check_generic", "few_shot_strategy"}:
            control_key = key_for(
                {**row, "prompt": "zero_shot"},
                ["env", "model", "memory", "controller", "sensitivity_setting", "prompt"],
            )
            add_delta(lines, f"prompt:{prompt}-vs-zero_shot", prompt_index.get(control_key), row)

    # Memory ablations against state_only.
    memory_index = index_rows(rows, ["env", "model", "prompt", "controller", "memory"])
    for row in rows:
        memory = row.get("memory")
        if memory and memory != "state_only":
            control_key = key_for(
                {**row, "memory": "state_only"},
                ["env", "model", "prompt", "controller", "memory"],
            )
            add_delta(lines, f"memory:{memory}-vs-state_only", memory_index.get(control_key), row)

    # Controller ablations against greedy.
    ctrl_index = index_rows(rows, ["env", "model", "prompt", "memory", "controller"])
    for row in rows:
        ctrl = row.get("controller")
        if ctrl and ctrl != "greedy":
            control_key = key_for(
                {**row, "controller": "greedy"},
                ["env", "model", "prompt", "memory", "controller"],
            )
            add_delta(lines, f"controller:{ctrl}-vs-greedy", ctrl_index.get(control_key), row)

    # Sensitivity settings against default within the same env/model/prompt.
    sens_index = index_rows(rows, ["env", "model", "prompt", "memory", "controller", "sensitivity_setting"])
    for row in rows:
        setting = row.get("sensitivity_setting")
        if setting and setting != "default":
            control_key = key_for(
                {**row, "sensitivity_setting": "default"},
                ["env", "model", "prompt", "memory", "controller", "sensitivity_setting"],
            )
            add_delta(lines, f"sensitivity:{setting}-vs-default", sens_index.get(control_key), row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
