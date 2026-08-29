#!/usr/bin/env python3
"""Add bootstrap CIs and compact Markdown tables to experiment summaries."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "benchmark_v4").exists() and (parent / "results_v4").exists():
            return parent
    return here.parents[2]


ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark_v4.model_registry import get_model_metadata


def ci_from_counts(successes: int, total: int, n_boot: int = 2000, seed: int = 0):
    if total <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = rng.binomial(total, successes / total, size=n_boot) / total
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fmt_pct(x):
    if x is None:
        return "-"
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="Path or glob to *_summary.json")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.summary))
    if not paths:
        raise FileNotFoundError(args.summary)

    rows = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for row in data.get("results", []):
            row = dict(row)
            meta = get_model_metadata(row.get("model", ""))
            for key, value in meta.items():
                row.setdefault(f"model_{key}", value)
            ci = ci_from_counts(row.get("n_success", 0), row.get("n_total", 0))
            row["ci_low"], row["ci_high"] = ci
            row["source"] = path
            rows.append(row)

    headers = [
        "env", "model", "prompt", "memory", "controller", "temp", "scorer",
        "family", "scale", "scale_bin", "panel", "openness", "setting", "N", "SR", "95% CI", "LocalValid",
        "SR given SurfaceClean", "Calls/Ep", "Tokens/Ep", "SR/100Calls",
        "SR/100kTok", "Surface", "Structural", "Top failures",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        top_fail = ", ".join(
            f"{k}:{v}" for k, v in sorted(
                row.get("terminal_failures", {}).items(),
                key=lambda kv: -kv[1],
            )[:3]
        )
        lines.append("| " + " | ".join([
            str(row.get("env", "")),
            str(row.get("model", "")),
            str(row.get("prompt", "")),
            str(row.get("memory", "")),
            str(row.get("controller", "")),
            str(row.get("temperature", "")),
            str(row.get("scorer", "")),
            str(row.get("model_family", "")),
            str(row.get("model_scale", "")),
            str(row.get("model_scale_bin", "")),
            str(row.get("model_panel", "")),
            str(row.get("model_openness", "")),
            str(row.get("sensitivity_setting", "")),
            str(row.get("n_total", "")),
            fmt_pct(row.get("SR")),
            f"[{fmt_pct(row.get('ci_low'))}, {fmt_pct(row.get('ci_high'))}]",
            fmt_pct(row.get("local_valid_edit_rate")),
            fmt_pct(row.get("SR_given_surface_clean")),
            f"{row.get('avg_model_calls'):.1f}" if row.get("avg_model_calls") is not None else "-",
            f"{row.get('avg_tokens'):.0f}" if row.get("avg_tokens") is not None else "-",
            f"{row.get('SR_per_100_calls'):.2f}" if row.get("SR_per_100_calls") is not None else "-",
            f"{row.get('SR_per_100k_tokens'):.2f}" if row.get("SR_per_100k_tokens") is not None else "-",
            str(row.get("surface_failure_count", "")),
            str(row.get("structural_failure_count", "")),
            top_fail,
        ]) + " |")

    output = args.output
    if output is None:
        base = Path(paths[0]).with_suffix("")
        output = str(base) + "_with_ci.md"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
