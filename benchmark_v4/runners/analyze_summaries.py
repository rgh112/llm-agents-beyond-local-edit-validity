#!/usr/bin/env python3
"""Analyze pre-aggregated summary JSONs (no raw access needed).

Reads any *_summary.json in the v4 format and emits:
  - SR + 95% bootstrap CI per cell (using n_success / n_total)
  - Surface-vs-structural total event counts
  - Top 3 terminal failures
  - Markdown table

This is the fallback when raw episode logs are not available
or are too slow to read at scale.

Example:
    python -m benchmark_v4.runners.analyze_summaries \
        --summaries results_v4/main_prompt_20260311_223450/main_prompt_summary.json \
                    results_v4/gb1_main_rerun_20260312_132622/gb1_main_rerun_summary.json \
        --out-dir results_v4/rebuttal --label qwen_summary_baseline
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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


def bootstrap_ci(successes, n, n_boot=2000, alpha=0.05, seed=42):
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    boots = []
    p_hat = successes / n
    for _ in range(n_boot):
        s = sum(1 for _ in range(n) if rng.random() < p_hat)
        # Use empirical resample of successes/failures distribution
        # (above is binomial approx; use exact resample for tiny n)
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def bootstrap_ci_exact(successes, n, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap from the empirical episode-level success indicator."""
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


def render_md(rows, label):
    lines = [f"# Rebuttal evidence (summary-based): {label}\n"]
    lines.append(
        "Aggregated from `*_summary.json`. **SR**: success rate; "
        "**CI**: 2.5–97.5% bootstrap percentile (2,000 episode-level resamples). "
        "**Surface events**: `MALFORMED_ACTION / INVALID_POSITION / INVALID_VALUE / "
        "INVALID_WORD / ILLEGAL_EDIT / REPEATED_EXACT_EDIT`. "
        "**Structural events**: `BUDGET_UNAWARE_ACTION / LOCAL_OPTIMUM_TRAP / "
        "OBJECTIVE_TRADEOFF_FAILURE / PREMATURE_FINALIZE / HARD_CONSTRAINT_VIOLATION / "
        "RECOVERY_COST_EXPLOSION / GLOBAL_FEASIBILITY_LOSS / OSCILLATION`.\n"
    )
    by_env = defaultdict(list)
    for r in rows:
        by_env[r["env"]].append(r)

    for env in sorted(by_env):
        lines.append(f"\n## {env}\n")
        lines.append(
            "| Model | Prompt | Mem | N | SR | 95% CI | "
            "Surface events | Structural events | Top terminal failures |"
        )
        lines.append("|---|---|---|---:|---:|---|---:|---:|---|")
        rs = sorted(by_env[env], key=lambda x: (x["model"], x["prompt"], x["memory"]))
        for r in rs:
            lo, hi = r["SR_ci95"]
            ci_str = f"[{lo:.0%}, {hi:.0%}]"
            top = ", ".join(
                f"{k}:{v}" for k, v in
                sorted(r["terminal_failures"].items(), key=lambda x: -x[1])[:3]
            ) or "—"
            lines.append(
                f"| {r['model']} | {r['prompt']} | {r['memory']} | "
                f"{r['n']} | {r['SR']:.0%} | {ci_str} | "
                f"{r['surface_failure_count']} | {r['structural_failure_count']} | {top} |"
            )

    lines.append("\n### Reading guide\n")
    lines.append(
        "- Use the **CI** column to test prompt/model effects. If CIs overlap, the "
        "data does not establish a separation in that cell.\n"
        "- High **Structural events** with low **Surface events** = "
        "the model produces grammatically valid actions but fails the global objective. "
        "This is the diagnostic separation the paper claims.\n"
        "- For the local-validity-conditioned analysis (success rate among surface-clean "
        "episodes), the raw per-episode logs are required — see `analyze_rebuttal.py`.\n"
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="Paths to *_summary.json files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="combined")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--restrict-prompts", nargs="*",
                    help="If set, only include these prompt conditions.")
    ap.add_argument("--restrict-envs", nargs="*",
                    help="If set, only include these env names.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()  # de-dup (env, model, prompt, memory)
    for spath in args.summaries:
        d = json.load(open(spath))
        cfg = d.get("config", {})
        default_model = (cfg.get("model")
                         or (cfg.get("models", [None])[0] if cfg.get("models") else None)
                         or "unknown_model")
        for r in d.get("results", []):
            env = r.get("env") or r.get("env_name")
            model = r.get("model") or default_model
            prompt = r.get("prompt") or r.get("prompt_condition")
            memory = r.get("memory") or r.get("memory_condition")
            if args.restrict_envs and env not in args.restrict_envs:
                continue
            if args.restrict_prompts and prompt not in args.restrict_prompts:
                continue
            key = (env, model, prompt, memory)
            if key in seen:
                continue  # later file wins?  no — first wins, skip dup
            seen.add(key)
            n = int(r.get("n_total", 0))
            n_succ = int(r.get("n_success", 0))
            sr = r.get("SR", n_succ / n if n else 0)
            lo, hi = bootstrap_ci_exact(n_succ, n, n_boot=args.bootstrap_iters)
            events = r.get("all_events", {})
            surf = sum(events.get(e, 0) for e in SURFACE_EVENTS)
            struct = sum(events.get(e, 0) for e in STRUCTURAL_EVENTS)
            rows.append({
                "env": env,
                "model": model,
                "prompt": prompt,
                "memory": memory,
                "n": n,
                "n_success": n_succ,
                "SR": sr,
                "SR_ci95": [lo, hi],
                "surface_failure_count": surf,
                "structural_failure_count": struct,
                "terminal_failures": dict(r.get("terminal_failures", {})),
                "source_summary": spath,
            })

    json_path = Path(args.out_dir) / f"summary_table_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump({"label": args.label, "rows": rows}, f, indent=2, default=str)

    md_path = Path(args.out_dir) / f"summary_table_{args.label}.md"
    md = render_md(rows, args.label)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
