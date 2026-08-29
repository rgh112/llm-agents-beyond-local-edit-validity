#!/usr/bin/env python3
"""Rebuttal analysis: bootstrap CIs + local-validity-conditioned success.

Reads raw/*.json episode logs from any results directory and produces:
  Table R2: per (env, model, prompt, memory) cell
    - SR + 95% bootstrap CI
    - local valid edit rate
    - surface-clean episode share
    - SR | surface-clean
    - top residual failures (structural vs surface)

Output: <out_dir>/rebuttal_table.json + <out_dir>/rebuttal_table.md.

Example:
    python -m benchmark_v4.runners.analyze_rebuttal \
        --raw-dirs results_v4/main_prompt_20260311_223450/raw \
                   results_v4/gb1_main_rerun_20260312_132622/raw \
        --out-dir results_v4/rebuttal --label qwen_main
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
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
    data = [1] * successes + [0] * (n - successes)
    boots = []
    for _ in range(n_boot):
        s = sum(rng.choice(data) for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def load_episode(path):
    with open(path) as f:
        return json.load(f)


def cell_key(ep):
    return (ep["env_name"], ep["model"], ep["prompt_condition"], ep["memory_condition"])


def summarize_episode(ep):
    """Return per-episode flags used for aggregation."""
    success = bool(ep["success"])
    terminal_failure = ep.get("terminal_failure")
    edit_attempts = 0
    valid_edits = 0
    has_surface_event = False
    structural_events_in_ep = Counter()
    surface_events_in_ep = Counter()

    for step in ep["steps"]:
        pa = step.get("parsed_action") or {}
        if pa.get("type") == "edit":
            edit_attempts += 1
            if step.get("valid"):
                valid_edits += 1
        for ev in step.get("events", []):
            ev_name = ev if isinstance(ev, str) else ev.get("name", str(ev))
            if ev_name in SURFACE_EVENTS:
                has_surface_event = True
                surface_events_in_ep[ev_name] += 1
            elif ev_name in STRUCTURAL_EVENTS:
                structural_events_in_ep[ev_name] += 1

    return {
        "success": success,
        "terminal_failure": terminal_failure,
        "edit_attempts": edit_attempts,
        "valid_edits": valid_edits,
        "has_surface_event": has_surface_event,
        "surface_events": surface_events_in_ep,
        "structural_events": structural_events_in_ep,
    }


def aggregate(cells, n_boot=2000):
    rows = []
    for key, eps in sorted(cells.items()):
        env, model, prompt, mem = key
        n = len(eps)
        successes = sum(e["success"] for e in eps)
        sr = successes / n if n else 0
        lo, hi = bootstrap_ci(successes, n, n_boot=n_boot)

        edit_attempts = sum(e["edit_attempts"] for e in eps)
        valid_edits = sum(e["valid_edits"] for e in eps)
        local_valid_rate = valid_edits / edit_attempts if edit_attempts else None

        clean_eps = [e for e in eps if not e["has_surface_event"]]
        n_clean = len(clean_eps)
        n_clean_succ = sum(e["success"] for e in clean_eps)
        sr_clean = n_clean_succ / n_clean if n_clean else None
        clean_lo, clean_hi = (
            bootstrap_ci(n_clean_succ, n_clean, n_boot=n_boot)
            if n_clean else (None, None)
        )

        structural_terminal = Counter()
        surface_terminal = Counter()
        other_terminal = Counter()
        for e in eps:
            if e["success"]:
                continue
            tf = e["terminal_failure"]
            if tf is None:
                other_terminal["UNKNOWN"] += 1
            elif tf in STRUCTURAL_EVENTS:
                structural_terminal[tf] += 1
            elif tf in SURFACE_EVENTS:
                surface_terminal[tf] += 1
            else:
                other_terminal[tf] += 1

        all_struct = Counter()
        all_surf = Counter()
        for e in eps:
            all_struct.update(e["structural_events"])
            all_surf.update(e["surface_events"])

        rows.append({
            "env": env,
            "model": model,
            "prompt": prompt,
            "memory": mem,
            "n": n,
            "n_success": successes,
            "SR": sr,
            "SR_ci95": [lo, hi],
            "local_valid_edit_rate": local_valid_rate,
            "n_edit_attempts": edit_attempts,
            "n_local_valid_edits": valid_edits,
            "n_surface_clean_episodes": n_clean,
            "n_surface_clean_successes": n_clean_succ,
            "SR_given_surface_clean": sr_clean,
            "SR_given_surface_clean_ci95":
                [clean_lo, clean_hi] if sr_clean is not None else None,
            "terminal_structural": dict(structural_terminal.most_common()),
            "terminal_surface": dict(surface_terminal.most_common()),
            "terminal_other": dict(other_terminal.most_common()),
            "all_structural_events": dict(all_struct.most_common()),
            "all_surface_events": dict(all_surf.most_common()),
            "top_residual_structural":
                structural_terminal.most_common(1)[0][0]
                if structural_terminal else None,
        })
    return rows


def fmt_pct(x):
    return f"{x:.0%}" if x is not None else "—"


def render_md(rows, label):
    lines = [f"# Rebuttal evidence: {label}\n"]
    lines.append(
        "Each cell aggregates episode-level outcomes from the raw logs. "
        "**SR**: success rate; **CI**: 2.5–97.5% bootstrap percentile (2,000 resamples). "
        "**LocalValid**: fraction of edit attempts that the env accepted as a valid edit. "
        "**SurfClean**: episodes with no surface failure event "
        "(invalid action / invalid position / invalid value / invalid word / illegal edit / repeated edit). "
        "**SR|SurfClean**: success rate restricted to surface-clean episodes — "
        "tests whether failure is structural rather than schema-following.\n"
    )
    by_env = defaultdict(list)
    for r in rows:
        by_env[r["env"]].append(r)

    for env in sorted(by_env):
        lines.append(f"\n## {env}\n")
        lines.append(
            "| Model | Prompt | Mem | N | SR | 95% CI | LocalValid | "
            "SurfClean | SR\\|SurfClean | Top residual structural failure |"
        )
        lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---|")
        rs = sorted(by_env[env], key=lambda x: (x["model"], x["prompt"], x["memory"]))
        for r in rs:
            lo, hi = r["SR_ci95"]
            ci_str = f"[{lo:.0%}, {hi:.0%}]"
            clean_share = (r["n_surface_clean_episodes"] / r["n"]) if r["n"] else 0
            sr_clean_str = "—"
            if r["SR_given_surface_clean"] is not None:
                cl, ch = r["SR_given_surface_clean_ci95"]
                sr_clean_str = (
                    f"{r['SR_given_surface_clean']:.0%} "
                    f"[{cl:.0%}, {ch:.0%}] "
                    f"({r['n_surface_clean_successes']}/{r['n_surface_clean_episodes']})"
                )
            top = r["top_residual_structural"] or "—"
            top_n = r["terminal_structural"].get(top, 0) if top != "—" else 0
            top_str = f"{top} ({top_n})" if top != "—" else "—"
            lines.append(
                f"| {r['model']} | {r['prompt']} | {r['memory']} | "
                f"{r['n']} | {r['SR']:.0%} | {ci_str} | "
                f"{fmt_pct(r['local_valid_edit_rate'])} | "
                f"{clean_share:.0%} ({r['n_surface_clean_episodes']}/{r['n']}) | "
                f"{sr_clean_str} | {top_str} |"
            )

    lines.append("\n### Reading guide\n")
    lines.append(
        "- If `SR|SurfClean` ≪ 100% with `LocalValid` near 100%, "
        "the model is producing well-formed, locally valid actions yet still failing the "
        "delayed global objective — this is the structural-failure signature claimed by the paper.\n"
        "- Compare across `Model` rows for the same (env, prompt) to gauge whether the qualitative "
        "pattern recurs across model families (cross-model sanity check).\n"
        "- Compare `zero_shot` vs `few_shot_format` SR to test the format-only-does-not-help claim. "
        "If the CIs overlap, the data does not support a separation in that cell.\n"
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dirs", nargs="+", required=True,
                    help="One or more directories containing per-episode raw json files.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for rebuttal_table.json/md.")
    ap.add_argument("--label", default="combined",
                    help="Label used in the markdown title and filename suffix.")
    ap.add_argument("--restrict-prompts", nargs="*",
                    help="If set, only include these prompt conditions.")
    ap.add_argument("--restrict-envs", nargs="*",
                    help="If set, only include these env names.")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--progress-every", type=int, default=0,
                    help="If >0, print progress every N raw files while loading.")
    ap.add_argument("--verbose-files", action="store_true",
                    help="Print each raw filename before loading it.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cells = defaultdict(list)
    n_files = 0
    for raw_dir in args.raw_dirs:
        for path in sorted(Path(raw_dir).glob("*.json")):
            if args.verbose_files:
                print(f"Loading {n_files + 1}: {path.name}", flush=True)
            ep = load_episode(path)
            if args.restrict_envs and ep["env_name"] not in args.restrict_envs:
                continue
            if args.restrict_prompts and ep["prompt_condition"] not in args.restrict_prompts:
                continue
            cells[cell_key(ep)].append(summarize_episode(ep))
            n_files += 1
            if args.progress_every and n_files % args.progress_every == 0:
                print(f"Loaded {n_files} raw episodes...", flush=True)
    print(f"Loaded {n_files} episodes from {len(args.raw_dirs)} dir(s); "
          f"{len(cells)} cells.")

    rows = aggregate(cells, n_boot=args.bootstrap_iters)

    json_path = Path(args.out_dir) / f"rebuttal_table_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump({
            "label": args.label,
            "n_episodes": n_files,
            "n_cells": len(cells),
            "rows": rows,
        }, f, indent=2, default=str)

    md_path = Path(args.out_dir) / f"rebuttal_table_{args.label}.md"
    md = render_md(rows, args.label)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
