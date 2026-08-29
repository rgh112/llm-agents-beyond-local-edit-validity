#!/usr/bin/env python3
"""Aggregate the eleven award-closure experiments into LaTeX-ready tables.

Reads `results_v4/` summaries for:
  - scorer ablation (text_visible / proxy / oracle)
  - p1 api stability (gb1 / alloy)
  - p1 sensitivity (gb1 / alloy)
  - p1 memory extended (word_ladder / alloy / gb1)
  - p2 decoding robustness (consolidated)

Prints a single .tex fragment containing all new tables, plus a short
machine-readable JSON dump used by verification scripts.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v4"


def _load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _model_short(m: str) -> str:
    short = m.split("/")[-1]
    return {
        "qwen3-8b": "Qwen3-8B",
        "ministral-8b-2512": "Ministral-8B",
        "llama-3.1-8b-instruct": "Llama-8B",
    }.get(short, short)


def _env_short(e: str) -> str:
    return {"word_ladder": "Word Ladder", "alloy": "Alloy", "gb1_sequence": "GB1"}.get(e, e)


def _prompt_short(p: str) -> str:
    return {"zero_shot": "Zero-shot", "self_check": "Self-check", "scaffold": "Scaffold"}.get(p, p)


def _mem_short(m: str) -> str:
    return {
        "state_only": "State only",
        "window_3": "Window-3",
        "full_history": "Full history",
        "summary": "Summary",
        "best_state": "Best-state",
        "randomized_history": "Randomized",
        "misleading_history": "Misleading",
    }.get(m, m)


def load_scorer_ablation() -> dict:
    out = {}
    for scorer, dirname in [
        ("text_visible", "planning_wrappers_scorer_ablation_text_visible_20260515_213651"),
        ("proxy", "planning_wrappers_scorer_ablation_proxy_20260515_034537"),
        ("oracle", "planning_wrappers_scorer_ablation_oracle_20260515_034542"),
    ]:
        d = _load(RESULTS / dirname / "planning_wrapper_summary.json")
        rows = []
        for r in d["results"]:
            rows.append({
                "model": _model_short(r["model"]),
                "env": _env_short(r["env"]),
                "SR": r["SR"],
                "n_success": r["n_success"],
                "n_total": r["n_total"],
                "local_valid": r.get("local_valid_edit_rate", 0.0),
                "calls_per_ep": r.get("avg_model_calls", 0.0),
                "tokens_per_ep": r.get("avg_tokens", 0.0),
                "structural": r.get("structural_failure_count", 0),
                "surface": r.get("surface_failure_count", 0),
                "terminal_failures": dict(r.get("terminal_failures", {})),
                "all_events": dict(r.get("all_events", {})),
            })
        out[scorer] = rows
    return out


def load_api_stability() -> dict:
    out = {}
    for env, dirname in [
        ("Alloy", "cross_model_p1_api_stability_alloy_20260515_20260515_034537"),
        ("GB1", "cross_model_p1_api_stability_gb1_20260515_20260515_034533"),
    ]:
        d = _load(RESULTS / dirname / "cross_model_summary.json")
        rows = []
        for r in d["results"]:
            rows.append({
                "model": _model_short(r["model"]),
                "prompt": _prompt_short(r["prompt"]),
                "SR": r["SR"],
                "n_success": r["n_success"],
                "n_total": r["n_total"],
                "local_valid": r.get("local_valid_edit_rate", 0.0),
                "structural": r.get("structural_failure_count", 0),
            })
        out[env] = rows
    return out


def load_sensitivity() -> dict:
    out = {}
    for env, dirname in [
        ("Alloy", "sensitivity_p1_sensitivity_alloy_20260515_034537"),
        ("GB1", "sensitivity_p1_sensitivity_gb1_20260515_034532"),
    ]:
        d = _load(RESULTS / dirname / "sensitivity_summary.json")
        rows = []
        for r in d["results"]:
            rows.append({
                "model": _model_short(r["model"]),
                "setting": r.get("sensitivity_setting", "?"),
                "prompt": _prompt_short(r["prompt"]),
                "SR": r["SR"],
                "n_success": r["n_success"],
                "n_total": r["n_total"],
                "local_valid": r.get("local_valid_edit_rate", 0.0),
                "structural": r.get("structural_failure_count", 0),
            })
        out[env] = rows
    return out


def load_memory() -> dict:
    out = {}
    for env, dirname in [
        ("Word Ladder", "memory_extended_p1_memory_word_ladder_20260517_064424"),
        ("Alloy", "memory_extended_p1_memory_alloy_20260517_023415"),
        ("GB1", "memory_extended_p1_memory_gb1_20260517_010326"),
    ]:
        d = _load(RESULTS / dirname / "memory_extended_summary.json")
        rows = []
        for r in d["results"]:
            rows.append({
                "model": _model_short(r["model"]),
                "memory": _mem_short(r["memory"]),
                "SR": r["SR"],
                "n_success": r["n_success"],
                "n_total": r["n_total"],
                "local_valid": r.get("local_valid_edit_rate", 0.0),
                "structural": r.get("structural_failure_count", 0),
            })
        out[env] = rows
    return out


def load_decoding() -> dict:
    d = _load(RESULTS / "cross_model_decoding_robustness_20260513_160325" / "cross_model_summary.json")
    rows = []
    for r in d["results"]:
        rows.append({
            "model": _model_short(r["model"]),
            "env": _env_short(r["env"]),
            "prompt": _prompt_short(r["prompt"]),
            "SR": r["SR"],
            "n_success": r["n_success"],
            "n_total": r["n_total"],
            "local_valid": r.get("local_valid_edit_rate", 0.0),
            "structural": r.get("structural_failure_count", 0),
        })
    return rows


# ---- LaTeX table writers ----------------------------------------------------

MODELS = ["Qwen3-8B", "Ministral-8B", "Llama-8B"]
ENVS = ["Word Ladder", "Alloy", "GB1"]
MEMS = ["State only", "Window-3", "Full history", "Summary", "Best-state", "Randomized", "Misleading"]


def tex_scorer_ablation(data) -> str:
    """3 scorers × 3 models × 3 envs; show SR and calls/ep."""
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Open-family scorer ablation for shallow planning wrappers (beam, "
                 r"$k=3$, depth 1). Each cell is success rate over 20 seeds; \texttt{text\_visible} "
                 r"and \texttt{proxy} use only information already exposed to the policy, while "
                 r"\texttt{oracle} uses the ground-truth target distance and is reported as an "
                 r"upper bound only. Calls/episode in the rightmost column are for the "
                 r"\texttt{text\_visible} scorer.}")
    lines.append(r"\label{tab:app_open_family_scorer}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Env. & Text-vis. & Proxy & Oracle & Calls/ep \\")
    lines.append(r"\midrule")
    # group by model, then env
    by_key = {(r["model"], r["env"]): r for r in data["text_visible"]}
    by_key_proxy = {(r["model"], r["env"]): r for r in data["proxy"]}
    by_key_oracle = {(r["model"], r["env"]): r for r in data["oracle"]}
    for m in MODELS:
        for e in ENVS:
            tv = by_key.get((m, e))
            px = by_key_proxy.get((m, e))
            orc = by_key_oracle.get((m, e))
            if not tv:
                continue
            lines.append(
                f"{m} & {e} & "
                f"{int(round(tv['SR']*100))}\\% & "
                f"{int(round(px['SR']*100)) if px else '--'}\\% & "
                f"{int(round(orc['SR']*100)) if orc else '--'}\\% & "
                f"{tv['calls_per_ep']:.1f} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tex_memory_grid(data) -> str:
    """7 memories × 3 envs × 3 models, 50 seeds each."""
    lines = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{Open-family extended-memory grid. Each cell is success rate over 50 "
                 r"seeds, structural prompt, deterministic decoding. \texttt{Randomized} shuffles "
                 r"the trajectory history before each call; \texttt{Misleading} injects a "
                 r"length-matched but reward-mismatched history. The state-only column matches the "
                 r"main paper's deterministic baseline.}")
    lines.append(r"\label{tab:app_open_family_memory}")
    lines.append(r"\begin{tabular}{ll" + "r" * len(MEMS) + r"}")
    lines.append(r"\toprule")
    header = " & ".join(["Env.", "Model"] + MEMS)
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for env in ENVS:
        env_rows = {(r["model"], r["memory"]): r for r in data[env]}
        for i, m in enumerate(MODELS):
            row_cells = [env if i == 0 else "", m]
            for mem in MEMS:
                r = env_rows.get((m, mem))
                row_cells.append(f"{int(round(r['SR']*100))}\\%" if r else "--")
            lines.append(" & ".join(row_cells) + r" \\")
        if env != ENVS[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def tex_sensitivity(data) -> str:
    """3 models × {2 settings} × 2 prompts × 50 seeds; one table per env."""
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Open-family surrogate sensitivity on Alloy (heavy-scaling toggle) and "
                 r"GB1 (additive evaluator). Each cell is success rate over 50 seeds; \texttt{ZS} "
                 r"is zero-shot, \texttt{SC} is self-check.}")
    lines.append(r"\label{tab:app_open_family_sensitivity}")
    lines.append(r"\begin{tabular}{lllrr}")
    lines.append(r"\toprule")
    lines.append(r"Env. & Setting & Model & ZS & SC \\")
    lines.append(r"\midrule")
    for env in ["Alloy", "GB1"]:
        rows = data[env]
        # discover settings present
        settings = []
        for r in rows:
            if r["setting"] not in settings:
                settings.append(r["setting"])
        for si, s in enumerate(settings):
            for mi, m in enumerate(MODELS):
                zs = next((r for r in rows if r["model"] == m and r["setting"] == s and r["prompt"] == "Zero-shot"), None)
                sc = next((r for r in rows if r["model"] == m and r["setting"] == s and r["prompt"] == "Self-check"), None)
                row_cells = [env if (si == 0 and mi == 0) else "",
                             s.replace("_", " ") if mi == 0 else "",
                             m,
                             f"{int(round(zs['SR']*100))}\\%" if zs else "--",
                             f"{int(round(sc['SR']*100))}\\%" if sc else "--"]
                lines.append(" & ".join(row_cells) + r" \\")
            if not (env == "GB1" and si == len(settings) - 1):
                lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tex_api_stability(data) -> str:
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Open-family hosted-API stability rerun on Alloy and GB1 (20 seeds, "
                 r"deterministic decoding, 2026-05-15). Each cell is success rate; \texttt{ZS} is "
                 r"zero-shot, \texttt{SC} is self-check.}")
    lines.append(r"\label{tab:app_open_family_api_stability}")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(r"Env. & Model & ZS & SC \\")
    lines.append(r"\midrule")
    for env in ["Alloy", "GB1"]:
        rows = data[env]
        for mi, m in enumerate(MODELS):
            zs = next((r for r in rows if r["model"] == m and r["prompt"] == "Zero-shot"), None)
            sc = next((r for r in rows if r["model"] == m and r["prompt"] == "Self-check"), None)
            row_cells = [env if mi == 0 else "",
                         m,
                         f"{int(round(zs['SR']*100))}\\%" if zs else "--",
                         f"{int(round(sc['SR']*100))}\\%" if sc else "--"]
            lines.append(" & ".join(row_cells) + r" \\")
        if env != "GB1":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tex_decoding(rows) -> str:
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Open-family decoding-robustness panel. Sampled decoding uses "
                 r"temperature 0.7 and top-$p$ 0.9; 20 seeds per cell. Compare against the "
                 r"deterministic main-results panel for the same model and prompt.}")
    lines.append(r"\label{tab:app_open_family_decoding}")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(r"Env. & Model & Zero-shot & Self/Scaffold \\")
    lines.append(r"\midrule")
    by_key = {(r["model"], r["env"], r["prompt"]): r for r in rows}
    for ei, env in enumerate(ENVS):
        for mi, m in enumerate(MODELS):
            zs = by_key.get((m, env, "Zero-shot"))
            if env == "Word Ladder":
                sc = by_key.get((m, env, "Scaffold"))
            else:
                sc = by_key.get((m, env, "Self-check"))
            row_cells = [env if mi == 0 else "",
                         m,
                         f"{int(round(zs['SR']*100))}\\%" if zs else "--",
                         f"{int(round(sc['SR']*100))}\\%" if sc else "--"]
            lines.append(" & ".join(row_cells) + r" \\")
        if ei != len(ENVS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    scorer = load_scorer_ablation()
    api = load_api_stability()
    sens = load_sensitivity()
    mem = load_memory()
    decode = load_decoding()

    # Write per-table LaTeX fragments directly into the paper directory so
    # `\input{open_family_*.tex}` resolves without path acrobatics.
    paper_dir = ROOT / "EMNLP_Word_Ladder_MARCH" / "latex"
    header = "% Auto-generated by supplementary/code/aggregate_award_closure.py\n% Do not hand-edit.\n"
    for name, body in [
        ("open_family_scorer.tex", tex_scorer_ablation(scorer)),
        ("open_family_memory.tex", tex_memory_grid(mem)),
        ("open_family_sensitivity.tex", tex_sensitivity(sens)),
        ("open_family_api_stability.tex", tex_api_stability(api)),
        ("open_family_decoding.tex", tex_decoding(decode)),
    ]:
        (paper_dir / name).write_text(header + body + "\n")
        print(f"Wrote {paper_dir / name}")

    # Write machine-readable dump for verifier
    out_json = ROOT / "supplementary" / "award_closure_summary.json"
    out_json.write_text(json.dumps({
        "scorer_ablation": scorer,
        "api_stability": api,
        "sensitivity": sens,
        "memory": mem,
        "decoding": decode,
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
