#!/usr/bin/env python3
"""Summarize prospective named-intervention runs.

The analyzer accepts one or more planning-wrapper summary files or directories
and produces weighted environment/controller aggregates for the manuscript
case-study table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def iter_summary_files(paths: Iterable[str]) -> Iterable[Path]:
    for item in paths:
        p = Path(item)
        if p.is_file() and p.name == "planning_wrapper_summary.json":
            yield p
        elif p.is_dir() and (p / "planning_wrapper_summary.json").exists():
            yield p / "planning_wrapper_summary.json"
        elif p.is_dir():
            yield from sorted(p.glob("**/planning_wrapper_summary.json"))


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []
    for path in iter_summary_files(paths):
        with open(path) as f:
            data = json.load(f)
        for row in data.get("results") or []:
            out = dict(row)
            out["source_summary"] = str(path)
            rows.append(out)
    return rows


def row_n(row: Dict[str, Any]) -> int:
    return int(row.get("n_total") or row.get("n") or 0)


def weighted_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = sum(row_n(r) for r in rows)
    success = sum(int(r.get("n_success") or r.get("successes") or 0) for r in rows)
    surface = sum(int(r.get("n_surface_clean_episodes") or r.get("surface_clean_episodes") or 0) for r in rows)
    surface_success = sum(
        int(r.get("n_surface_clean_successes") or r.get("surface_clean_successes") or 0)
        for r in rows
    )
    edits = sum(int(r.get("n_edit_attempts") or 0) for r in rows)
    valid_edits = sum(int(r.get("n_local_valid_edits") or 0) for r in rows)
    weighted_calls = sum((r.get("avg_model_calls") or 0.0) * row_n(r) for r in rows)
    weighted_tokens = sum((r.get("avg_tokens") or 0.0) * row_n(r) for r in rows)
    return {
        "n": n,
        "successes": success,
        "SR": success / n if n else None,
        "surface_clean_rate": surface / n if n else None,
        "surface_clean_episodes": surface,
        "surface_clean_successes": surface_success,
        "SR_given_surface_clean": surface_success / surface if surface else None,
        "local_valid_edit_rate": valid_edits / edits if edits else None,
        "avg_model_calls": weighted_calls / n if n else None,
        "avg_tokens": weighted_tokens / n if n else None,
        "models": sorted({r.get("model") for r in rows if r.get("model")}),
        "source_summaries": sorted({r.get("source_summary") for r in rows if r.get("source_summary")}),
    }


def metric_delta(row: Dict[str, Any], baseline: Dict[str, Any], key: str) -> Optional[float]:
    a = row["aggregate"].get(key)
    b = baseline["aggregate"].get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def metric_ratio(row: Dict[str, Any], baseline: Dict[str, Any], key: str) -> Optional[float]:
    a = row["aggregate"].get(key)
    b = baseline["aggregate"].get(key)
    if a is None or b in (None, 0):
        return None
    return float(a) / float(b)


def build_delta_rows(aggregates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["env"], row["controller"]): row for row in aggregates}
    deltas = []
    for row in aggregates:
        env = row["env"]
        controller = row["controller"]
        if controller == "greedy":
            continue
        baseline = by_key.get((env, "greedy"))
        if baseline is None:
            continue
        deltas.append(
            {
                "env": env,
                "controller": controller,
                "baseline": "greedy",
                "n": row["aggregate"]["n"],
                "baseline_n": baseline["aggregate"]["n"],
                "delta_SR": metric_delta(row, baseline, "SR"),
                "delta_surface_clean_rate": metric_delta(row, baseline, "surface_clean_rate"),
                "delta_surface_clean_SR": metric_delta(row, baseline, "SR_given_surface_clean"),
                "delta_local_validity": metric_delta(row, baseline, "local_valid_edit_rate"),
                "call_ratio": metric_ratio(row, baseline, "avg_model_calls"),
                "token_ratio": metric_ratio(row, baseline, "avg_tokens"),
            }
        )
    return deltas


def _delta_lookup(deltas: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    return {(row["env"], row["controller"]): row for row in deltas}


def _available(aggregates: List[Dict[str, Any]], env: str, controllers: List[str]) -> List[str]:
    have = {(row["env"], row["controller"]) for row in aggregates}
    return [controller for controller in controllers if (env, controller) in have]


def _max_delta(
    deltas: List[Dict[str, Any]], env: str, controllers: List[str], metric: str
) -> Optional[Dict[str, Any]]:
    candidates = [
        row for row in deltas
        if row["env"] == env and row["controller"] in controllers and row.get(metric) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric]))


def _best_controller(aggregates: List[Dict[str, Any]], env: str) -> Optional[str]:
    candidates = [
        row for row in aggregates
        if row["env"] == env and row["controller"] != "greedy" and row["aggregate"].get("SR") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["aggregate"]["SR"]))["controller"]


def assess_preregistered_hypotheses(
    aggregates: List[Dict[str, Any]],
    deltas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    named_required = {"reflexion_retry", "self_consistency", "tot_style_beam"}
    named_panel_complete = all(
        set(_available(aggregates, env, sorted(named_required))) == named_required
        for env in ["word_ladder", "alloy", "gb1_sequence"]
    )

    def scope_status(base: str) -> str:
        if base.startswith("pending"):
            return base
        if named_panel_complete:
            return base
        return f"preliminary_{base}_existing_wrappers"

    checks: List[Dict[str, Any]] = []

    word_named = _available(aggregates, "word_ladder", sorted(named_required))
    word_any = _available(aggregates, "word_ladder", ["reflexion_retry", "self_consistency"])
    word_lv = _max_delta(deltas, "word_ladder", ["reflexion_retry", "self_consistency"], "delta_local_validity")
    word_surface = _max_delta(
        deltas,
        "word_ladder",
        ["reflexion_retry", "self_consistency"],
        "delta_surface_clean_rate",
    )
    if not word_any:
        h1_status = "pending_named_results"
        h1_summary = "No Reflexion-style or self-consistency Word Ladder result is available yet."
    elif (
        (word_lv and float(word_lv["delta_local_validity"]) > 0.03)
        or (word_surface and float(word_surface["delta_surface_clean_rate"]) > 0.03)
    ):
        h1_status = "supported"
        h1_summary = "Retry/sampling improves the interface-side Word Ladder measures versus greedy."
    else:
        h1_status = "weakened"
        h1_summary = "Available retry/sampling wrappers do not improve Word Ladder interface-side measures."
    checks.append(
        {
            "id": "H1",
            "status": scope_status(h1_status),
            "named_panel_complete": named_panel_complete,
            "available_named_controllers": word_named,
            "summary": h1_summary,
            "best_local_validity_delta": word_lv,
            "best_surface_clean_delta": word_surface,
        }
    )

    search_controllers = ["self_consistency", "tot_style_beam", "beam"]
    alloy_search = _max_delta(deltas, "alloy", search_controllers, "delta_SR")
    gb1_search = _max_delta(deltas, "gb1_sequence", search_controllers, "delta_SR")
    if alloy_search is None or gb1_search is None:
        h2_status = "pending_named_results"
        h2_summary = "Comparable Alloy/GB1 search-style wrapper results are not both available."
    else:
        margin = float(alloy_search["delta_SR"]) - float(gb1_search["delta_SR"])
        if margin > 0.05:
            h2_status = "supported"
            h2_summary = "Alloy gains more from search-style wrappers than GB1 under the available results."
        elif margin < -0.05:
            h2_status = "weakened"
            h2_summary = "GB1 gains more than Alloy from the available search-style wrappers."
        else:
            h2_status = "mixed"
            h2_summary = "Alloy and GB1 search-style gains are close under the available results."
    checks.append(
        {
            "id": "H2",
            "status": scope_status(h2_status),
            "named_panel_complete": named_panel_complete,
            "summary": h2_summary,
            "alloy_best_search_delta": alloy_search,
            "gb1_best_search_delta": gb1_search,
        }
    )

    gb1_visible = _max_delta(
        deltas,
        "gb1_sequence",
        ["reflexion_retry", "self_consistency", "tot_style_beam", "beam"],
        "delta_SR",
    )
    if gb1_visible is None:
        h3_status = "pending_named_results"
        h3_summary = "No GB1 visible-wrapper delta is available."
    elif float(gb1_visible["delta_SR"]) <= 0.05:
        h3_status = "supported"
        h3_summary = "GB1 visible-only wrapper gains remain small versus greedy."
    elif float(gb1_visible["delta_SR"]) >= 0.15:
        h3_status = "weakened"
        h3_summary = "A visible-only wrapper substantially improves GB1, weakening the limited-wrapper prediction."
    else:
        h3_status = "mixed"
        h3_summary = "GB1 visible-only wrapper gains are moderate."
    checks.append(
        {
            "id": "H3",
            "status": scope_status(h3_status),
            "named_panel_complete": named_panel_complete,
            "summary": h3_summary,
            "gb1_best_visible_delta": gb1_visible,
        }
    )

    best_by_env = {
        env: _best_controller(aggregates, env)
        for env in ["word_ladder", "alloy", "gb1_sequence"]
    }
    present_best = {env: controller for env, controller in best_by_env.items() if controller}
    if len(present_best) < 3:
        h4_status = "pending_named_results"
        h4_summary = "Best-wrapper comparison is incomplete across environments."
    elif len(set(present_best.values())) > 1:
        h4_status = "supported"
        h4_summary = "The best non-greedy wrapper differs across environments."
    else:
        h4_status = "weakened"
        h4_summary = "One wrapper is best in all environments under the available results."
    checks.append(
        {
            "id": "H4",
            "status": scope_status(h4_status),
            "named_panel_complete": named_panel_complete,
            "summary": h4_summary,
            "best_non_greedy_controller_by_env": best_by_env,
        }
    )
    return checks


def fmt_pct(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}%"


def fmt_num(value: Any) -> str:
    return "--" if value is None else f"{float(value):.1f}"


def write_markdown(
    path: Path,
    aggregates: List[Dict[str, Any]],
    deltas: List[Dict[str, Any]],
    hypothesis_checks: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Intervention-Prediction Summary",
        "",
        "This table is descriptive. Hypothesis labels are generated from the fixed",
        "rules in `INTERVENTION_PREDICTION_PREREGISTRATION.md` and should not be",
        "manually relabeled after hosted results are known.",
        "",
        "## Aggregates",
        "",
        "| Environment | Controller | N | SR | Surface-clean SR | Local validity | Calls/ep | Tokens/ep |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {env} | {controller} | {n} | {sr} | {scsr} | {lv} | {calls} | {tokens} |".format(
                env=row["env"],
                controller=row["controller"],
                n=row["aggregate"]["n"],
                sr=fmt_pct(row["aggregate"]["SR"]),
                scsr=fmt_pct(row["aggregate"]["SR_given_surface_clean"]),
                lv=fmt_pct(row["aggregate"]["local_valid_edit_rate"]),
                calls=fmt_num(row["aggregate"]["avg_model_calls"]),
                tokens=fmt_num(row["aggregate"]["avg_tokens"]),
            )
        )
    lines.extend(
        [
            "",
            "## Deltas Versus Greedy",
            "",
            "| Environment | Controller | N | Delta SR | Delta surface-clean rate | Delta surface-clean SR | Delta local validity | Call ratio | Token ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        lines.append(
            "| {env} | {controller} | {n} | {dsr} | {dclean} | {dscsr} | {dlv} | {calls} | {tokens} |".format(
                env=row["env"],
                controller=row["controller"],
                n=row["n"],
                dsr=fmt_pct(row["delta_SR"]),
                dclean=fmt_pct(row["delta_surface_clean_rate"]),
                dscsr=fmt_pct(row["delta_surface_clean_SR"]),
                dlv=fmt_pct(row["delta_local_validity"]),
                calls=fmt_num(row["call_ratio"]),
                tokens=fmt_num(row["token_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Preregistered Hypothesis Checks",
            "",
            "| Hypothesis | Status | Summary |",
            "| --- | --- | --- |",
        ]
    )
    for row in hypothesis_checks:
        lines.append(f"| {row['id']} | {row['status']} | {row['summary']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output-json", default="results_v4/intervention_prediction_summary.json")
    ap.add_argument("--output-md", default="results_v4/intervention_prediction_summary.md")
    args = ap.parse_args()

    rows = load_rows(args.inputs)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("env"), row.get("controller"))].append(row)
    aggregates = []
    for (env, controller), group_rows in sorted(groups.items()):
        aggregates.append({
            "env": env,
            "controller": controller,
            "aggregate": weighted_aggregate(group_rows),
        })
    deltas = build_delta_rows(aggregates)
    hypothesis_checks = assess_preregistered_hypotheses(aggregates, deltas)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "n_source_rows": len(rows),
            "aggregates": aggregates,
            "deltas_vs_greedy": deltas,
            "preregistered_hypothesis_checks": hypothesis_checks,
        }, f, indent=2)
    write_markdown(Path(args.output_md), aggregates, deltas, hypothesis_checks)
    print(json.dumps({
        "n_source_rows": len(rows),
        "n_groups": len(aggregates),
        "n_deltas": len(deltas),
    }, indent=2))
    print(f"Saved to {out_json} and {args.output_md}")


if __name__ == "__main__":
    main()
