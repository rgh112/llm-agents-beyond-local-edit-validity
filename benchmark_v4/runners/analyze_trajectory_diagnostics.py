#!/usr/bin/env python3
"""Aggregate recoverability and local-deception diagnostics from raw logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


SURFACE_EVENTS = {
    "MALFORMED_ACTION",
    "INVALID_POSITION",
    "INVALID_VALUE",
    "INVALID_WORD",
    "ILLEGAL_EDIT",
    "REPEATED_EXACT_EDIT",
}


def _iter_raw_files(paths: Iterable[str]) -> Iterable[Path]:
    for item in paths:
        p = Path(item)
        if p.is_file():
            yield p
        elif (p / "raw").is_dir():
            yield from sorted((p / "raw").glob("*.json"))
        elif p.is_dir():
            yield from sorted(p.glob("**/raw/*.json"))


def _load(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _key(ep: Dict[str, Any]) -> tuple:
    return (
        ep.get("env_name"),
        ep.get("model"),
        ep.get("prompt_condition"),
        ep.get("memory_condition"),
        ep.get("regime"),
    )


def _step_diag(step: Dict[str, Any]) -> Dict[str, Any]:
    return ((step.get("info") or {}).get("diagnostics") or {})


def _episode_row(ep: Dict[str, Any]) -> Dict[str, Any]:
    steps = ep.get("steps") or []
    diag_steps = [_step_diag(s) for s in steps if _step_diag(s)]
    events = [e for s in steps for e in (s.get("events") or [])]
    surface_clean = not any(e in SURFACE_EVENTS for e in events)
    rec_drops = [d for d in diag_steps if d.get("recoverability_drop")]
    deceptions = [d for d in diag_steps if d.get("local_improvement_deception")]
    score_deltas = [
        float(d["recoverability_score_delta"])
        for d in diag_steps
        if d.get("recoverability_score_delta") is not None
    ]
    local_deltas = [
        float(d["local_proxy_delta"])
        for d in diag_steps
        if d.get("local_proxy_delta") is not None
    ]
    finalize_steps = [
        s for s in steps
        if (s.get("parsed_action") or {}).get("type") == "finalize"
    ]
    finalize_success = None
    if finalize_steps:
        finfo = finalize_steps[-1].get("info") or {}
        finalize_success = bool(finfo.get("success"))
    return {
        "success": bool(ep.get("success")),
        "surface_clean": surface_clean,
        "n_steps": len(steps),
        "n_diag_steps": len(diag_steps),
        "n_recoverability_drops": len(rec_drops),
        "n_local_deceptions": len(deceptions),
        "has_recoverability_drop": bool(rec_drops),
        "has_local_deception": bool(deceptions),
        "mean_recoverability_delta": mean(score_deltas) if score_deltas else None,
        "mean_local_proxy_delta": mean(local_deltas) if local_deltas else None,
        "finalized": bool(finalize_steps),
        "finalize_success": finalize_success,
    }


def _avg(values: List[Any]):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Experiment dirs, raw dirs, or raw JSON files.")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    rows_by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for path in _iter_raw_files(args.paths):
        ep = _load(path)
        rows_by_key[_key(ep)].append(_episode_row(ep))

    results = []
    for key, rows in sorted(rows_by_key.items()):
        n = len(rows)
        surface_clean = [r for r in rows if r["surface_clean"]]
        finalized = [r for r in rows if r["finalized"]]
        results.append({
            "env_name": key[0],
            "model": key[1],
            "prompt_condition": key[2],
            "memory_condition": key[3],
            "regime": key[4],
            "n": n,
            "success_rate": sum(r["success"] for r in rows) / n,
            "surface_clean_rate": sum(r["surface_clean"] for r in rows) / n,
            "success_given_surface_clean": (
                sum(r["success"] for r in surface_clean) / len(surface_clean)
                if surface_clean else None
            ),
            "recoverability_drop_episode_rate": (
                sum(r["has_recoverability_drop"] for r in rows) / n
            ),
            "local_deception_episode_rate": (
                sum(r["has_local_deception"] for r in rows) / n
            ),
            "recoverability_drops_per_episode": (
                sum(r["n_recoverability_drops"] for r in rows) / n
            ),
            "local_deceptions_per_episode": (
                sum(r["n_local_deceptions"] for r in rows) / n
            ),
            "mean_recoverability_delta": _avg(
                [r["mean_recoverability_delta"] for r in rows]
            ),
            "mean_local_proxy_delta": _avg(
                [r["mean_local_proxy_delta"] for r in rows]
            ),
            "finalize_rate": len(finalized) / n,
            "finalize_success_rate": (
                sum(r["finalize_success"] for r in finalized) / len(finalized)
                if finalized else None
            ),
        })

    payload = {"results": results}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)

    print(
        "| env | model | prompt | memory | regime | n | SR | surface-clean SR | "
        "rec-drop eps | deception eps |"
    )
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        sc = r["success_given_surface_clean"]
        print(
            f"| {r['env_name']} | {r['model']} | {r['prompt_condition']} | "
            f"{r['memory_condition']} | {r['regime']} | {r['n']} | "
            f"{r['success_rate']:.3f} | "
            f"{'' if sc is None else f'{sc:.3f}'} | "
            f"{r['recoverability_drop_episode_rate']:.3f} | "
            f"{r['local_deception_episode_rate']:.3f} |"
        )


if __name__ == "__main__":
    main()
