#!/usr/bin/env python3
"""Exact recoverability validation for restricted GB1 sequence-landscape editing.

This script independently enumerates feasible completion over the finite GB1
edit graph and compares it to the environment's stored recoverability probe. It
also audits raw LLM trajectories for exact recoverable -> unrecoverable
transitions and exact local-deception events.
"""
from __future__ import annotations

import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv

SURFACE_EVENTS = {
    "MALFORMED_ACTION",
    "INVALID_POSITION",
    "INVALID_VALUE",
    "INVALID_WORD",
    "ILLEGAL_EDIT",
    "REPEATED_EXACT_EDIT",
}


def iter_raw_files(paths: Iterable[str]) -> Iterable[Path]:
    for item in paths:
        p = Path(item)
        if p.is_file() and p.suffix == ".json":
            yield p
        elif (p / "raw").is_dir():
            yield from sorted((p / "raw").glob("*.json"))
        elif p.is_dir():
            yield from sorted(p.glob("**/raw/*.json"))


def make_env() -> GB1SequenceEnv:
    return GB1SequenceEnv()


def build_exact_solver(env: GB1SequenceEnv):
    @lru_cache(maxsize=None)
    def exact(seq_tuple: Tuple[str, ...], stability: float, budget: int) -> Tuple[bool, float]:
        seq = list(seq_tuple)
        stability = round(float(stability), 4)
        if env._success_for(seq, stability):
            return True, float(env._true_fitness(seq))
        best = float(env._true_fitness(seq))
        if budget <= 0:
            return False, best
        any_recoverable = False
        for pos, aa in env._available_edits_for(seq, stability):
            nxt = list(seq)
            nxt[pos] = aa
            nxt_stab = env._next_stability(seq, stability, pos, aa)
            rec, child_best = exact(tuple(nxt), round(float(nxt_stab), 4), budget - 1)
            best = max(best, child_best)
            any_recoverable = any_recoverable or rec
        return any_recoverable, best

    return exact


def all_variants(env: GB1SequenceEnv) -> List[Tuple[str, ...]]:
    allowed = [env._allowed[i] for i in range(4)]
    return [tuple(v) for v in itertools.product(*allowed)]


def canonical_state_grid(env: GB1SequenceEnv) -> List[Tuple[Tuple[str, ...], float]]:
    # Primary exact table requested by reviewers: all 5^4 restricted GB1 variants
    # at the canonical pre-edit stability. Path-dependent stability is audited
    # separately on every stored LLM trajectory state.
    return [(v, 1.0) for v in all_variants(env)]


def validate_probe(env: GB1SequenceEnv, max_budget: int) -> Dict[str, Any]:
    exact = build_exact_solver(env)
    states = canonical_state_grid(env)
    n = mismatches = 0
    by_budget = []
    false_pos = false_neg = 0
    max_best_gap = 0.0
    examples = []
    for b in range(max_budget + 1):
        bn = bm = bfp = bfn = 0
        for seq_tuple, stab in states:
            seq = list(seq_tuple)
            rec, best = exact(seq_tuple, stab, b)
            probe = env._recoverability_probe(seq, stab, b)
            prec = bool(probe["recoverable"])
            gap = abs(float(probe["best_true_fitness"]) - float(best))
            max_best_gap = max(max_best_gap, gap)
            n += 1
            bn += 1
            if prec != rec:
                mismatches += 1
                bm += 1
                if prec and not rec:
                    false_pos += 1
                    bfp += 1
                elif rec and not prec:
                    false_neg += 1
                    bfn += 1
                if len(examples) < 10:
                    examples.append({
                        "sequence": "".join(seq_tuple),
                        "stability": stab,
                        "budget": b,
                        "probe_recoverable": prec,
                        "exact_recoverable": rec,
                        "probe_best_true": probe["best_true_fitness"],
                        "exact_best_true": round(best, 4),
                    })
        by_budget.append({
            "budget": b,
            "n_variant_states": bn,
            "mismatches": bm,
            "agreement": (bn - bm) / bn if bn else None,
            "false_positive": bfp,
            "false_negative": bfn,
        })
    return {
        "variant_count": len(all_variants(env)),
        "canonical_variant_state_count": len(states),
        "budgets": list(range(max_budget + 1)),
        "total_state_budget_pairs": n,
        "mismatches": mismatches,
        "agreement": (n - mismatches) / n if n else None,
        "false_positive": false_pos,
        "false_negative": false_neg,
        "max_best_true_fitness_gap": round(max_best_gap, 8),
        "by_budget": by_budget,
        "mismatch_examples": examples,
    }


def step_exact_labels(env: GB1SequenceEnv, exact, step: Dict[str, Any]) -> Dict[str, Any]:
    info = step.get("info") or {}
    diag = info.get("diagnostics") or {}
    out: Dict[str, Any] = {}
    for side, struct_key in [("before", "structure_before"), ("after", "structure_after")]:
        snap = diag.get(f"recoverability_{side}") or {}
        seq_s = step.get(struct_key) or ""
        if not seq_s or "stability" not in snap or "remaining_budget" not in snap:
            continue
        rec, best = exact(tuple(seq_s), round(float(snap["stability"]), 4), int(snap["remaining_budget"]))
        out[side] = {
            "exact_recoverable": rec,
            "exact_best_true_fitness": round(best, 4),
            "probe_recoverable": bool(snap.get("recoverable")),
            "probe_best_true_fitness": snap.get("recoverability_score"),
            "local_proxy_score": snap.get("local_proxy_score"),
        }
    return out


def audit_raw(paths: Iterable[str], env: GB1SequenceEnv) -> Dict[str, Any]:
    exact = build_exact_solver(env)
    episodes = []
    n = success = surface_clean_n = surface_clean_success = 0
    exact_loss_eps = exact_loss_surface_clean_failures = 0
    exact_local_deception_eps = 0
    exact_loss_steps = exact_deception_steps = 0
    initial_exact_recoverable = 0
    initial_probe_recoverable = 0
    probe_exact_mismatch_steps = 0
    total_diag_steps = 0

    for path in iter_raw_files(paths):
        try:
            ep = json.load(open(path))
        except Exception:
            continue
        if ep.get("env_name") != "gb1_sequence":
            continue
        n += 1
        ep_success = bool(ep.get("success"))
        success += int(ep_success)
        steps = ep.get("steps") or []
        events = [e for st in steps for e in (st.get("events") or [])]
        surface_clean = not any(e in SURFACE_EVENTS for e in events)
        surface_clean_n += int(surface_clean)
        surface_clean_success += int(surface_clean and ep_success)
        has_loss = False
        has_deception = False
        first_loss_step = None
        if steps:
            first_labels = step_exact_labels(env, exact, steps[0])
            if "before" in first_labels:
                initial_exact_recoverable += int(bool(first_labels["before"]["exact_recoverable"]))
                initial_probe_recoverable += int(bool(first_labels["before"]["probe_recoverable"]))
        for st in steps:
            labels = step_exact_labels(env, exact, st)
            if "before" not in labels or "after" not in labels:
                continue
            total_diag_steps += 1
            b = labels["before"]
            a = labels["after"]
            if b["probe_recoverable"] != b["exact_recoverable"] or a["probe_recoverable"] != a["exact_recoverable"]:
                probe_exact_mismatch_steps += 1
            loss = bool(b["exact_recoverable"] and not a["exact_recoverable"])
            local_delta = None
            if b.get("local_proxy_score") is not None and a.get("local_proxy_score") is not None:
                local_delta = float(a["local_proxy_score"]) - float(b["local_proxy_score"])
            exact_best_delta = float(a["exact_best_true_fitness"]) - float(b["exact_best_true_fitness"])
            deception = bool(local_delta is not None and local_delta > 0 and (loss or exact_best_delta < 0))
            if loss:
                exact_loss_steps += 1
                has_loss = True
                if first_loss_step is None:
                    first_loss_step = st.get("t")
            if deception:
                exact_deception_steps += 1
                has_deception = True
        exact_loss_eps += int(has_loss)
        exact_local_deception_eps += int(has_deception)
        if surface_clean and not ep_success and has_loss:
            exact_loss_surface_clean_failures += 1
        episodes.append({
            "path": str(path),
            "model": ep.get("model"),
            "prompt_condition": ep.get("prompt_condition"),
            "seed": ep.get("seed"),
            "success": ep_success,
            "surface_clean": surface_clean,
            "has_exact_recoverability_loss": has_loss,
            "has_exact_local_deception": has_deception,
            "first_exact_loss_step": first_loss_step,
        })

    surface_clean_fail = surface_clean_n - surface_clean_success
    return {
        "n_gb1_episodes": n,
        "success_rate": success / n if n else None,
        "surface_clean_episodes": surface_clean_n,
        "surface_clean_success_rate": surface_clean_success / surface_clean_n if surface_clean_n else None,
        "surface_clean_failures": surface_clean_fail,
        "surface_clean_failures_with_exact_loss": exact_loss_surface_clean_failures,
        "surface_clean_failure_exact_loss_rate": exact_loss_surface_clean_failures / surface_clean_fail if surface_clean_fail else None,
        "initial_exact_recoverable_rate": initial_exact_recoverable / n if n else None,
        "initial_probe_recoverable_rate": initial_probe_recoverable / n if n else None,
        "exact_recoverability_loss_episode_rate": exact_loss_eps / n if n else None,
        "exact_local_deception_episode_rate": exact_local_deception_eps / n if n else None,
        "exact_recoverability_loss_steps": exact_loss_steps,
        "exact_local_deception_steps": exact_deception_steps,
        "total_diagnostic_steps": total_diag_steps,
        "probe_exact_label_mismatch_steps": probe_exact_mismatch_steps,
        "probe_exact_step_agreement": (total_diag_steps - probe_exact_mismatch_steps) / total_diag_steps if total_diag_steps else None,
        "episodes": episodes[:200],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Experiment dirs/raw dirs/raw JSONs for trajectory audit.")
    ap.add_argument("--max-budget", type=int, default=6)
    ap.add_argument("--out", default="results_v4/gb1_exact_recoverability/gb1_exact_recoverability_summary.json")
    args = ap.parse_args()

    env = make_env()
    validation = validate_probe(env, args.max_budget)
    audit = audit_raw(args.paths, env) if args.paths else None
    payload = {"probe_validation": validation, "trajectory_audit": audit}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        "out": str(out),
        "probe_agreement": validation["agreement"],
        "probe_mismatches": validation["mismatches"],
        "canonical_variant_state_count": validation["canonical_variant_state_count"],
        "trajectory_audit": None if audit is None else {
            k: audit[k] for k in [
                "n_gb1_episodes",
                "surface_clean_episodes",
                "surface_clean_success_rate",
                "surface_clean_failure_exact_loss_rate",
                "initial_exact_recoverable_rate",
                "exact_recoverability_loss_episode_rate",
                "exact_local_deception_episode_rate",
                "probe_exact_step_agreement",
            ]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
