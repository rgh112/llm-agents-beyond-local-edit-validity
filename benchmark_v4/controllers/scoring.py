"""Visible heuristic scoring for controller simulations."""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Tuple


SURFACE_EVENTS = {
    "MALFORMED_ACTION", "INVALID_POSITION", "INVALID_VALUE",
    "INVALID_WORD", "ILLEGAL_EDIT", "REPEATED_EXACT_EDIT",
}


def event_name(event: Any) -> str:
    return event.name if hasattr(event, "name") else str(event)


def _parse_float_after(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def text_visible_score(env) -> float:
    """Score using only the rendered observation text.

    This scorer is intentionally conservative. It supports the main non-oracle
    search condition where reviewers may worry about environment leakage.
    """
    text = env.render_observation()
    score = 0.0
    if env.env_name == "word_ladder":
        mismatches = text.count("^")
        return -float(mismatches)
    if env.env_name == "alloy":
        uts = _parse_float_after(r"UTS:\s*~?([0-9,.]+)", text)
        density = _parse_float_after(r"Density:\s*([0-9.]+)", text)
        if uts is not None:
            score += uts / 1000.0
        if density is not None:
            score -= density
        if "ABOVE limit" in text:
            score -= 2.0
        if "WARNING" in text:
            score -= 2.0
        return score
    if env.env_name == "gb1_sequence":
        fitness = _parse_float_after(r"Estimated fitness:\s*([0-9.\-]+)", text)
        if fitness is not None:
            score += fitness
        if "Stability: stable" in text:
            score += 0.5
        elif "Stability: moderate" in text:
            score += 0.25
        elif "Stability: critical" in text:
            score -= 2.0
        return score
    return score


def progress_score(env, scorer: str = "proxy") -> float:
    """Environment-aware visible score for a current state.

    The score is intentionally heuristic and should be reported as a controller
    design choice, not an oracle. It uses quantities already exposed by the
    environment implementation to rank simulated local edits.
    """
    if scorer == "text_visible":
        return text_visible_score(env)

    name = env.env_name
    if name == "word_ladder":
        cur = getattr(env.state, "structure", "")
        target = getattr(env, "target_word", None) or getattr(env, "_target_word", None)
        if target:
            mismatches = sum(1 for a, b in zip(str(cur), str(target)) if a != b)
            return -float(mismatches)
        return 0.0

    if name == "alloy":
        comp = env.state.structure
        uts = env._predict_uts(comp)
        density = env._calculate_density(comp)
        rc = env.state.latent.get("recovery_cost", 0.0)
        uts_gap = max(0.0, env._uts_target - uts) / max(env._uts_target, 1.0)
        density_gap = max(0.0, density - env._density_target)
        return -(2.0 * uts_gap + density_gap + 0.25 * rc)

    if name == "gb1_sequence":
        seq = env.state.structure
        if scorer == "oracle":
            return float(env._true_fitness(seq)) + 0.5 * float(env.state.latent.get("stability", 0.0))
        est = env._estimated_fitness(seq)
        stability = env.state.latent.get("stability", 0.0)
        return float(est) + 0.5 * float(stability)

    return 0.0


def simulate_one_step(env, raw_action: str, scorer: str = "proxy") -> Tuple[Any, Dict[str, Any]]:
    sim = copy.deepcopy(env)
    before = progress_score(sim, scorer=scorer)
    result = sim.step(raw_action)
    events = [event_name(e) for e in result.events]
    parsed = sim.trajectory[-1].parsed_action if sim.trajectory else None
    valid = bool(result.info.get("valid"))
    score = progress_score(sim, scorer=scorer)
    if sim.is_success():
        score += 1000.0
    if sim.is_done() and not sim.is_success():
        score -= 10.0
    if not valid:
        score -= 5.0
    score -= sum(1.0 for e in events if e in SURFACE_EVENTS)
    score -= 0.25 * max(0, len(events) - sum(1 for e in events if e in SURFACE_EVENTS))
    if parsed and parsed.get("type") == "finalize" and not sim.is_success():
        score -= 3.0
    return sim, {
        "raw_action": raw_action,
        "parsed_action": parsed,
        "valid": valid,
        "events": events,
        "score": score,
        "progress_before": before,
        "progress_after": progress_score(sim, scorer=scorer),
        "scorer": scorer,
        "done": sim.is_done(),
        "success": sim.is_success(),
    }
