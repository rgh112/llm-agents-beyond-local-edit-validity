"""Shared helpers for next-cycle constructive-editing experiments."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

from benchmark_v4.model_registry import get_model_metadata

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


def parse_seeds(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def model_short_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def safe_name(value) -> str:
    return str(value).replace("/", "__").replace(":", "_").replace(" ", "_")


def make_env(env_name: str, **kwargs):
    if env_name == "word_ladder":
        from benchmark_v4.envs.word_ladder import WordLadderEnv
        return WordLadderEnv(
            data_dir=kwargs.pop("data_dir", "wordladder_data"),
            word_length=int(kwargs.pop("word_length", 4)),
            shortest_path_range=kwargs.pop("shortest_path_range", (3, 6)),
            **kwargs,
        )
    if env_name == "alloy":
        from benchmark_v4.envs.alloy import AlloyEnv
        return AlloyEnv(**kwargs)
    if env_name == "gb1_sequence":
        from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv
        return GB1SequenceEnv(**kwargs)
    if env_name == "document_plan":
        from benchmark_v4.envs.document_plan import DocumentPlanEnv
        return DocumentPlanEnv(**kwargs)
    raise ValueError(f"Unknown env: {env_name}")


def summarize_episode_collection(rows: Iterable[Any]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    successes = sum(1 for r in rows if r.success)
    events = Counter()
    terminal = Counter()
    total_edits = 0
    local_valid = 0
    surface_clean = 0
    surface_clean_success = 0
    model_calls = 0
    total_tokens = 0

    for summary in rows:
        if summary.terminal_failure:
            terminal[summary.terminal_failure] += 1
        for ename, count in summary.event_counts.items():
            events[ename] += count
        ep_surface = False
        for step in summary.steps:
            if step.parsed_action and step.parsed_action.get("type") == "edit":
                total_edits += 1
                if step.valid:
                    local_valid += 1
            if any((e.name if hasattr(e, "name") else str(e)) in SURFACE_EVENTS
                   for e in step.events):
                ep_surface = True
        if not ep_surface:
            surface_clean += 1
            if summary.success:
                surface_clean_success += 1
        model_calls += int(summary.token_usage.get("controller_model_calls", 0))
        total_tokens += int(summary.token_usage.get("total_tokens", 0))

    sr = successes / n if n else 0.0
    avg_calls = model_calls / n if n else None
    avg_tokens = total_tokens / n if n else None
    return {
        "SR": sr,
        "n_success": successes,
        "n_total": n,
        "terminal_failures": dict(terminal),
        "all_events": dict(events),
        "surface_failure_count": sum(events.get(e, 0) for e in SURFACE_EVENTS),
        "structural_failure_count": sum(events.get(e, 0) for e in STRUCTURAL_EVENTS),
        "local_valid_edit_rate": local_valid / total_edits if total_edits else None,
        "n_edit_attempts": total_edits,
        "n_local_valid_edits": local_valid,
        "n_surface_clean_episodes": surface_clean,
        "n_surface_clean_successes": surface_clean_success,
        "SR_given_surface_clean": (
            surface_clean_success / surface_clean if surface_clean else None
        ),
        "avg_model_calls": avg_calls,
        "avg_tokens": avg_tokens,
        "SR_per_100_calls": (
            sr / avg_calls * 100.0 if avg_calls and avg_calls > 0 else None
        ),
        "SR_per_100k_tokens": (
            sr / avg_tokens * 100000.0 if avg_tokens and avg_tokens > 0 else None
        ),
    }


def attach_model_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    model_id = row.get("model", "")
    meta = get_model_metadata(model_id)
    for key, value in meta.items():
        row[f"model_{key}"] = value
    return row


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def raw_episode_path(
    output_dir: str | Path,
    *,
    env_name: str,
    prompt_name: str,
    memory_name: str,
    regime: str,
    model_id: str,
    seed: int,
    sensitivity_setting: str | None = None,
) -> Path:
    setting_part = f"_{safe_name(sensitivity_setting)}" if sensitivity_setting else ""
    filename = (
        f"{env_name}_{prompt_name}_{memory_name}{setting_part}_"
        f"{safe_name(regime)}_{safe_name(model_id)}_seed{seed}.json"
    )
    return Path(output_dir) / "raw" / filename


def load_raw_episode(path: str | Path):
    """Load a raw JSON episode into the minimal EpisodeSummary-like shape."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    steps = []
    event_counts = Counter()
    finalize_step = None
    first_failure_step = None
    for step in data.get("steps", []):
        events = list(step.get("events", []))
        for event in events:
            event_counts[str(event)] += 1
            if first_failure_step is None:
                first_failure_step = step.get("t")
        parsed = step.get("parsed_action")
        if parsed and parsed.get("type") == "finalize":
            finalize_step = step.get("t")
        steps.append(SimpleNamespace(
            t=step.get("t"),
            parsed_action=parsed,
            valid=bool(step.get("valid")),
            events=events,
        ))
    token_usage = dict(data.get("token_usage") or {})
    return SimpleNamespace(
        env_name=data.get("env_name"),
        regime=data.get("regime"),
        model=data.get("model"),
        prompt_condition=data.get("prompt_condition"),
        memory_condition=data.get("memory_condition"),
        seed=data.get("seed"),
        steps=steps,
        success=bool(data.get("success")),
        terminal_failure=data.get("terminal_failure"),
        failure_events=list(data.get("failure_events") or []),
        final_metrics=dict(data.get("final_metrics") or {}),
        token_usage=token_usage,
        wall_time_s=float(data.get("wall_time_s") or 0.0),
        event_counts=dict(event_counts),
        finalize_step=finalize_step,
        first_failure_step=first_failure_step,
        prompt_metadata={},
    )
