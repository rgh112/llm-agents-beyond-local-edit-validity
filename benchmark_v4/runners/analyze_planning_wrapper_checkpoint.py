#!/usr/bin/env python3
"""Controller-aware checkpoint table for planning-wrapper sweeps."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SURFACE_EVENTS = {
    "MALFORMED_ACTION",
    "INVALID_ACTION",
    "INVALID_POSITION",
    "INVALID_VALUE",
    "INVALID_WORD",
    "ILLEGAL_EDIT",
    "REPEATED_EDIT",
    "REPEATED_EXACT_EDIT",
}

API_ERROR_MARKERS = (
    "[API_ERROR:",
    "nodename nor servname",
    "read operation timed out",
    "CERTIFICATE_VERIFY_FAILED",
    "urlopen error",
)


def _iter_raw_files(paths: Iterable[str]) -> Iterable[Path]:
    for item in paths:
        path = Path(item)
        if path.is_file():
            yield path
        elif (path / "raw").is_dir():
            yield from sorted((path / "raw").glob("*.json"))
        elif path.is_dir():
            yield from sorted(path.glob("*.json"))
        else:
            raise FileNotFoundError(path)


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _key(ep: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        ep["env_name"],
        ep["model"],
        ep["prompt_condition"],
        ep["memory_condition"],
        ep["regime"],
    )


def _contains_api_error(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in API_ERROR_MARKERS)
    if isinstance(value, dict):
        return any(_contains_api_error(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_api_error(v) for v in value)
    return False


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def _summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(episodes)
    successes = sum(bool(ep.get("success")) for ep in episodes)
    terminal = Counter(
        ep.get("terminal_failure") for ep in episodes if ep.get("terminal_failure")
    )
    edit_attempts = 0
    valid_edits = 0
    surface_clean = 0
    surface_clean_successes = 0
    calls = 0
    tokens = 0

    for ep in episodes:
        ep_events = []
        for step in ep.get("steps", []):
            if step.get("parsed_action", {}).get("type") == "edit":
                edit_attempts += 1
                if step.get("valid"):
                    valid_edits += 1
            ep_events.extend(step.get("events") or [])

        is_surface_clean = not any(event in SURFACE_EVENTS for event in ep_events)
        if is_surface_clean:
            surface_clean += 1
            if ep.get("success"):
                surface_clean_successes += 1

        usage = ep.get("token_usage") or {}
        calls += int(usage.get("controller_model_calls") or 0)
        tokens += int(usage.get("total_tokens") or 0)

    return {
        "n": n,
        "n_success": successes,
        "SR": successes / n if n else None,
        "local_valid_edit_rate": valid_edits / edit_attempts if edit_attempts else None,
        "n_edit_attempts": edit_attempts,
        "n_local_valid_edits": valid_edits,
        "n_surface_clean_episodes": surface_clean,
        "n_surface_clean_successes": surface_clean_successes,
        "SR_given_surface_clean": (
            surface_clean_successes / surface_clean if surface_clean else None
        ),
        "avg_model_calls": calls / n if n else None,
        "avg_tokens": tokens / n if n else None,
        "terminal_failures": dict(terminal),
        "top_terminal_failure": terminal.most_common(1)[0][0] if terminal else None,
    }


def _render_md(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Planning-wrapper checkpoint by controller",
        "",
        "Completed controller-aware checkpoint; controllers are kept separate and quarantined API/DNS-error logs are excluded by default.",
        "",
        "| Env | Model | Prompt | Controller | N | SR | LocalValid | SurfClean | SR|SurfClean | Calls/ep | Tokens/ep | Top terminal failure |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        surf = f"{row['n_surface_clean_episodes']}/{row['n']}"
        scsr = (
            f"{row['n_surface_clean_successes']}/{row['n_surface_clean_episodes']}"
            if row["n_surface_clean_episodes"]
            else "—"
        )
        lines.append(
            f"| {row['env']} | {row['model']} | {row['prompt']} | "
            f"{row['controller']} | {row['n']} | {_pct(row['SR'])} | "
            f"{_pct(row['local_valid_edit_rate'])} | {surf} | {scsr} | "
            f"{row['avg_model_calls']:.1f} | {row['avg_tokens']:.0f} | "
            f"{row['top_terminal_failure'] or '—'} |"
        )
    return "\n".join(lines).replace("| SR|SurfClean |", "| SR\\|SurfClean |") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dirs",
        nargs="+",
        required=True,
        help="Raw dirs, experiment dirs containing raw/, or raw JSON files.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--label",
        default="small_block_text_visible_checkpoint_by_controller",
    )
    parser.add_argument(
        "--include-api-error-episodes",
        action="store_true",
        help="Include raw episodes containing API/DNS/SSL error strings.",
    )
    args = parser.parse_args()

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    excluded_api_error_paths = []
    for path in _iter_raw_files(args.raw_dirs):
        ep = _load(path)
        if not args.include_api_error_episodes and _contains_api_error(ep):
            excluded_api_error_paths.append(str(path))
            continue
        groups[_key(ep)].append(ep)

    rows = []
    for (env, model, prompt, memory, regime), episodes in sorted(groups.items()):
        row = {
            "env": env,
            "model": model,
            "prompt": prompt,
            "memory": memory,
            "regime": regime,
            "controller": regime.split(":")[-1],
        }
        row.update(_summarize(episodes))
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "planning_wrapper_checkpoint_by_controller.json"
    out_md = out_dir / "planning_wrapper_checkpoint_by_controller.md"
    payload = {
        "label": args.label,
        "source_raw_dirs": [str(Path(p)) for p in args.raw_dirs],
        "status": "completed controller-aware checkpoint; quarantined API/DNS-error episodes excluded by default",
        "api_error_episode_policy": (
            "included" if args.include_api_error_episodes else "excluded"
        ),
        "n_excluded_api_error_episodes": len(excluded_api_error_paths),
        "excluded_api_error_paths": excluded_api_error_paths,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text(_render_md(rows))
    print(out_md.read_text())
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
