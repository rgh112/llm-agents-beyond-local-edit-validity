#!/usr/bin/env python3
"""Run cost-matched independent restart baselines.

This is a deliberately strong sampling-only baseline for planning-wrapper
claims. For each task seed, it runs K independent full trajectories with the
same environment seed and reports task success if any restart succeeds. This
separates search/selection gains from simply spending more model calls.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.model_panels import available_panels, expand_model_selection
from benchmark_v4.runners.experiment_utils import (
    STRUCTURAL_EVENTS,
    SURFACE_EVENTS,
    attach_model_metadata,
    load_raw_episode,
    make_env,
    parse_seeds,
    raw_episode_path,
    write_json,
)


ENV_STRUCTURAL_PROMPT = {
    "word_ladder": "scaffold",
    "alloy": "self_check",
    "gb1_sequence": "self_check",
    "document_plan": "scaffold",
}

ENV_DEFAULT_HORIZON = {
    "word_ladder": 30,  # shortest-path range 3--6 with 5x budget factor
    "alloy": 8,
    "gb1_sequence": 6,
    "document_plan": 6,
}


def expand_prompt(env_name: str, prompt: str) -> str:
    return ENV_STRUCTURAL_PROMPT[env_name] if prompt == "__structural__" else prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--model-panel", default=None, choices=available_panels(),
                    help="Pre-registered model panel. Can be combined with --models.")
    ap.add_argument("--envs", nargs="+", default=["word_ladder", "alloy", "gb1_sequence"])
    ap.add_argument("--prompts", nargs="+", default=["__structural__"])
    ap.add_argument("--memory", default="state_only")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--restarts", type=int, default=5,
                    help="Independent full-trajectory attempts per task seed.")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="Optional smoke-test cap applied after env construction.")
    ap.add_argument("--output-suffix", default="independent_restarts")
    ap.add_argument("--output-root", default="results_v4",
                    help="Root directory for newly created output folders.")
    ap.add_argument("--resume-from", default=None,
                    help="Existing output directory. Completed raw episodes are reused.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.models = expand_model_selection(args.models, args.model_panel)

    seeds = parse_seeds(args.seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.resume_from) if args.resume_from else Path(
        args.output_root
    ) / f"independent_restarts_{args.output_suffix}_{timestamp}"

    plan = []
    for model_id in args.models:
        for env_name in args.envs:
            horizon = ENV_DEFAULT_HORIZON.get(env_name)
            if horizon is None:
                horizon = make_env(env_name).max_steps
            if args.max_episode_steps is not None:
                horizon = min(horizon, args.max_episode_steps)
            for prompt_spec in args.prompts:
                prompt_name = expand_prompt(env_name, prompt_spec)
                max_calls_per_task = horizon * args.restarts
                plan.append((
                    model_id,
                    env_name,
                    prompt_name,
                    len(seeds),
                    args.restarts,
                    horizon,
                    max_calls_per_task,
                ))

    total_attempts = sum(row[3] * row[4] for row in plan)
    max_planned_model_calls = sum(row[3] * row[6] for row in plan)
    print("=== Independent Restart Baseline ===")
    print(f"Models:    {args.models}")
    print(f"Envs:      {args.envs}")
    print(f"Prompts:   {args.prompts}")
    print(f"Memory:    {args.memory}")
    print(f"Seeds:     {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Restarts:  {args.restarts}")
    print(f"Attempts:  {total_attempts}")
    print(f"Max calls: {max_planned_model_calls} (dry-run upper bound)")
    print(f"Output:    {output_dir}")
    if args.dry_run:
        for row in plan:
            print("  ", row)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("TOGETHER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        print("ERROR: no API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(2)

    from benchmark_v4.controllers import get_controller
    from benchmark_v4.logging_utils.episode_logger import EpisodeLogger
    from benchmark_v4.memory import get_memory_module
    from benchmark_v4.models.api_model import APIModel
    from benchmark_v4.prompts import get_prompt_builder
    from benchmark_v4.runners.single_episode_runner import run_episode_with_controller

    logger = EpisodeLogger(str(output_dir))
    controller = get_controller("greedy_sampled")
    model_cache = {
        model_id: APIModel(
            model_id=model_id,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.api_timeout,
            max_retries=args.max_retries,
        )
        for model_id in args.models
    }

    rows = []
    for model_id, env_name, prompt_name, _, _, horizon, max_calls_per_task in plan:
        start = time.time()
        task_successes = 0
        attempt_successes = 0
        terminal = Counter()
        events = Counter()
        total_tokens = 0
        total_calls = 0
        failed_task_terminal = Counter()

        for seed in seeds:
            seed_succeeded = False
            seed_failures = Counter()
            for restart in range(args.restarts):
                regime = f"constructive_editing:greedy_sampled:independent_restart_r{restart}"
                resume_path = raw_episode_path(
                    output_dir,
                    env_name=env_name,
                    prompt_name=prompt_name,
                    memory_name=args.memory,
                    regime=regime,
                    model_id=model_id.split("/")[-1],
                    seed=seed,
                )
                summary = load_raw_episode(resume_path)
                if summary is None:
                    env = make_env(env_name)
                    if args.max_episode_steps is not None:
                        env.max_steps = min(env.max_steps, args.max_episode_steps)
                    prompt = get_prompt_builder(prompt_name)
                    memory = get_memory_module(args.memory)
                    summary = run_episode_with_controller(
                        env=env,
                        prompt_builder=prompt,
                        memory_module=memory,
                        model=model_cache[model_id],
                        controller=controller,
                        seed=seed,
                        verbose=False,
                        max_episode_steps=args.max_episode_steps,
                    )
                    summary.regime = regime
                    logger.log_episode(summary)

                if summary.success:
                    attempt_successes += 1
                    seed_succeeded = True
                if summary.terminal_failure:
                    terminal[summary.terminal_failure] += 1
                    seed_failures[summary.terminal_failure] += 1
                for name, count in summary.event_counts.items():
                    events[name] += count
                total_tokens += summary.token_usage.get("total_tokens", 0)
                total_calls += int(summary.token_usage.get("controller_model_calls", 0) or 0)

            if seed_succeeded:
                task_successes += 1
            elif seed_failures:
                failed_task_terminal[seed_failures.most_common(1)[0][0]] += 1

            print(
                f"  {model_id}|{env_name}|{prompt_name}|seed{seed} -> "
                f"{'OK' if seed_succeeded else 'FAIL'}",
                flush=True,
            )

        n_tasks = len(seeds)
        n_attempts = n_tasks * args.restarts
        row = {
            "model": model_id,
            "env": env_name,
            "prompt": prompt_name,
            "memory": args.memory,
            "controller": "independent_restarts",
            "temperature": args.temperature,
            "restarts": args.restarts,
            "planned_horizon": horizon,
            "planned_max_calls_per_task": max_calls_per_task,
            "n_total": n_tasks,
            "n_success": task_successes,
            "SR": task_successes / n_tasks if n_tasks else 0.0,
            "n_attempts": n_attempts,
            "n_attempt_successes": attempt_successes,
            "attempt_SR": attempt_successes / n_attempts if n_attempts else 0.0,
            "terminal_failures": dict(failed_task_terminal),
            "attempt_terminal_failures": dict(terminal),
            "event_counts": dict(events),
            "surface_failure_count": sum(events.get(e, 0) for e in SURFACE_EVENTS),
            "structural_failure_count": sum(events.get(e, 0) for e in STRUCTURAL_EVENTS),
            "avg_model_calls": total_calls / n_tasks if n_tasks else None,
            "avg_tokens": total_tokens / n_tasks if n_tasks else None,
            "SR_per_100_calls": (
                100 * task_successes / total_calls if total_calls else None
            ),
            "SR_per_100k_tokens": (
                100000 * task_successes / total_tokens if total_tokens else None
            ),
            "wall_time_s": time.time() - start,
        }
        attach_model_metadata(row)
        rows.append(row)

    summary = {
        "experiment": "independent_restarts",
        "config": vars(args),
        "cost_plan": {
            "total_task_seeds": sum(row[3] for row in plan),
            "total_attempts": total_attempts,
            "max_planned_model_calls": max_planned_model_calls,
            "call_budget_is_upper_bound": True,
        },
        "results": rows,
    }
    out = output_dir / "independent_restarts_summary.json"
    write_json(out, summary)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
