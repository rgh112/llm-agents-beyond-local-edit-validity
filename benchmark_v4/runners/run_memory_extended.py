#!/usr/bin/env python3
"""Run extended memory experiments for constructive editing."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.runners.experiment_utils import (
    make_env,
    model_short_name,
    parse_seeds,
    summarize_episode_collection,
    attach_model_metadata,
    load_raw_episode,
    raw_episode_path,
    write_json,
)
from benchmark_v4.model_panels import available_panels, expand_model_selection


ENV_STRUCTURAL_PROMPT = {
    "word_ladder": "scaffold",
    "alloy": "self_check",
    "gb1_sequence": "self_check",
    "document_plan": "scaffold",
}

ENV_DEFAULT_HORIZON = {
    "word_ladder": 30,
    "alloy": 8,
    "gb1_sequence": 6,
    "document_plan": 6,
}


def controller_call_budget(controller: str) -> int:
    if controller in {"greedy", "greedy_sampled"}:
        return 1
    if controller in {"self_consistency", "loop_avoidant", "backtracking"}:
        return 5
    if controller == "beam":
        return 12
    return 1


def expand_prompt(env_name: str, prompt: str) -> str:
    return ENV_STRUCTURAL_PROMPT[env_name] if prompt == "__structural__" else prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--model-panel", default=None, choices=available_panels(),
                    help="Pre-registered model panel. Can be combined with --models.")
    ap.add_argument("--envs", nargs="+", default=["word_ladder", "alloy", "gb1_sequence"])
    ap.add_argument("--prompts", nargs="+", default=["__structural__"])
    ap.add_argument("--memories", nargs="+",
                    default=["state_only", "window_1", "window_3",
                             "full_history", "summary", "best_state",
                             "randomized_history", "misleading_history"])
    ap.add_argument("--controller", default="greedy")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="Optional smoke-test cap applied after env construction.")
    ap.add_argument("--output-suffix", default="memory")
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
    ) / f"memory_extended_{args.output_suffix}_{timestamp}"

    plan = []
    calls_per_action = controller_call_budget(args.controller)
    max_planned_model_calls = 0
    for model_id in args.models:
        for env_name in args.envs:
            horizon = ENV_DEFAULT_HORIZON.get(env_name)
            if horizon is None:
                horizon = make_env(env_name).max_steps
            if args.max_episode_steps is not None:
                horizon = min(horizon, args.max_episode_steps)
            for prompt_spec in args.prompts:
                prompt_name = expand_prompt(env_name, prompt_spec)
                for memory_name in args.memories:
                    plan.append((model_id, env_name, prompt_name, memory_name, len(seeds), horizon))
                    max_planned_model_calls += len(seeds) * horizon * calls_per_action

    total = sum(row[4] for row in plan)
    print("=== Extended Memory Experiment ===")
    print(f"Models:   {args.models}")
    print(f"Envs:     {args.envs}")
    print(f"Prompts:  {args.prompts}")
    print(f"Memories: {args.memories}")
    print(f"Total:    {total} episodes")
    print(f"Max calls:{max_planned_model_calls} (dry-run upper bound)")
    print(f"Output:   {output_dir}")
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

    from benchmark_v4.models.api_model import APIModel
    from benchmark_v4.controllers import get_controller
    from benchmark_v4.logging_utils.episode_logger import EpisodeLogger
    from benchmark_v4.memory import get_memory_module
    from benchmark_v4.prompts import get_prompt_builder
    from benchmark_v4.runners.single_episode_runner import run_episode_with_controller

    logger = EpisodeLogger(str(output_dir))
    controller = get_controller(args.controller)
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

    episodes_by_cell = defaultdict(list)
    summaries = []
    done = 0
    started = time.time()

    for model_id, env_name, prompt_name, memory_name, _, horizon in plan:
        for seed in seeds:
            resume_path = raw_episode_path(
                output_dir,
                env_name=env_name,
                prompt_name=prompt_name,
                memory_name=memory_name,
                regime=f"constructive_editing:{args.controller}",
                model_id=model_id.split("/")[-1],
                seed=seed,
            )
            summary = load_raw_episode(resume_path)
            resumed = summary is not None
            if summary is None:
                env = make_env(env_name)
                if args.max_episode_steps is not None:
                    env.max_steps = min(env.max_steps, args.max_episode_steps)
                summary = run_episode_with_controller(
                    env=env,
                    prompt_builder=get_prompt_builder(prompt_name),
                    memory_module=get_memory_module(memory_name),
                    model=model_cache[model_id],
                    controller=controller,
                    seed=seed,
                    max_episode_steps=args.max_episode_steps,
                )
                logger.log_episode(summary)
            key = (model_id, env_name, prompt_name, memory_name)
            episodes_by_cell[key].append(summary)
            done += 1
            status = "OK" if summary.success else "FAIL"
            print(
                f"[{done}/{total}] {model_short_name(model_id)}|{env_name}|"
                f"{prompt_name}|{memory_name}|s{seed} -> {status}"
                f"{' (resumed)' if resumed else ''}",
                flush=True,
            )

        cell = episodes_by_cell[(model_id, env_name, prompt_name, memory_name)]
        row = summarize_episode_collection(cell)
        row.update({
            "model": model_id,
            "env": env_name,
            "prompt": prompt_name,
            "memory": memory_name,
            "controller": args.controller,
            "planned_horizon": horizon,
            "planned_calls_per_action": calls_per_action,
            "planned_max_calls_per_episode": horizon * calls_per_action,
        })
        attach_model_metadata(row)
        summaries.append(row)
        write_json(output_dir / "memory_extended_summary.json", {
            "config": vars(args),
            "cost_plan": {
                "total_episodes_planned": total,
                "max_planned_model_calls": max_planned_model_calls,
                "call_budget_is_upper_bound": True,
            },
            "results": summaries,
            "total_episodes_so_far": done,
            "complete": False,
        })

    write_json(output_dir / "memory_extended_summary.json", {
        "config": vars(args),
        "cost_plan": {
            "total_episodes_planned": total,
            "max_planned_model_calls": max_planned_model_calls,
            "call_budget_is_upper_bound": True,
        },
        "results": summaries,
        "total_episodes": done,
        "wall_time_s": time.time() - started,
        "complete": True,
    })

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
