#!/usr/bin/env python3
"""Run controlled planning/search wrapper experiments.

This is the next-cycle experiment for testing whether simple search wrappers
reduce structural failures while preserving the EDIT/FINALIZE environment
interface.
"""
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
    "word_ladder": 30,  # shortest-path range 3--6 with 5x budget factor
    "alloy": 8,
    "gb1_sequence": 6,
    "document_plan": 6,
}


def expand_prompt(env_name: str, prompt: str) -> str:
    return ENV_STRUCTURAL_PROMPT[env_name] if prompt == "__structural__" else prompt


def parse_controller_temperatures(spec: str):
    temps = {
        "greedy": 0.0,
        "greedy_sampled": 0.7,
        "self_consistency": 0.7,
        "beam": 0.7,
        "tot_style_beam": 0.7,
        "loop_avoidant": 0.7,
        "backtracking": 0.7,
        "reflexion_retry": 0.7,
    }
    if not spec:
        return temps
    for item in spec.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        temps[name.strip()] = float(value)
    return temps


def controller_call_budget(controller: str, *, k: int, beam_width: int, beam_depth: int) -> int:
    """Worst-case model calls per environment action for one controller.

    This is a planning-time upper bound used for experiment design and dry-run
    auditability. Actual calls can be lower when trajectories terminate early or
    beam nodes are already done.
    """
    if controller == "greedy":
        return 1
    if controller == "reflexion_retry":
        return 2
    if controller in {"greedy_sampled", "self_consistency", "loop_avoidant", "backtracking"}:
        return k
    if controller in {"beam", "tot_style_beam"}:
        calls = 0
        nodes = 1
        for _ in range(max(0, beam_depth)):
            calls += nodes * k
            nodes = beam_width
        return max(1, calls)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--model-panel", default=None, choices=available_panels(),
                    help="Pre-registered model panel. Can be combined with --models.")
    ap.add_argument("--envs", nargs="+", default=["word_ladder", "alloy", "gb1_sequence"])
    ap.add_argument("--prompts", nargs="+", default=["__structural__"])
    ap.add_argument("--controllers", nargs="+",
                    default=["greedy", "greedy_sampled", "self_consistency", "beam", "loop_avoidant"])
    ap.add_argument("--scorer", default="text_visible",
                    choices=["text_visible", "proxy", "oracle"],
                    help="Scorer for sampling/search controllers. Use oracle only as an upper bound.")
    ap.add_argument("--memory", default="state_only")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override temperature for all controllers.")
    ap.add_argument("--controller-temperatures", default="",
                    help="Comma list like greedy=0,greedy_sampled=0.7,beam=0.7.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="Optional smoke-test cap applied after env construction.")
    ap.add_argument("--k", type=int, default=5,
                    help="Candidates for self_consistency; samples/node for beam.")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--beam-depth", type=int, default=2)
    ap.add_argument("--output-suffix", default="planning")
    ap.add_argument("--output-root", default="results_v4",
                    help="Root directory for newly created output folders.")
    ap.add_argument("--resume-from", default=None,
                    help="Existing output directory. Completed raw episodes are reused.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.models = expand_model_selection(args.models, args.model_panel)

    seeds = parse_seeds(args.seeds)
    controller_temperatures = parse_controller_temperatures(args.controller_temperatures)
    if args.temperature is not None:
        controller_temperatures = {k: args.temperature for k in controller_temperatures}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.resume_from) if args.resume_from else Path(
        args.output_root
    ) / f"planning_wrappers_{args.output_suffix}_{timestamp}"

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
                for controller in args.controllers:
                    calls_per_action = controller_call_budget(
                        controller,
                        k=args.k,
                        beam_width=args.beam_width,
                        beam_depth=args.beam_depth,
                    )
                    max_calls_per_episode = horizon * calls_per_action
                    plan.append((
                        model_id,
                        env_name,
                        prompt_name,
                        controller,
                        len(seeds),
                        horizon,
                        calls_per_action,
                        max_calls_per_episode,
                    ))

    total = sum(row[4] for row in plan)
    max_planned_model_calls = sum(row[4] * row[7] for row in plan)
    print("=== Planning Wrapper Experiment ===")
    print(f"Models:      {args.models}")
    print(f"Envs:        {args.envs}")
    print(f"Prompts:     {args.prompts}")
    print(f"Controllers: {args.controllers}")
    print(f"Temperatures:{controller_temperatures}")
    print(f"Memory:      {args.memory}")
    print(f"Seeds:       {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Total eps:   {total}")
    print(f"Max calls:   {max_planned_model_calls} (dry-run upper bound)")
    print(f"Output:      {output_dir}")
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
    model_cache = {}
    for model_id in args.models:
        for controller_name in args.controllers:
            temp = controller_temperatures.get(controller_name, 0.7)
            model_cache[(model_id, controller_name)] = APIModel(
                model_id=model_id,
                base_url=args.base_url,
                api_key=api_key,
                temperature=temp,
                max_tokens=args.max_tokens,
                timeout=args.api_timeout,
                max_retries=args.max_retries,
            )

    summaries = []
    episodes_by_cell = defaultdict(list)
    started = time.time()
    done = 0

    for (
        model_id,
        env_name,
        prompt_name,
        controller_name,
        _,
        horizon,
        calls_per_action,
        max_calls_per_episode,
    ) in plan:
        model = model_cache[(model_id, controller_name)]
        controller = get_controller(
            controller_name,
            k=args.k,
            width=args.beam_width,
            depth=args.beam_depth,
            samples_per_node=args.k,
            scorer=args.scorer,
        )
        for seed in seeds:
            resume_path = raw_episode_path(
                output_dir,
                env_name=env_name,
                prompt_name=prompt_name,
                memory_name=args.memory,
                regime=f"constructive_editing:{controller_name}",
                model_id=model_id.split("/")[-1],
                seed=seed,
            )
            summary = load_raw_episode(resume_path)
            resumed = summary is not None
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
                    model=model,
                    controller=controller,
                    seed=seed,
                    max_episode_steps=args.max_episode_steps,
                )
                logger.log_episode(summary)
            key = (model_id, env_name, prompt_name, args.memory, controller_name)
            episodes_by_cell[key].append(summary)
            done += 1
            status = "OK" if summary.success else "FAIL"
            print(
                f"[{done}/{total}] {model_short_name(model_id)}|{env_name}|"
                f"{prompt_name}|{controller_name}|s{seed} -> {status}"
                f"{' (resumed)' if resumed else ''}",
                flush=True,
            )

        cell_rows = episodes_by_cell[(model_id, env_name, prompt_name, args.memory, controller_name)]
        row = summarize_episode_collection(cell_rows)
        row.update({
            "model": model_id,
            "env": env_name,
            "prompt": prompt_name,
            "memory": args.memory,
            "controller": controller_name,
            "temperature": controller_temperatures.get(controller_name, 0.7),
            "scorer": args.scorer if controller_name not in {"greedy", "greedy_sampled"} else None,
            "planned_horizon": horizon,
            "planned_calls_per_action": calls_per_action,
            "planned_max_calls_per_episode": max_calls_per_episode,
        })
        attach_model_metadata(row)
        summaries.append(row)
        write_json(output_dir / "planning_wrapper_summary.json", {
            "config": vars(args),
            "cost_plan": {
                "total_episodes_planned": total,
                "max_planned_model_calls": max_planned_model_calls,
                "call_budget_is_upper_bound": True,
            },
            "results": summaries,
            "total_episodes_so_far": done,
            "total_episodes_planned": total,
            "complete": False,
        })

    wall = time.time() - started
    write_json(output_dir / "planning_wrapper_summary.json", {
        "config": vars(args),
        "cost_plan": {
            "total_episodes_planned": total,
            "max_planned_model_calls": max_planned_model_calls,
            "call_budget_is_upper_bound": True,
        },
        "results": summaries,
        "total_episodes": done,
        "total_episodes_planned": total,
        "wall_time_s": wall,
        "complete": True,
    })

    print("\nSummary:")
    for row in summaries:
        print(
            f"{model_short_name(row['model'])}|{row['env']}|{row['prompt']}|"
            f"{row['controller']}: SR={row['SR']:.0%} "
            f"LocalValid={row['local_valid_edit_rate'] if row['local_valid_edit_rate'] is not None else 'n/a'} "
            f"Calls/ep={row['avg_model_calls']} Tokens/ep={row['avg_tokens']} "
            f"T={row['temperature']}"
        )
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
