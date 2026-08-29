#!/usr/bin/env python3
"""Run construct-validity sensitivity experiments for Alloy and GB1."""
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
    "alloy": "self_check",
    "gb1_sequence": "self_check",
}

ENV_DEFAULT_HORIZON = {
    "alloy": 8,
    "gb1_sequence": 6,
}


def controller_call_budget(controller: str) -> int:
    if controller in {"greedy", "greedy_sampled"}:
        return 1
    if controller in {"self_consistency", "loop_avoidant", "backtracking"}:
        return 5
    if controller == "beam":
        return 12  # default width=3, depth=2, samples_per_node=3
    return 1


SENSITIVITY_GRID = {
    "alloy": {
        "default": {},
        "heavy_scaling_off": {"heavy_floor": 1.0},
        "heavy_scaling_strong": {"heavy_floor": 0.15, "heavy_threshold": 24.0},
        "recovery_off": {"rc_rate": 0.0, "rc_threshold": 1e9},
        "recovery_strict": {"rc_threshold": 2.5},
        "uts_relaxed": {"uts_target": 1800.0},
        "uts_strict": {"uts_target": 2200.0},
        "density_relaxed": {"density_target": 8.4},
        "density_strict": {"density_target": 8.0},
        "noise_off": {"uts_noise_std": 0.0},
        "noise_high": {"uts_noise_std": 60.0},
        "easy_combined": {"heavy_floor": 1.0, "rc_rate": 0.0, "rc_threshold": 1e9},
        "hard_combined": {"heavy_floor": 0.15, "heavy_threshold": 24.0, "rc_threshold": 2.5},
    },
    "gb1_sequence": {
        "default": {},
        "additive_evaluator": {"evaluator_mode": "additive"},
        "threshold_4p7": {"fitness_target": 4.7},
        "threshold_4p8": {"fitness_target": 4.8},
        "threshold_5p2": {"fitness_target": 5.2},
        "noise_off": {"noise_std": 0.0},
        "noise_high": {"noise_std": 0.40},
        "stability_gate_off": {"stability_gate": -1.0},
        "stability_gate_strict": {"stability_gate": 0.70},
        "easy_combined": {"evaluator_mode": "additive", "stability_gate": -1.0},
        "hard_combined": {"fitness_target": 5.2, "noise_std": 0.40, "stability_gate": 0.70},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--model-panel", default=None, choices=available_panels(),
                    help="Pre-registered model panel. Can be combined with --models.")
    ap.add_argument("--envs", nargs="+", default=["alloy", "gb1_sequence"])
    ap.add_argument("--settings", nargs="+", default=["all"],
                    help="Sensitivity settings, or 'all'.")
    ap.add_argument("--prompts", nargs="+", default=["zero_shot", "__structural__"])
    ap.add_argument("--memory", default="state_only")
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
    ap.add_argument("--output-suffix", default="sensitivity")
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
    ) / f"sensitivity_{args.output_suffix}_{timestamp}"

    plan = []
    calls_per_action = controller_call_budget(args.controller)
    max_planned_model_calls = 0
    for env_name in args.envs:
        available = SENSITIVITY_GRID[env_name]
        settings = list(available.keys()) if args.settings == ["all"] else args.settings
        horizon = ENV_DEFAULT_HORIZON.get(env_name)
        if horizon is None:
            horizon = make_env(env_name).max_steps
        if args.max_episode_steps is not None:
            horizon = min(horizon, args.max_episode_steps)
        for setting in settings:
            if setting not in available:
                continue
            for model_id in args.models:
                for prompt in args.prompts:
                    prompt_name = ENV_STRUCTURAL_PROMPT.get(env_name, prompt) if prompt == "__structural__" else prompt
                    plan.append((env_name, setting, available[setting], model_id, prompt_name, len(seeds), horizon))
                    max_planned_model_calls += len(seeds) * horizon * calls_per_action

    total = sum(row[5] for row in plan)
    print("=== Sensitivity Experiment ===")
    print(f"Total:  {total} episodes")
    print(f"Max calls: {max_planned_model_calls} (dry-run upper bound)")
    print(f"Output: {output_dir}")
    if args.dry_run:
        for row in plan:
            print("  ", row[:2], row[3:])
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
    models = {
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

    for env_name, setting, env_kwargs, model_id, prompt_name, _, horizon in plan:
        key = (env_name, setting, model_id, prompt_name)
        for seed in seeds:
            resume_path = raw_episode_path(
                output_dir,
                env_name=env_name,
                prompt_name=prompt_name,
                memory_name=args.memory,
                sensitivity_setting=setting,
                regime=f"constructive_editing:{args.controller}",
                model_id=model_id.split("/")[-1],
                seed=seed,
            )
            summary = load_raw_episode(resume_path)
            resumed = summary is not None
            if summary is None:
                env = make_env(env_name, **env_kwargs)
                if args.max_episode_steps is not None:
                    env.max_steps = min(env.max_steps, args.max_episode_steps)
                summary = run_episode_with_controller(
                    env=env,
                    prompt_builder=get_prompt_builder(prompt_name),
                    memory_module=get_memory_module(args.memory),
                    model=models[model_id],
                    controller=controller,
                    seed=seed,
                    max_episode_steps=args.max_episode_steps,
                )
                summary.final_metrics["sensitivity_setting"] = setting
                summary.final_metrics["sensitivity_kwargs"] = env_kwargs
                logger.log_episode(summary)
            episodes_by_cell[key].append(summary)
            done += 1
            print(
                f"[{done}/{total}] {model_short_name(model_id)}|{env_name}|"
                f"{setting}|{prompt_name}|s{seed} -> {'OK' if summary.success else 'FAIL'}"
                f"{' (resumed)' if resumed else ''}",
                flush=True,
            )

        row = summarize_episode_collection(episodes_by_cell[key])
        row.update({
            "env": env_name,
            "sensitivity_setting": setting,
            "sensitivity_kwargs": env_kwargs,
            "model": model_id,
            "prompt": prompt_name,
            "memory": args.memory,
            "controller": args.controller,
            "planned_horizon": horizon,
            "planned_calls_per_action": calls_per_action,
            "planned_max_calls_per_episode": horizon * calls_per_action,
        })
        attach_model_metadata(row)
        summaries.append(row)
        write_json(output_dir / "sensitivity_summary.json", {
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

    write_json(output_dir / "sensitivity_summary.json", {
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
