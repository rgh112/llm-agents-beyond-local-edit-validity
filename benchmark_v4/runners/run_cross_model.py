#!/usr/bin/env python3
"""Cross-model sanity check (rebuttal Package R1).

Goal: test whether the qualitative patterns observed for Qwen2.5-7B-Instruct-Turbo
recur across other model families. Not a leaderboard.

Default sweep:
    3 envs × 3 prompts (zero_shot, few_shot_format, env-specific structural) × M0
    × N additional models × N seeds

Per-env structural prompt:
    word_ladder   -> scaffold
    alloy         -> self_check
    gb1_sequence  -> self_check

Default API: OpenRouter (https://openrouter.ai/api/v1).
Set OPENROUTER_API_KEY in env. Falls back to TOGETHER_API_KEY/OPENAI_API_KEY only
if --base-url is overridden.

Example:
    OPENROUTER_API_KEY=... python -m benchmark_v4.runners.run_cross_model \
        --models meta-llama/llama-3.1-8b-instruct openai/gpt-4o-mini \
        --seeds 0-49 \
        --output-suffix r1
"""
import argparse
import json
import os
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.logging_utils.episode_logger import EpisodeLogger
from benchmark_v4.memory import get_memory_module
from benchmark_v4.models.api_model import APIModel
from benchmark_v4.prompts import get_prompt_builder
from benchmark_v4.runners.experiment_utils import load_raw_episode, make_env, raw_episode_path
from benchmark_v4.runners.single_episode_runner import run_episode
from benchmark_v4.model_registry import get_model_metadata
from benchmark_v4.model_panels import available_panels, expand_model_selection

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

DEFAULT_PROMPTS = ["zero_shot", "few_shot_format", "__structural__"]


def parse_seeds(spec: str):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def model_short_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Model IDs to evaluate (e.g. meta-llama/llama-3.1-8b-instruct).")
    ap.add_argument("--model-panel", default=None, choices=available_panels(),
                    help="Pre-registered model panel. Can be combined with --models.")
    ap.add_argument("--seeds", default="0-49",
                    help="Seed range like '0-49' or comma list '0,1,5'. Default: 0-49.")
    ap.add_argument("--envs", nargs="+",
                    default=["word_ladder", "alloy", "gb1_sequence"])
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="Use '__structural__' to expand to env-specific structural prompt.")
    ap.add_argument("--memory", default="state_only")
    ap.add_argument("--gb1-fitness-target", type=float,
                    default=float(os.environ.get("GB1_FITNESS_TARGET", "5.0")))
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key", default=None,
                    help="Overrides OPENROUTER_API_KEY/TOGETHER_API_KEY env vars.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None,
                    help="Optional nucleus-sampling top_p value.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="Optional smoke-test cap applied after env construction.")
    ap.add_argument("--output-suffix", default="",
                    help="Optional suffix appended to output dir name.")
    ap.add_argument("--output-root", default="results_v4",
                    help="Root directory for newly created output folders.")
    ap.add_argument("--resume-from", default=None,
                    help="Existing output directory. Completed raw episodes are reused.")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Concurrent episodes per (model, env, prompt) cell. "
                         "Default 1 (sequential). 8 is a reasonable OpenRouter setting.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit without API calls.")
    args = ap.parse_args()
    args.models = expand_model_selection(args.models, args.model_panel)

    seeds = parse_seeds(args.seeds)
    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("TOGETHER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key and not args.dry_run:
        print("ERROR: no API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_dir = args.resume_from or str(Path(args.output_root) / f"cross_model{suffix}_{timestamp}")

    total = 0
    max_planned_model_calls = 0
    plan = []
    for model_id in args.models:
        for env_name in args.envs:
            horizon = ENV_DEFAULT_HORIZON.get(env_name)
            if horizon is None:
                horizon = make_env(env_name).max_steps
            if args.max_episode_steps is not None:
                horizon = min(horizon, args.max_episode_steps)
            for prompt_spec in args.prompts:
                prompt_name = (
                    ENV_STRUCTURAL_PROMPT[env_name]
                    if prompt_spec == "__structural__"
                    else prompt_spec
                )
                plan.append((model_id, env_name, prompt_name, len(seeds), horizon))
                total += len(seeds)
                max_planned_model_calls += len(seeds) * horizon

    print("=== Cross-Model Sanity Check (Rebuttal Package R1) ===")
    print(f"  Models:  {args.models}")
    print(f"  Envs:    {args.envs}")
    print(f"  Prompts: {args.prompts}")
    print(f"  Memory:  {args.memory}")
    print(f"  Decoding: temperature={args.temperature}, top_p={args.top_p}")
    print(f"  Seeds:   {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"  Total:   {total} episodes")
    print(f"  Max calls: {max_planned_model_calls} (dry-run upper bound)")
    print(f"  Base URL: {args.base_url}")
    print(f"  Output:  {output_dir}")
    print()

    if args.dry_run:
        print("Dry run — exiting without API calls.")
        for row in plan:
            print(f"  {row}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(output_dir)
    log_lock = threading.Lock()
    print_lock = threading.Lock()
    results_table = []
    combo_counter = {"n": 0}
    sweep_start = time.time()

    def run_one(model, env_name, prompt_name, seed):
        resume_path = raw_episode_path(
            output_dir,
            env_name=env_name,
            prompt_name=prompt_name,
            memory_name=args.memory,
            regime="constructive_editing",
            model_id=model.get_model_name(),
            seed=seed,
        )
        resumed = load_raw_episode(resume_path)
        if resumed is not None:
            return ("resumed", seed, resumed)
        # Build per-call instances (env, memory) so threads don't share state.
        if env_name == "gb1_sequence":
            env = make_env(env_name, fitness_target=args.gb1_fitness_target)
        else:
            env = make_env(env_name)
        if args.max_episode_steps is not None:
            env.max_steps = min(env.max_steps, args.max_episode_steps)
        prompt = get_prompt_builder(prompt_name)
        memory = get_memory_module(args.memory)
        try:
            summary = run_episode(
                env=env, prompt_builder=prompt, memory_module=memory,
                model=model, seed=seed, verbose=False,
                max_episode_steps=args.max_episode_steps,
            )
            with log_lock:
                logger.log_episode(summary)
            return ("ok", seed, summary)
        except Exception as e:
            return ("error", seed, e)

    for model_id in args.models:
        model = APIModel(
            model_id=model_id,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.api_timeout,
            max_retries=args.max_retries,
        )

        for env_name in args.envs:
            for prompt_spec in args.prompts:
                prompt_name = (
                    ENV_STRUCTURAL_PROMPT[env_name]
                    if prompt_spec == "__structural__"
                    else prompt_spec
                )

                successes = 0
                steps_total = 0
                failures = Counter()
                events = Counter()
                finalize_steps = []
                first_failures = []
                surface_clean_episodes = 0
                surface_clean_successes = 0
                local_valid_edits = 0
                total_edits = 0
                total_tokens = 0

                with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                    futures = {
                        ex.submit(run_one, model, env_name, prompt_name, seed): seed
                        for seed in seeds
                    }
                    for fut in as_completed(futures):
                        seed = futures[fut]
                        with print_lock:
                            combo_counter["n"] += 1
                            combo_count = combo_counter["n"]
                        status, _, payload = fut.result()
                        if status == "error":
                            with print_lock:
                                print(f"  [{combo_count}/{total}] {model_short_name(model_id)}|"
                                      f"{env_name}|{prompt_name}|s{seed} -> ERROR: {payload}")
                            continue
                        summary = payload

                        if summary.success:
                            successes += 1
                        steps_total += len(summary.steps)
                        total_tokens += int(summary.token_usage.get("total_tokens", 0))
                        if summary.terminal_failure:
                            failures[summary.terminal_failure] += 1
                        for ename, cnt in summary.event_counts.items():
                            events[ename] += cnt
                        if summary.finalize_step is not None:
                            finalize_steps.append(summary.finalize_step)
                        if summary.first_failure_step is not None:
                            first_failures.append(summary.first_failure_step)

                        ep_surface_event = False
                        for step in summary.steps:
                            if step.parsed_action and step.parsed_action.get("type") == "edit":
                                total_edits += 1
                                if step.valid:
                                    local_valid_edits += 1
                            for ev in step.events:
                                ename = ev.name if hasattr(ev, "name") else str(ev)
                                if ename in SURFACE_EVENTS:
                                    ep_surface_event = True
                                    break
                        if not ep_surface_event:
                            surface_clean_episodes += 1
                            if summary.success:
                                surface_clean_successes += 1

                        status_str = "OK" if summary.success else "FAIL"
                        with print_lock:
                            resume_note = " (resumed)" if status == "resumed" else ""
                            print(f"  [{combo_count}/{total}] {model_short_name(model_id)}|"
                                  f"{env_name}|{prompt_name}|s{seed} -> {status_str}{resume_note} "
                                  f"({len(summary.steps)}steps, {summary.wall_time_s:.1f}s)",
                                  flush=True)

                n = len(seeds)
                sr = successes / n if n else 0.0
                avg_steps = steps_total / n if n else 0.0
                surface_count = sum(events.get(e, 0) for e in SURFACE_EVENTS)
                structural_count = sum(events.get(e, 0) for e in STRUCTURAL_EVENTS)
                local_valid_rate = (
                    local_valid_edits / total_edits if total_edits else None
                )
                clean_sr = (
                    surface_clean_successes / surface_clean_episodes
                    if surface_clean_episodes else None
                )
                avg_model_calls = steps_total / n if n else None
                avg_tokens = total_tokens / n if n else None

                results_table.append({
                    "model": model_id,
                    **{f"model_{k}": v for k, v in get_model_metadata(model_id).items()},
                    "env": env_name,
                    "prompt": prompt_name,
                    "memory": args.memory,
                    "SR": sr,
                    "n_success": successes,
                    "n_total": n,
                    "avg_steps": round(avg_steps, 1),
                    "terminal_failures": dict(failures),
                    "all_events": dict(events),
                    "surface_failure_count": surface_count,
                    "structural_failure_count": structural_count,
                    "local_valid_edit_rate": local_valid_rate,
                    "n_edit_attempts": total_edits,
                    "n_local_valid_edits": local_valid_edits,
                    "n_surface_clean_episodes": surface_clean_episodes,
                    "n_surface_clean_successes": surface_clean_successes,
                    "SR_given_surface_clean": clean_sr,
                    "avg_model_calls": avg_model_calls,
                    "avg_tokens": avg_tokens,
                    "SR_per_100_calls": (
                        sr / avg_model_calls * 100.0
                        if avg_model_calls and avg_model_calls > 0 else None
                    ),
                    "SR_per_100k_tokens": (
                        sr / avg_tokens * 100000.0
                        if avg_tokens and avg_tokens > 0 else None
                    ),
                    "avg_finalize_step": round(sum(finalize_steps) / len(finalize_steps), 1)
                        if finalize_steps else None,
                    "avg_first_failure": round(sum(first_failures) / len(first_failures), 1)
                        if first_failures else None,
                    "prompt_metadata": get_prompt_builder(prompt_name).metadata,
                })

                lv_str = f"{local_valid_rate:.0%}" if local_valid_rate is not None else "n/a"
                print(f"  >> {model_short_name(model_id)}|{env_name}|{prompt_name}: "
                      f"SR={sr:.0%} ({successes}/{n}) local_valid={lv_str}", flush=True)
                # Persist intermediate summary so user can monitor progress.
                with open(Path(output_dir) / "cross_model_summary.json", "w") as f:
                    json.dump({
                        "config": {
                            "models": args.models,
                            "model_panel": args.model_panel,
                            "seeds": seeds,
                            "memory": args.memory,
                            "prompts": args.prompts,
                            "envs": args.envs,
                            "base_url": args.base_url,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens,
                            "gb1_fitness_target": args.gb1_fitness_target,
                            "concurrency": args.concurrency,
                        "resume_from": args.resume_from,
                    },
                    "cost_plan": {
                        "total_episodes_planned": total,
                        "max_planned_model_calls": max_planned_model_calls,
                        "call_budget_is_upper_bound": True,
                    },
                    "results": results_table,
                        "total_episodes_so_far": combo_counter["n"],
                        "total_episodes_planned": total,
                        "wall_time_s_so_far": time.time() - sweep_start,
                        "timestamp": timestamp,
                        "complete": False,
                    }, f, indent=2, default=str)

    sweep_time = time.time() - sweep_start

    print()
    print("=" * 90)
    print(f"CROSS-MODEL SANITY CHECK COMPLETE: {combo_count} episodes in {sweep_time:.0f}s")
    print("=" * 90)

    for model_id in args.models:
        for env_name in args.envs:
            print(f"\n{model_short_name(model_id)} / {env_name}:")
            print(f"  {'Prompt':<22} {'SR':>8} {'LocalValid':>11} {'Surf':>5} "
                  f"{'Struct':>6}  Top Failures")
            print("  " + "-" * 90)
            for r in results_table:
                if r["model"] != model_id or r["env"] != env_name:
                    continue
                top_fail = ", ".join(
                    f"{k}:{v}" for k, v in
                    sorted(r["terminal_failures"].items(), key=lambda x: -x[1])[:3]
                )
                lv = r["local_valid_edit_rate"]
                lv_str = f"{lv:.0%}" if lv is not None else "-"
                print(f"  {r['prompt']:<22} {r['n_success']:>2}/{r['n_total']:<3} "
                      f"({r['SR']:>4.0%}) {lv_str:>11} "
                      f"{r['surface_failure_count']:>5} {r['structural_failure_count']:>6}  {top_fail}")

    summary_path = Path(output_dir) / "cross_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "models": args.models,
                "model_panel": args.model_panel,
                "seeds": seeds,
                "memory": args.memory,
                "prompts": args.prompts,
                "envs": args.envs,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "gb1_fitness_target": args.gb1_fitness_target,
                "concurrency": args.concurrency,
                "resume_from": args.resume_from,
            },
            "cost_plan": {
                "total_episodes_planned": total,
                "max_planned_model_calls": max_planned_model_calls,
                "call_budget_is_upper_bound": True,
            },
            "results": results_table,
            "total_episodes": combo_counter["n"],
            "total_episodes_planned": total,
            "wall_time_s": sweep_time,
            "timestamp": timestamp,
            "complete": True,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
