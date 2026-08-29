"""Sweep runner - full experimental matrix sweep."""
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmark_v4.prompts import get_prompt_builder
from benchmark_v4.memory import get_memory_module
from benchmark_v4.models.api_model import APIModel
from benchmark_v4.failure_taxonomy import EpisodeSummary
from benchmark_v4.logging_utils.episode_logger import EpisodeLogger
from benchmark_v4.runners.batch_runner import run_batch


# Environment factories
def _create_env(env_name: str, max_steps: Optional[int] = None, **kwargs):
    """Create environment by name."""
    if env_name == "word_ladder":
        from benchmark_v4.envs.word_ladder import WordLadderEnv
        data_dir = kwargs.get("data_dir", "wordladder_data")
        return WordLadderEnv(data_dir=data_dir)
    elif env_name == "alloy":
        from benchmark_v4.envs.alloy import AlloyEnv
        csv_path = kwargs.get("data_path", kwargs.get("csv_path", "alloy_data/mpea_mech.csv"))
        return AlloyEnv(csv_path=csv_path, max_steps=max_steps or 6)
    elif env_name == "protein":
        from benchmark_v4.envs.protein import ProteinEnv
        return ProteinEnv(max_steps=max_steps or 8, **kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def run_sweep(config: Dict, verbose: bool = True) -> List[EpisodeSummary]:
    """Run full experimental sweep from config dict.

    Config format:
    {
        "environments": ["word_ladder", "alloy", ...],
        "prompt_conditions": ["zero_shot", "few_shot", "scaffold", "self_check"],
        "memory_conditions": ["state_only", "window_1", "window_3", "window_5"],
        "models": [
            {"model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "base_url": null},
            ...
        ],
        "n_episodes": 20,
        "seeds": [0, 1, 2, ..., 19],  # or auto-generated
        "output_dir": "results_v4",
        "env_kwargs": {
            "word_ladder": {"data_dir": "wordladder_data"},
            "alloy": {"csv_path": "alloy_data/mpea_mech.csv"},
        }
    }
    """
    envs = config["environments"]
    prompts = config["prompt_conditions"]
    memories = config["memory_conditions"]
    model_configs = config["models"]
    n_episodes = config.get("n_episodes", 20)
    seeds = config.get("seeds", list(range(n_episodes)))
    output_dir = config.get("output_dir", "results_v4")
    env_kwargs = config.get("env_kwargs", {})

    all_summaries = []
    total_combos = len(envs) * len(prompts) * len(memories) * len(model_configs)
    combo_idx = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = str(Path(output_dir) / f"sweep_{timestamp}")

    for env_name in envs:
        kwargs = env_kwargs.get(env_name, {})
        env = _create_env(env_name, **kwargs)

        for prompt_name in prompts:
            prompt_builder = get_prompt_builder(prompt_name)

            for memory_name in memories:
                memory_module = get_memory_module(memory_name)

                for model_cfg in model_configs:
                    combo_idx += 1
                    model = APIModel(
                        model_id=model_cfg["model_id"],
                        base_url=model_cfg.get("base_url"),
                        api_key=model_cfg.get("api_key"),
                    )

                    if verbose:
                        print(f"\n=== Combo {combo_idx}/{total_combos} ===")
                        print(f"  Env: {env_name} | Prompt: {prompt_name} | "
                              f"Memory: {memory_name} | Model: {model.get_model_name()}")

                    summaries = run_batch(
                        env=env,
                        prompt_builder=prompt_builder,
                        memory_module=memory_module,
                        model=model,
                        seeds=seeds,
                        output_dir=sweep_dir,
                        verbose=verbose,
                    )
                    all_summaries.extend(summaries)

                    # Quick summary
                    sr = sum(1 for s in summaries if s.success) / len(summaries)
                    if verbose:
                        print(f"  -> SR: {sr:.1%} ({len(summaries)} episodes)")

    # Save full sweep summary
    sweep_logger = EpisodeLogger(sweep_dir)
    sweep_logger.log_batch_summary(all_summaries, config)

    if verbose:
        print(f"\nSweep complete. {len(all_summaries)} episodes saved to {sweep_dir}")

    return all_summaries
