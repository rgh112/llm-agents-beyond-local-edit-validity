"""Batch runner - multiple episodes for one condition combination."""
from typing import List, Optional

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv
from benchmark_v4.prompts.base_prompt import BasePromptBuilder
from benchmark_v4.memory.base_memory import BaseMemory
from benchmark_v4.models.base_model import BaseModel
from benchmark_v4.failure_taxonomy import EpisodeSummary
from benchmark_v4.logging_utils.episode_logger import EpisodeLogger
from benchmark_v4.runners.single_episode_runner import run_episode

def run_batch(
    env: BaseConstructiveEditEnv,
    prompt_builder: BasePromptBuilder,
    memory_module: BaseMemory,
    model: BaseModel,
    seeds: List[int],
    pair_indices: Optional[List[int]] = None,
    output_dir: str = "results_v4",
    verbose: bool = False,
) -> List[EpisodeSummary]:
    """Run multiple episodes and log results.

    Args:
        env: Environment instance
        prompt_builder: Prompt condition
        memory_module: Memory condition
        model: LLM model
        seeds: List of random seeds (one per episode)
        pair_indices: Optional list of pair indices (for word ladder etc.)
        output_dir: Directory for output logs
        verbose: Print progress
    """
    logger = EpisodeLogger(output_dir)
    summaries = []

    n = len(seeds)
    for i, seed in enumerate(seeds):
        pair_idx = pair_indices[i] if pair_indices and i < len(pair_indices) else None

        if verbose:
            print(f"[{i+1}/{n}] {env.env_name} | {prompt_builder.condition_name} | "
                  f"{memory_module.condition_name} | seed={seed}")

        summary = run_episode(
            env=env,
            prompt_builder=prompt_builder,
            memory_module=memory_module,
            model=model,
            seed=seed,
            pair_index=pair_idx,
            verbose=verbose,
        )

        logger.log_episode(summary)
        summaries.append(summary)

        if verbose:
            status = "SUCCESS" if summary.success else f"FAIL ({summary.terminal_failure})"
            print(f"  -> {status} in {len(summary.steps)} steps ({summary.wall_time_s:.1f}s)")

    # Log batch summary
    config = {
        "env_name": env.env_name,
        "regime": "constructive_editing",
        "prompt_condition": prompt_builder.condition_name,
        "memory_condition": memory_module.condition_name,
        "model": model.get_model_name(),
        "n_episodes": n,
        "seeds": seeds,
    }
    logger.log_batch_summary(summaries, config)

    return summaries
