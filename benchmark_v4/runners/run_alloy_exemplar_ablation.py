#!/usr/bin/env python3
"""Package D: Alloy exemplar robustness ablation.

Tests whether Alloy few-shot collapse is exemplar artifact or strategy-prior problem.
2 exemplar variants × 2 controls × 30 seeds = 120 episodes.
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.alloy import AlloyEnv
from benchmark_v4.prompts import get_prompt_builder
from benchmark_v4.prompts.few_shot_strategy import FewShotStrategyPrompt
from benchmark_v4.memory import get_memory_module
from benchmark_v4.models.api_model import APIModel
from benchmark_v4.runners.single_episode_runner import run_episode
from benchmark_v4.logging_utils.episode_logger import EpisodeLogger

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"
BASE_URL = "https://api.together.xyz/v1"
API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY before running hosted experiments.")

SEEDS = list(range(30))
MEMORY = "state_only"

# ── Exemplar variant B: different numbers, same strategic direction ──
# Original (A): coarse EDIT Co Fe 10, fine EDIT Cr Ni 2
# Variant B: coarse EDIT Ni Mn 15, fine EDIT Mo Co 5
VARIANT_B_EXEMPLARS = {
    "alloy": [
        {
            "observation_snippet": (
                "Composition: Fe:30% Ni:10% Cr:20% Co:15% Mn:15% Mo:10%\n"
                "UTS: ~1750 MPa  Density: 8.05\n"
                "Phase: coarse  Allowed deltas: {10, 15}"
            ),
            "reasoning": (
                "UTS is below target. Ni is heavy and helps UTS. "
                "Mn is light but less useful for UTS. "
                "Transfer from Mn to Ni in coarse phase."
            ),
            "action": "EDIT Ni Mn 15",
        },
        {
            "observation_snippet": (
                "Composition: Fe:30% Ni:25% Cr:20% Co:15% Mn:0% Mo:10%\n"
                "UTS: ~2050 MPa  Density: 8.35\n"
                "Phase: fine  Allowed deltas: {2, 5}"
            ),
            "reasoning": (
                "UTS meets target but density is slightly over. "
                "Reduce heavy Mo, increase lighter Co slightly. "
                "Fine-tune in small steps."
            ),
            "action": "EDIT Co Mo 5",
        },
    ],
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/alloy_exemplar_{timestamp}"

model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

# Build variant B prompt by subclassing
class FewShotStrategyVariantB(FewShotStrategyPrompt):
    """Variant B exemplars for Alloy — same strategy direction, different numbers."""
    def __init__(self):
        super().__init__()
        self.condition_name = "few_shot_strategy_B"

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        if env_name == "alloy":
            examples = VARIANT_B_EXEMPLARS["alloy"]
        else:
            from benchmark_v4.prompts.few_shot_strategy import STRATEGY_EXEMPLARS
            examples = STRATEGY_EXEMPLARS.get(env_name, [])

        example_text = ""
        for i, ex in enumerate(examples, 1):
            obs_key = "observation" if "observation" in ex else "observation_snippet"
            example_text += (
                f"\nExample {i}:\n"
                f"Observation:\n{ex[obs_key]}\n"
                f"Action: {ex['action']}\n"
            )

        return (
            f"You are solving a constructive editing task: {env_name}.\n\n"
            f"Task: {task_description}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Action format:\n{action_format}\n\n"
            f"Examples:{example_text}\n"
            f"Respond with ONLY your action. Do not include any explanation."
        )


CONDITIONS = {
    "zero_shot": get_prompt_builder("zero_shot"),
    "few_shot_format": get_prompt_builder("few_shot_format"),
    "few_shot_strategy_A": get_prompt_builder("few_shot_strategy"),
    "few_shot_strategy_B": FewShotStrategyVariantB(),
}

results_table = []
combo_count = 0
total = len(CONDITIONS) * len(SEEDS)

print(f"=== Package D: Alloy Exemplar Robustness ===")
print(f"  {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {total} episodes")
print(f"  Output: {OUTPUT_DIR}")
print()

sweep_start = time.time()

for cond_name, prompt in CONDITIONS.items():
    memory = get_memory_module(MEMORY)
    successes = 0
    failures = Counter()

    for seed in SEEDS:
        combo_count += 1
        env = AlloyEnv()

        try:
            summary = run_episode(
                env=env, prompt_builder=prompt, memory_module=memory,
                model=model, seed=seed, verbose=False,
            )
            logger.log_episode(summary)
            if summary.success:
                successes += 1
            if summary.terminal_failure:
                failures[summary.terminal_failure] += 1

            status = "OK" if summary.success else "FAIL"
            print(f"  [{combo_count}/{total}] {cond_name}|s{seed} -> {status}")

        except Exception as e:
            print(f"  [{combo_count}/{total}] {cond_name}|s{seed} -> ERROR: {e}")

    sr = successes / len(SEEDS)
    results_table.append({
        "condition": cond_name, "SR": sr,
        "n_success": successes, "n_total": len(SEEDS),
        "terminal_failures": dict(failures),
    })
    print(f"  >> {cond_name}: SR={sr:.0%} ({successes}/{len(SEEDS)})")
    print()

sweep_time = time.time() - sweep_start

print(f"\n{'='*60}")
print(f"ALLOY EXEMPLAR ABLATION COMPLETE in {sweep_time:.0f}s")
print(f"{'='*60}")
for r in results_table:
    print(f"  {r['condition']:<25} SR={r['n_success']}/{r['n_total']} ({r['SR']:.0%})")

# Save
with open(Path(OUTPUT_DIR) / "exemplar_ablation_summary.json", "w") as f:
    json.dump({"results": results_table, "wall_time_s": sweep_time}, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
