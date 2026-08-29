"""Scaffold prompt condition: step-by-step planning checklist before each action.

Generic checks apply to all environments. Environment-specific checks add
domain-aware planning hints without giving explicit strategy.
"""
from benchmark_v4.prompts.base_prompt import BasePromptBuilder


GENERIC_SCAFFOLD = (
    "Before choosing your edit, consider:\n"
    "1. LEGALITY: Is this edit in valid format and within constraints?\n"
    "2. PROGRESS: Does this edit move the structure toward the final target?\n"
    "3. FUTURE OPTIONS: Could this edit reduce future editing flexibility?\n"
    "4. TIMING: Is it too early or too late to FINALIZE?"
)

# NOTE: These are domain-aware planning hints, not explicit strategies.
# They help the model think about the right dimensions without telling it what to do.
ENV_SCAFFOLD = {
    "word_ladder": (
        "5. PATH: Will the resulting word leave enough valid neighbors to continue?\n"
        "6. DISTANCE: Does this edit reduce the number of mismatched positions?"
    ),
    "alloy": (
        "5. BALANCE: Are you considering BOTH UTS and density simultaneously?\n"
        "6. PHASE: Are you using the right delta size for the current phase?"
    ),
    "gb1_sequence": (
        "5. EPISTASIS: Could this position interact with others in unexpected ways?\n"
        "6. ESTIMATES: Are you relying too heavily on single-site estimates?"
    ),
}


class ScaffoldPrompt(BasePromptBuilder):
    def __init__(self):
        super().__init__("scaffold", metadata={
            "family": "scaffold",
            "strategy_bearing": False,
            "task_specific": True,
            "uses_examples": False,
        })

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        return (
            f"You are solving a constructive editing task: {env_name}.\n\n"
            f"Task: {task_description}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Action format:\n{action_format}\n\n"
            f"Consider the planning checks below internally, "
            f"then respond with ONLY your action."
        )

    def build_step_prompt(self, observation, memory_context, step, budget, env_name):
        prompt = f"Step {step}.\n\n"
        if memory_context:
            prompt += f"{memory_context}\n\n"
        prompt += f"Current state:\n{observation}\n\n"

        scaffold = GENERIC_SCAFFOLD
        env_extra = ENV_SCAFFOLD.get(env_name, "")
        if env_extra:
            scaffold += "\n" + env_extra

        prompt += f"{scaffold}\n\nConsider these checks internally, then respond with ONLY your action.\n\nYour action:"
        return prompt
