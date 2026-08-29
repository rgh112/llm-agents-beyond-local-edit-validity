"""Self-check prompt condition: verification rules before each action.

Two modes:
  - generic: legality, budget, stopping checks only
  - task_specific: adds domain-specific behavioral rules that shape editing policy

NOTE: Task-specific rules are behavioral interventions, not generic output checks.
They encode domain knowledge about what errors to watch for.
"""
from benchmark_v4.prompts.base_prompt import BasePromptBuilder


GENERIC_RULES = [
    "Verify your action uses the correct format.",
    "Check that you are not repeating a previous failed action.",
    "Consider whether you have enough budget remaining.",
    "Do not FINALIZE unless you believe the target is met.",
]

# NOTE: These are task-specific behavioral rules, not generic output checks.
# They shape the model's editing policy by highlighting domain-specific risks.
TASK_SPECIFIC_RULES = {
    "word_ladder": [
        "Verify the resulting word is a valid English word.",
        "Check that the resulting word has not been used before in the path.",
        "Consider whether this edit moves you closer to the target.",
    ],
    "alloy": [
        "Check that both UTS and density targets are considered together.",
        "Avoid edits that previously worsened performance.",
        "Consider whether this edit increases recovery cost.",
    ],
    "gb1_sequence": [
        "Single-site estimates may be WRONG due to epistasis — do not trust them blindly.",
        "Consider editing positions even if their estimated effect looks negative.",
        "Check stability before destabilizing mutations — they may gate future options.",
    ],
}


class SelfCheckPrompt(BasePromptBuilder):
    """Task-specific self-check (generic + domain rules)."""
    def __init__(self):
        super().__init__("self_check", metadata={
            "family": "self_check",
            "strategy_bearing": True,
            "task_specific": True,
            "uses_examples": False,
        })

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        rules = GENERIC_RULES + TASK_SPECIFIC_RULES.get(env_name, [])
        rules_text = "\n".join(f"- {r}" for r in rules)

        return (
            f"You are solving a constructive editing task: {env_name}.\n\n"
            f"Task: {task_description}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Action format:\n{action_format}\n\n"
            f"IMPORTANT — Before each action, verify:\n{rules_text}\n\n"
            f"Respond with ONLY your action in the specified format."
        )

    def build_step_prompt(self, observation, memory_context, step, budget, env_name):
        prompt = f"Step {step}.\n\n"
        if memory_context:
            prompt += f"{memory_context}\n\n"
        prompt += f"Current state:\n{observation}\n\n"
        prompt += "Verify your action before responding.\n\nYour action:"
        return prompt


class SelfCheckGenericPrompt(BasePromptBuilder):
    """Generic self-check only (no domain-specific rules)."""
    def __init__(self):
        super().__init__("self_check_generic", metadata={
            "family": "self_check",
            "strategy_bearing": False,
            "task_specific": False,
            "uses_examples": False,
        })

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        rules_text = "\n".join(f"- {r}" for r in GENERIC_RULES)

        return (
            f"You are solving a constructive editing task: {env_name}.\n\n"
            f"Task: {task_description}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Action format:\n{action_format}\n\n"
            f"IMPORTANT — Before each action, verify:\n{rules_text}\n\n"
            f"Respond with ONLY your action in the specified format."
        )

    def build_step_prompt(self, observation, memory_context, step, budget, env_name):
        prompt = f"Step {step}.\n\n"
        if memory_context:
            prompt += f"{memory_context}\n\n"
        prompt += f"Current state:\n{observation}\n\n"
        prompt += "Verify your action before responding.\n\nYour action:"
        return prompt
