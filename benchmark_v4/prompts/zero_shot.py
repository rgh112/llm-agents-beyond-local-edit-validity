"""Zero-shot prompt condition: minimal instruction, no examples, no strategic hints."""
from benchmark_v4.prompts.base_prompt import BasePromptBuilder


class ZeroShotPrompt(BasePromptBuilder):
    def __init__(self):
        super().__init__("zero_shot", metadata={
            "family": "zero_shot",
            "strategy_bearing": False,
            "task_specific": False,
            "uses_examples": False,
        })

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        return (
            f"You are solving a constructive editing task: {env_name}.\n\n"
            f"Task: {task_description}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Action format:\n{action_format}\n\n"
            f"Respond with ONLY your action in the specified format. "
            f"Do not include any explanation."
        )

    def build_step_prompt(self, observation, memory_context, step, budget, env_name):
        prompt = f"Step {step}.\n\n"
        if memory_context:
            prompt += f"{memory_context}\n\n"
        prompt += f"Current state:\n{observation}\n\nYour action:"
        return prompt
