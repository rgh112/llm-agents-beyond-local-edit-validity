"""Prompt builders for constructive editing experiments."""

from benchmark_v4.prompts.zero_shot import ZeroShotPrompt
from benchmark_v4.prompts.few_shot_format import FewShotFormatPrompt
from benchmark_v4.prompts.few_shot import FewShotPrompt
from benchmark_v4.prompts.few_shot_strategy import FewShotStrategyPrompt
from benchmark_v4.prompts.scaffold import ScaffoldPrompt
from benchmark_v4.prompts.self_check import SelfCheckPrompt


PROMPT_BUILDERS = {
    "zero_shot": ZeroShotPrompt,
    "few_shot_format": FewShotFormatPrompt,
    "few_shot": FewShotPrompt,
    "few_shot_strategy": FewShotStrategyPrompt,
    "scaffold": ScaffoldPrompt,
    "self_check": SelfCheckPrompt,
}


def get_prompt_builder(name: str):
    if name not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown prompt builder {name!r}; choose from {sorted(PROMPT_BUILDERS)}"
        )
    return PROMPT_BUILDERS[name]()


__all__ = [
    "ZeroShotPrompt",
    "FewShotFormatPrompt",
    "FewShotPrompt",
    "FewShotStrategyPrompt",
    "ScaffoldPrompt",
    "SelfCheckPrompt",
    "get_prompt_builder",
]
