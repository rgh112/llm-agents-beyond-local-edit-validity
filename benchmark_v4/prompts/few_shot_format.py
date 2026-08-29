"""Format-only few-shot: shows action grammar and observation reading only.

Exemplars demonstrate correct syntax without encoding any domain strategy.
This isolates the formatting effect of few-shot from strategic prior injection.
"""
from benchmark_v4.prompts.base_prompt import BasePromptBuilder


# Exemplars that show format only — no strategic reasoning, trivial/neutral edits
FORMAT_EXEMPLARS = {
    "word_ladder": [
        {
            "observation": (
                "Target:     warm\n"
                "Current:    cold\n"
                "Positions:  0123\n"
                "Mismatches: ^^^^\n"
                "Path: cold\n"
                "Steps used: 0/20  (remaining: 20)\n\n"
                "Editable positions and valid replacements:\n"
                "  0: {b, g, h, m, s, t}\n"
                "  1: {a*}\n"
                "  2: {l, n, r}\n"
                "  3: {e, s, t}"
            ),
            "action": "EDIT 3 e",
        },
    ],
    "alloy": [
        {
            "observation": (
                "Current composition:\n"
                "  Fe: 25.0  Ni: 15.0  Cr: 20.0  Co: 10.0  Mn: 20.0  Mo: 10.0\n\n"
                "Estimated status:\n"
                "  UTS: ~1340 MPa (target >= 2000)\n"
                "  Density: 8.05 g/cm^3 (target <= 8.20)\n"
                "  Path efficiency: efficient\n\n"
                "Phase: coarse (allowed deltas: 10, 15)\n"
                "Steps used: 0/8  (remaining: 8)\n\n"
                "Action format: EDIT <inc_element> <dec_element> <delta>"
            ),
            "action": "EDIT Ni Fe 10",
        },
    ],
    "gb1_sequence": [
        {
            "observation": (
                "Current sequence: V39=V  D40=D  G41=G  V54=V\n"
                "Wild-type:        V39=V  D40=D  G41=G  V54=V\n\n"
                "Estimated fitness: 1.02  (target: >= 5.0)\n"
                "  [Based on single-site measurements — may not reflect true combinatorial fitness]\n"
                "Stability: stable\n\n"
                "Available mutations (est. single-site effect, stability):\n"
                "  Position 0 (V39, current=V): {L(+0.70, neutral), I(+0.45, neutral)}\n"
                "  Position 1 (D40, current=D): {W(+2.90, mild risk), Y(+2.89, mild risk)}"
            ),
            "action": "EDIT 1 W",
        },
    ],
}


class FewShotFormatPrompt(BasePromptBuilder):
    def __init__(self):
        super().__init__("few_shot_format", metadata={
            "family": "few_shot",
            "strategy_bearing": False,
            "task_specific": False,
            "uses_examples": True,
        })

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        examples = FORMAT_EXEMPLARS.get(env_name, [])
        example_text = ""
        for i, ex in enumerate(examples, 1):
            example_text += (
                f"\nExample {i}:\n"
                f"Observation:\n{ex['observation']}\n"
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

    def build_step_prompt(self, observation, memory_context, step, budget, env_name):
        prompt = f"Step {step}.\n\n"
        if memory_context:
            prompt += f"{memory_context}\n\n"
        prompt += f"Current state:\n{observation}\n\nYour action:"
        return prompt
