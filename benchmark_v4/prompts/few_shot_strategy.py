"""Strategy-bearing few-shot: exemplars encode domain-specific editing heuristics.

Unlike few_shot_format (which shows syntax only), these exemplars demonstrate
strategic reasoning — e.g., choosing non-obvious edits, balancing tradeoffs,
or exploring despite negative local estimates. This injects a strategic prior
into the model's editing policy.
"""
from benchmark_v4.prompts.base_prompt import BasePromptBuilder


# NOTE: These exemplars are strategy priors, not just format demonstrations.
# They teach the model WHAT to do, not just HOW to format actions.
STRATEGY_EXEMPLARS = {
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
            "action": "EDIT 2 r",
        },
        {
            "observation": (
                "Target:     warm\n"
                "Current:    cord\n"
                "Positions:  0123\n"
                "Mismatches: ^ ^^\n"
                "Path: cold → cord\n"
                "Steps used: 1/20  (remaining: 19)\n\n"
                "Editable positions and valid replacements:\n"
                "  0: {b, f, g, l, w*}\n"
                "  1: {a*}\n"
                "  2: {l, n}\n"
                "  3: {e, k, n}"
            ),
            "action": "EDIT 0 w",
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
            "action": "EDIT Co Fe 10",
        },
        {
            "observation": (
                "Current composition:\n"
                "  Fe: 15.0  Ni: 15.0  Cr: 20.0  Co: 20.0  Mn: 20.0  Mo: 10.0\n\n"
                "Estimated status:\n"
                "  UTS: ~1890 MPa (target >= 2000)\n"
                "  Density: 8.19 g/cm^3 (target <= 8.20)\n"
                "  Path efficiency: efficient\n\n"
                "Phase: fine (allowed deltas: 2, 5)\n"
                "Steps used: 3/8  (remaining: 5)\n\n"
                "Action format: EDIT <inc_element> <dec_element> <delta>"
            ),
            "action": "EDIT Cr Ni 2",
        },
    ],
    "gb1_sequence": [
        {
            "observation": (
                "Current sequence: V39=V  D40=W  G41=G  V54=V\n"
                "Wild-type:        V39=V  D40=D  G41=G  V54=V\n\n"
                "Estimated fitness: 3.85  (target: >= 5.0)\n"
                "  [Based on single-site measurements — may not reflect true combinatorial fitness]\n"
                "Stability: moderate\n\n"
                "Available mutations (est. single-site effect, stability):\n"
                "  Position 0 (V39, current=V): {L(+0.68, neutral), F(-0.55, mild risk)}\n"
                "  Position 2 (G41, current=G): {A(-0.85, destabilizing), C(-0.92, destabilizing)}\n"
                "  Position 3 (V54, current=V): {C(+0.38, mild risk), A(+0.35, stabilizing)}"
            ),
            "action": "EDIT 0 F",
        },
        {
            "observation": (
                "Current sequence: V39=F  D40=W  G41=G  V54=A\n"
                "Wild-type:        V39=V  D40=D  G41=G  V54=V\n\n"
                "Estimated fitness: 3.62  (target: >= 5.0)\n"
                "  [Based on single-site measurements — may not reflect true combinatorial fitness]\n"
                "Stability: borderline\n\n"
                "Available mutations (est. single-site effect, stability):\n"
                "  Position 2 (G41, current=G): {A(-0.88, destabilizing), C(-0.95, destabilizing)}\n"
                "  Position 3 (V54, current=A): {C(+0.04, mild risk)}\n\n"
                "Edit history:\n"
                "  Step 1: pos 1: D->W  stability: stable\n"
                "  Step 2: pos 0: V->F  stability: moderate\n"
                "  Step 3: pos 3: V->A  stability: moderate"
            ),
            "action": "EDIT 2 A",
        },
    ],
}


class FewShotStrategyPrompt(BasePromptBuilder):
    def __init__(self):
        super().__init__("few_shot_strategy", metadata={
            "family": "few_shot",
            "strategy_bearing": True,
            "task_specific": True,
            "uses_examples": True,
        })

    def build_system_prompt(self, env_name, task_description, constraints, action_format):
        examples = STRATEGY_EXEMPLARS.get(env_name, [])
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
