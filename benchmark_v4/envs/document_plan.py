"""Document Plan Editing — NLP-native constructive editing environment.

The agent edits a short document outline one slot at a time. Local edits are
easy to make, but final success depends on delayed global constraints:
coverage, prerequisite order, duplicate avoidance, and contradiction control.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv, EditState, StepResult
from benchmark_v4.failure_taxonomy import FailureEvent, StepRecord


SECTIONS = {
    "PROBLEM": "state the task and motivation",
    "METHOD": "describe the constructive-editing protocol",
    "EXPERIMENT": "describe evaluation setup",
    "ANALYSIS": "interpret trajectory failures",
    "LIMITATION": "scope surrogate and diagnostic claims",
    "CLAIM": "make a broad unsupported claim",
    "BACKGROUND": "generic background",
}

VALID_OUTLINES = [
    ["PROBLEM", "METHOD", "EXPERIMENT", "ANALYSIS", "LIMITATION"],
    ["PROBLEM", "BACKGROUND", "METHOD", "EXPERIMENT", "ANALYSIS"],
]

START_OUTLINES = [
    ["PROBLEM", "EXPERIMENT", "METHOD", "CLAIM", "BACKGROUND"],
    ["BACKGROUND", "METHOD", "PROBLEM", "EXPERIMENT", "CLAIM"],
    ["PROBLEM", "METHOD", "BACKGROUND", "CLAIM", "ANALYSIS"],
    ["CLAIM", "PROBLEM", "EXPERIMENT", "BACKGROUND", "METHOD"],
]

REQUIRED = {"PROBLEM", "METHOD", "EXPERIMENT", "ANALYSIS"}
ORDER_CONSTRAINTS = [
    ("PROBLEM", "METHOD"),
    ("METHOD", "EXPERIMENT"),
    ("EXPERIMENT", "ANALYSIS"),
]
CONTRADICTIONS = {("CLAIM", "LIMITATION")}


class DocumentPlanEnv(BaseConstructiveEditEnv):
    @property
    def task_description(self) -> str:
        return (
            "You are editing a document outline for a technical paper. "
            "Your goal is an outline that covers the required content, orders "
            "prerequisites before dependent sections, avoids duplicates, and "
            "does not mix unsupported broad claims with explicit limitations."
        )

    @property
    def constraints(self) -> str:
        return (
            "1. Each step: EDIT <position> <section_id>.\n"
            "2. Positions are 0-4.\n"
            "3. Section IDs must be one of the listed candidates.\n"
            "4. FINALIZE triggers delayed global evaluation.\n"
            "5. Required sections: PROBLEM, METHOD, EXPERIMENT, ANALYSIS.\n"
            "6. Prerequisite order: PROBLEM before METHOD before EXPERIMENT before ANALYSIS.\n"
            "7. Avoid duplicate sections and avoid CLAIM together with LIMITATION."
        )

    @property
    def action_format(self) -> str:
        return "EDIT <position> <section_id>\nExample: EDIT 2 EXPERIMENT\nOr: FINALIZE"

    def __init__(self, max_steps: int = 6):
        super().__init__(env_name="document_plan", max_steps=max_steps)
        self._last_raw_action = None

    def reset(self, seed: int, **kwargs) -> str:
        super().reset(seed)
        outline = list(START_OUTLINES[int(self.rng.integers(0, len(START_OUTLINES)))])
        self.state = EditState(
            t=0,
            budget_left=self.max_steps,
            structure=outline,
            history=[],
            latent={"edited_positions": Counter()},
            done=False,
        )
        self._last_raw_action = None
        return self.render_observation()

    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        text = raw_action.strip()
        if re.match(r"(?i)^finalize\b", text):
            return {"type": "finalize"}
        m = re.match(r"(?i)^edit\s+(\d+)\s+([A-Za-z_]+)$", text)
        if m:
            return {"type": "edit", "position": int(m.group(1)), "section": m.group(2).upper()}
        return {"type": "invalid", "raw": text}

    def validate_action(self, parsed: Dict[str, Any]) -> Tuple[bool, List[FailureEvent]]:
        events: List[FailureEvent] = []
        if parsed["type"] == "invalid":
            return False, [FailureEvent.MALFORMED_ACTION]
        if parsed["type"] == "finalize":
            if not self._outline_success(self.state.structure):
                return False, [FailureEvent.PREMATURE_FINALIZE]
            return True, events
        pos = parsed["position"]
        section = parsed["section"]
        if pos < 0 or pos >= 5:
            return False, [FailureEvent.INVALID_POSITION]
        if section not in SECTIONS:
            return False, [FailureEvent.INVALID_VALUE]
        if self.state.structure[pos] == section:
            return False, [FailureEvent.ILLEGAL_EDIT]
        return True, events

    def _outline_success(self, outline: List[str]) -> bool:
        if not REQUIRED.issubset(set(outline)):
            return False
        if len(outline) != len(set(outline)):
            return False
        for a, b in ORDER_CONSTRAINTS:
            if a in outline and b in outline and outline.index(a) > outline.index(b):
                return False
        for a, b in CONTRADICTIONS:
            if a in outline and b in outline:
                return False
        return True

    def _hamming_to_valid(self, outline: List[str]) -> int:
        return min(
            sum(1 for x, y in zip(outline, target) if x != y)
            for target in VALID_OUTLINES
        )

    def _local_proxy_score(self, outline: List[str]) -> float:
        coverage = len(REQUIRED.intersection(outline))
        duplicate_penalty = len(outline) - len(set(outline))
        order_hits = 0
        for a, b in ORDER_CONSTRAINTS:
            if a in outline and b in outline and outline.index(a) < outline.index(b):
                order_hits += 1
        contradiction_penalty = sum(1 for a, b in CONTRADICTIONS if a in outline and b in outline)
        return coverage + 0.5 * order_hits - duplicate_penalty - contradiction_penalty

    def _diagnostic_snapshot(self) -> Dict[str, Any]:
        outline = list(self.state.structure)
        distance = self._hamming_to_valid(outline)
        recoverable = distance <= max(0, self.state.budget_left)
        score = 1.0 if distance == 0 else max(0.0, 1.0 - distance / float(self.state.budget_left + 1))
        return {
            "remaining_budget": self.state.budget_left,
            "remaining_distance": distance,
            "recoverable": recoverable,
            "recoverability_score": round(score, 4),
            "local_proxy_score": round(self._local_proxy_score(outline), 4),
        }

    def apply_edit(self, parsed: Dict[str, Any]) -> List[FailureEvent]:
        pos = parsed["position"]
        section = parsed["section"]
        old = self.state.structure[pos]
        self.state.structure[pos] = section
        self.state.latent["edited_positions"][pos] += 1
        self.state.history.append({
            "t": self.state.t,
            "edit": f"slot {pos}: {old}->{section}",
            "outline_after": list(self.state.structure),
        })
        events: List[FailureEvent] = []
        if self.state.latent["edited_positions"][pos] > 1:
            events.append(FailureEvent.OSCILLATION)
        if len(self.state.structure) != len(set(self.state.structure)):
            events.append(FailureEvent.GLOBAL_FEASIBILITY_LOSS)
        return events

    def finalize(self) -> Dict[str, Any]:
        success = self._outline_success(self.state.structure)
        return {
            "success": success,
            "outline": list(self.state.structure),
            "missing_required": sorted(REQUIRED.difference(self.state.structure)),
            "distance_to_valid_outline": self._hamming_to_valid(self.state.structure),
        }

    def step(self, raw_action: str) -> StepResult:
        structure_before = list(self.state.structure)
        diag_before = self._diagnostic_snapshot()
        parsed = self.parse_action(raw_action)
        valid, events = self.validate_action(parsed)
        self.state.t += 1
        self.state.budget_left = self.max_steps - self.state.t
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {"valid": valid}

        if valid and parsed["type"] == "finalize":
            terminated = True
            result = self.finalize()
            info.update(result)
            if result["success"]:
                self._success = True
                reward = 1.0
        elif valid and parsed["type"] == "edit":
            events.extend(self.apply_edit(parsed))
            if self.state.budget_left <= 0:
                truncated = True
                events.append(FailureEvent.BUDGET_UNAWARE_ACTION)
        else:
            reward = -0.1
            if self.state.budget_left <= 0:
                truncated = True
                events.append(FailureEvent.BUDGET_UNAWARE_ACTION)

        if terminated or truncated:
            self._done = True
        self._last_raw_action = raw_action

        diag_after = self._diagnostic_snapshot()
        info["diagnostics"] = {
            "recoverability_before": diag_before,
            "recoverability_after": diag_after,
            "recoverability_drop": bool(diag_before["recoverable"]) and not bool(diag_after["recoverable"]),
            "recoverability_score_delta": round(
                float(diag_after["recoverability_score"]) - float(diag_before["recoverability_score"]), 4
            ),
            "local_proxy_delta": round(
                float(diag_after["local_proxy_score"]) - float(diag_before["local_proxy_score"]), 4
            ),
        }
        info["diagnostics"]["local_improvement_deception"] = (
            info["diagnostics"]["local_proxy_delta"] > 0
            and info["diagnostics"]["recoverability_score_delta"] < 0
        )

        step_record = StepRecord(
            t=self.state.t,
            observation=self.render_observation() if not terminated else "",
            raw_action=raw_action,
            parsed_action=parsed,
            valid=valid,
            events=list(events),
            reward=reward,
            info=info,
            structure_before=structure_before,
            structure_after=list(self.state.structure),
        )
        self.record_step(step_record)
        return StepResult(
            observation=self.render_observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            events=list(events),
        )

    def render_observation(self) -> str:
        lines = [
            f"Step {self.state.t + 1}/{self.max_steps} | Budget remaining: {self.state.budget_left}",
            "Current outline:",
        ]
        for i, section in enumerate(self.state.structure):
            lines.append(f"  {i}: {section} - {SECTIONS[section]}")
        lines.extend([
            "",
            "Candidate sections:",
        ])
        for section, desc in SECTIONS.items():
            lines.append(f"  {section}: {desc}")
        return "\n".join(lines)

    def get_episode_summary(self) -> Dict[str, Any]:
        all_events: List[FailureEvent] = []
        for rec in self.trajectory:
            all_events.extend(rec.events)
        return {
            "success": self._success,
            "steps_taken": self.state.t,
            "max_steps": self.max_steps,
            "outline": list(self.state.structure),
            "distance_to_valid_outline": self._hamming_to_valid(self.state.structure),
            "violations_by_type": dict(Counter(e.name for e in all_events)),
            "events": [e.name for e in all_events],
        }
