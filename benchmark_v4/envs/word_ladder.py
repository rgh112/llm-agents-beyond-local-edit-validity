"""
Word Ladder — Constructive Lexical Editing Environment

The agent edits the current word one letter at a time to reach the target word.
Each intermediate result must be a valid English dictionary word.

Action format: EDIT <position> <letter>
             or: FINALIZE (auto-succeeds if current == target)

Constructive editing properties:
  C1. Edit locality — one character change per step
  C2. Path dependence — early edits determine reachable word regions
  C3. Partial-state semantics — intermediate words constrain future moves
  C4. Delayed evaluation — success only when target is reached
  C5. Order sensitivity — editing position order affects dead-end risk
"""
from __future__ import annotations

import csv
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv, EditState, StepResult
from benchmark_v4.failure_taxonomy import FailureEvent, StepRecord


# ---------------------------------------------------------------------------
def _find_project_root() -> pathlib.Path:
    """Find the repository root from either live or supplementary code paths."""
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "wordladder_data").is_dir():
            return candidate
    return pathlib.Path(__file__).resolve().parent.parent.parent


_PROJECT_ROOT = _find_project_root()
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "wordladder_data"


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return len(a) + len(b)
    return sum(x != y for x, y in zip(a, b))


def _bfs_distance(graph: nx.Graph, source: str, target: str) -> int:
    try:
        return nx.shortest_path_length(graph, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return -1


class WordLadderEnv(BaseConstructiveEditEnv):
    """Constructive lexical editing: edit one letter at a time to reach target.

    The agent decides WHICH position to edit and WHAT letter to put there.
    The resulting word must be in the dictionary and not previously visited.
    """

    # ── prompt-builder properties ─────────────────────────────

    @property
    def task_description(self) -> str:
        return (
            "You are editing a word one letter at a time to reach a target word. "
            "Each intermediate word must be a valid English word. "
            "You choose which position to change and what letter to put there. "
            "You cannot revisit any word already used."
        )

    @property
    def constraints(self) -> str:
        return (
            "1. Change exactly one letter per step using EDIT <position> <letter>.\n"
            "2. The resulting word must be a valid English word.\n"
            "3. You cannot revisit a word already in the path.\n"
            "4. You must reach the target within the step budget.\n"
            "5. Positions are 0-indexed from the left."
        )

    @property
    def action_format(self) -> str:
        return "EDIT <position> <letter>\nExample: EDIT 0 r"

    # ── construction ──────────────────────────────────────────

    def __init__(
        self,
        data_dir: str | pathlib.Path | None = None,
        word_length: int = 4,
        max_steps_factor: int = 5,
        shortest_path_range: Tuple[int, int] = (3, 6),
    ):
        super().__init__(env_name="word_ladder", max_steps=50)
        self.word_length = word_length
        self.max_steps_factor = max_steps_factor
        self.shortest_path_range = shortest_path_range
        self.data_dir = pathlib.Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

        self.vocab: Set[str] = self._load_vocab()
        self.graph: nx.Graph = self._build_graph()
        self.pairs: List[Tuple[str, str, int]] = self._load_pairs()

        self.start_word: str = ""
        self._last_raw_action: Optional[str] = None

    # ── data loading ──────────────────────────────────────────

    def _load_vocab(self) -> Set[str]:
        dict_path = self.data_dir / "enable1.txt"
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_path}")
        return {
            w.strip().lower()
            for w in dict_path.open()
            if w.strip().isalpha() and len(w.strip()) == self.word_length
        }

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.vocab)
        words = sorted(self.vocab)
        for i, w1 in enumerate(words):
            for w2 in words[i + 1:]:
                if _hamming(w1, w2) == 1:
                    G.add_edge(w1, w2)
        return G

    def _load_pairs(self) -> List[Tuple[str, str, int]]:
        pairs_file = self.data_dir / f"pairs_500_len{self.word_length}.csv"
        if not pairs_file.exists():
            return []
        lo, hi = self.shortest_path_range
        pairs = []
        with open(pairs_file) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                s, t = row[0].strip().lower(), row[1].strip().lower()
                d = int(row[2])
                if s in self.vocab and t in self.vocab and lo <= d <= hi:
                    pairs.append((s, t, d))
        return pairs

    # ── helpers ────────────────────────────────────────────────

    def _get_valid_replacements(self, word: str, used: Set[str]) -> Dict[int, List[str]]:
        """For each position, find letters that produce a valid unvisited word."""
        replacements: Dict[int, List[str]] = {}
        for pos in range(len(word)):
            valid_letters = []
            for c in "abcdefghijklmnopqrstuvwxyz":
                if c == word[pos]:
                    continue
                new_word = word[:pos] + c + word[pos + 1:]
                if new_word in self.vocab and new_word not in used:
                    valid_letters.append(c)
            if valid_letters:
                replacements[pos] = valid_letters
        return replacements

    def _compute_neighbors(self, word: str, used: Set[str]) -> List[str]:
        """All valid dictionary neighbors not yet visited."""
        neighbors = []
        for pos in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                if c == word[pos]:
                    continue
                new_word = word[:pos] + c + word[pos + 1:]
                if new_word in self.vocab and new_word not in used:
                    neighbors.append(new_word)
        return sorted(set(neighbors))

    def _distance_avoiding_used(self, word: str, target: str, used: Set[str]) -> int:
        """Shortest remaining path while respecting the no-revisit constraint."""
        if word == target:
            return 0
        blocked = set(used)
        blocked.discard(word)
        queue = [(word, 0)]
        seen = {word}
        head = 0
        while head < len(queue):
            cur, dist = queue[head]
            head += 1
            for nxt in self.graph.neighbors(cur):
                if nxt in blocked or nxt in seen:
                    continue
                if nxt == target:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return -1

    def _diagnostic_snapshot(self, word: str, used: Set[str], budget_left: int) -> Dict[str, Any]:
        """Visible future-feasibility diagnostics for recoverability analysis."""
        target = self.state.latent["target_word"]
        distance = self._distance_avoiding_used(word, target, used)
        recoverable = distance >= 0 and distance <= max(0, budget_left)
        local_proxy = -float(_hamming(word, target))
        if distance < 0:
            recoverability_score = 0.0
        elif budget_left <= 0:
            recoverability_score = 1.0 if distance == 0 else 0.0
        else:
            recoverability_score = max(0.0, 1.0 - distance / float(budget_left + 1))
        return {
            "remaining_distance": distance,
            "remaining_budget": budget_left,
            "recoverable": recoverable,
            "recoverability_score": round(recoverability_score, 4),
            "local_proxy_score": local_proxy,
        }

    # ── core interface ────────────────────────────────────────

    def reset(self, seed: int, **kwargs) -> str:
        super().reset(seed)

        pair_index = kwargs.get("pair_index", None)
        if pair_index is not None and pair_index < len(self.pairs):
            start, target, k_opt = self.pairs[pair_index]
        elif self.pairs:
            idx = int(self.rng.integers(0, len(self.pairs)))
            start, target, k_opt = self.pairs[idx]
        else:
            raise ValueError("No word pairs available")

        self.max_steps = max(k_opt * self.max_steps_factor, 15)
        self.start_word = start

        used_words = {start}
        dist_to_target = _bfs_distance(self.graph, start, target)

        self.state = EditState(
            t=0,
            budget_left=self.max_steps,
            structure=start,  # current word
            history=[],
            latent={
                "target_word": target,
                "used_words": used_words,
                "k_opt": k_opt,
                "optimal_distance": dist_to_target,
                "path": [start],
            },
            done=False,
        )

        self._last_raw_action = None
        return self.render_observation()

    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        text = raw_action.strip()

        # Check for FINALIZE
        if re.match(r"(?i)^finalize\b", text):
            return {"type": "finalize"}

        # Parse EDIT <pos> <letter>
        m = re.match(r"(?i)^edit\s+(\d+)\s+([a-zA-Z])$", text)
        if m:
            pos = int(m.group(1))
            letter = m.group(2).lower()
            return {"type": "edit", "position": pos, "letter": letter}

        return {"type": "invalid", "raw": text}

    def validate_action(self, parsed: Dict[str, Any]) -> Tuple[bool, List[FailureEvent]]:
        events: List[FailureEvent] = []

        if parsed["type"] == "invalid":
            events.append(FailureEvent.MALFORMED_ACTION)
            return False, events

        if parsed["type"] == "finalize":
            current = self.state.structure
            target = self.state.latent["target_word"]
            if current != target:
                events.append(FailureEvent.PREMATURE_FINALIZE)
                return False, events
            return True, events

        # type == "edit"
        pos = parsed["position"]
        letter = parsed["letter"]
        current = self.state.structure

        # Position range check
        if pos < 0 or pos >= len(current):
            events.append(FailureEvent.INVALID_POSITION)
            return False, events

        # Same letter check
        if letter == current[pos]:
            events.append(FailureEvent.ILLEGAL_EDIT)
            return False, events

        # Value validity (must be a-z)
        if not letter.isalpha() or len(letter) != 1:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events

        # Resulting word check
        new_word = current[:pos] + letter + current[pos + 1:]

        if new_word not in self.vocab:
            events.append(FailureEvent.INVALID_WORD)
            return False, events

        # Already visited check
        if new_word in self.state.latent["used_words"]:
            events.append(FailureEvent.OSCILLATION)
            return False, events

        return True, events

    def apply_edit(self, parsed: Dict[str, Any]) -> List[FailureEvent]:
        events: List[FailureEvent] = []
        pos = parsed["position"]
        letter = parsed["letter"]
        current = self.state.structure

        new_word = current[:pos] + letter + current[pos + 1:]

        # Detect repeated exact edit (same pos+letter as last step)
        if self._last_raw_action is not None:
            last_parsed = self.parse_action(self._last_raw_action)
            if (last_parsed.get("type") == "edit"
                    and last_parsed.get("position") == pos
                    and last_parsed.get("letter") == letter):
                events.append(FailureEvent.REPEATED_EXACT_EDIT)

        # Update state
        old_word = self.state.structure
        self.state.structure = new_word
        self.state.latent["used_words"].add(new_word)
        self.state.latent["path"].append(new_word)

        # Update optimal distance
        target = self.state.latent["target_word"]
        self.state.latent["optimal_distance"] = _bfs_distance(
            self.graph, new_word, target
        )

        # Record history
        self.state.history.append({
            "t": self.state.t,
            "edit": f"pos {pos}: {old_word[pos]}→{letter}",
            "word_before": old_word,
            "word_after": new_word,
            "distance_to_target": self.state.latent["optimal_distance"],
        })

        return events

    def finalize(self) -> Dict[str, Any]:
        current = self.state.structure
        target = self.state.latent["target_word"]
        success = current == target
        return {
            "success": success,
            "final_word": current,
            "target_word": target,
            "path_length": len(self.state.latent["path"]) - 1,
            "optimal_length": self.state.latent["k_opt"],
        }

    def step(self, raw_action: str) -> StepResult:
        structure_before = self.state.structure
        used_before = set(self.state.latent["used_words"])
        budget_before = self.state.budget_left
        diag_before = self._diagnostic_snapshot(
            structure_before, used_before, budget_before
        )
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
            # (if not success, premature_finalize was already added in validate)

        elif valid and parsed["type"] == "edit":
            edit_events = self.apply_edit(parsed)
            events.extend(edit_events)

            current = self.state.structure
            target = self.state.latent["target_word"]

            # Auto-success on target reached
            if current == target:
                terminated = True
                self._success = True
                reward = 1.0
                info["auto_success"] = True

            # Dead end check
            elif not self._compute_neighbors(
                current, self.state.latent["used_words"]
            ):
                terminated = True
                events.append(FailureEvent.LOCAL_OPTIMUM_TRAP)
                info["dead_end"] = True

            # Budget check
            elif self.state.budget_left <= 0:
                truncated = True
                events.append(FailureEvent.BUDGET_UNAWARE_ACTION)

        else:
            # Invalid action
            reward = -0.1
            if self.state.budget_left <= 0:
                truncated = True
                events.append(FailureEvent.BUDGET_UNAWARE_ACTION)

        if terminated or truncated:
            self._done = True

        self._last_raw_action = raw_action

        diag_after = self._diagnostic_snapshot(
            self.state.structure,
            set(self.state.latent["used_words"]),
            self.state.budget_left,
        )
        info.update({
            "diagnostics": {
                "recoverability_before": diag_before,
                "recoverability_after": diag_after,
                "recoverability_drop": (
                    bool(diag_before["recoverable"]) and not bool(diag_after["recoverable"])
                ),
                "recoverability_score_delta": round(
                    float(diag_after["recoverability_score"])
                    - float(diag_before["recoverability_score"]),
                    4,
                ),
                "local_proxy_delta": round(
                    float(diag_after["local_proxy_score"])
                    - float(diag_before["local_proxy_score"]),
                    4,
                ),
            }
        })
        info["diagnostics"]["local_improvement_deception"] = (
            info["diagnostics"]["local_proxy_delta"] > 0
            and info["diagnostics"]["recoverability_score_delta"] < 0
        )

        # Record step
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
            structure_after=self.state.structure,
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
        current = self.state.structure
        target = self.state.latent["target_word"]
        used = self.state.latent["used_words"]
        path = self.state.latent["path"]

        # Compute valid replacements per position
        replacements = self._get_valid_replacements(current, used)

        # Show current vs target with position markers
        pos_line = "".join(str(i) for i in range(len(current)))
        diff_line = "".join(
            "^" if current[i] != target[i] else " "
            for i in range(len(current))
        )

        # Path display
        if len(path) <= 6:
            path_display = " → ".join(path)
        else:
            path_display = " → ".join(path[:3]) + " → ... → " + " → ".join(path[-2:])

        lines = [
            f"Target:     {target}",
            f"Current:    {current}",
            f"Positions:  {pos_line}",
            f"Mismatches: {diff_line}",
            f"Path: {path_display}",
            f"Steps used: {self.state.t}/{self.max_steps}  "
            f"(remaining: {self.state.budget_left})",
            "",
            "Editable positions and valid replacements:",
        ]

        if replacements:
            for pos in sorted(replacements.keys()):
                letters = replacements[pos]
                # Highlight letters that match target
                annotated = []
                for l in sorted(letters):
                    if pos < len(target) and l == target[pos]:
                        annotated.append(f"{l}*")  # mark target-matching
                    else:
                        annotated.append(l)
                lines.append(
                    f"  {pos}: {{{', '.join(annotated)}}}"
                )
        else:
            lines.append("  (no valid edits — dead end)")

        return "\n".join(lines)

    def get_episode_summary(self) -> Dict[str, Any]:
        all_events: List[FailureEvent] = []
        for rec in self.trajectory:
            all_events.extend(rec.events)

        violation_by_type = Counter(
            e.name if isinstance(e, FailureEvent) else str(e)
            for e in all_events
        )

        return {
            "success": self._success,
            "steps_taken": self.state.t,
            "max_steps": self.max_steps,
            "start_word": self.start_word,
            "target_word": self.state.latent["target_word"],
            "final_word": self.state.structure,
            "path": list(self.state.latent["path"]),
            "path_length": len(self.state.latent["path"]) - 1,
            "optimal_length": self.state.latent["k_opt"],
            "violations_by_type": dict(violation_by_type),
            "total_violations": len(all_events),
            "optimal_distance_at_end": self.state.latent["optimal_distance"],
            "events": [
                e.name if isinstance(e, FailureEvent) else str(e)
                for e in all_events
            ],
        }
