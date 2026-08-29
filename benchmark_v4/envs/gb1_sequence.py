"""
GB1 landscape -- restricted constructive biological sequence editing

The agent edits a 4-position protein sequence (GB1 domain, positions V39/D40/G41/V54)
one residue at a time to maximize IgG-Fc binding fitness.

Local mutation priors are grounded in real Deep Mutational Scanning data (FLIP benchmark),
while path-dependent penalties and observation noise are designed to expose
constructive planning failures under delayed evaluation.

Action format: EDIT <position> <amino_acid>
             or: FINALIZE (triggers full evaluation)

Constructive editing properties:
  C1. Edit locality — one residue change per step
  C2. Path dependence — epistatic interactions make edit order matter
  C3. Partial-state semantics — single-site estimates are misleading (epistasis)
  C4. Delayed evaluation — true combinatorial fitness revealed only on FINALIZE
  C5. Order sensitivity — stability cost of early edits gates later options

Key biological grounding:
  - True fitness values come from GB1 DMS data (149K measured variants)
  - Position 2 (G41) is a natural epistatic trap: best single-site is WT (G),
    but top combinatorial variants all require mutation at this position
  - Single-site fitness estimates systematically mispredict combinatorial fitness
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv, EditState, StepResult
from benchmark_v4.failure_taxonomy import FailureEvent, StepRecord


_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# Stability deltas per (position, residue) — designed to create path dependence.
# Position 2 (G41) mutations are destabilizing because G provides backbone flexibility.
# This gates access to premium mutations at other positions if done carelessly.
_STABILITY_DELTAS = {
    # Position 0 (V39)
    (0, "V"): 0.0,   # WT, no change
    (0, "F"): -0.08,  # aromatic, mild destabilization
    (0, "L"): -0.02,  # conservative, nearly neutral
    (0, "I"): -0.03,  # conservative
    (0, "W"): -0.15,  # bulky aromatic, destabilizing
    # Position 1 (D40)
    (1, "D"): 0.0,
    (1, "W"): -0.10,  # large change
    (1, "Y"): -0.08,
    (1, "H"): -0.05,
    (1, "A"): +0.02,  # small residue, slightly stabilizing
    # Position 2 (G41) — key: mutations here are destabilizing but necessary
    (2, "G"): 0.0,
    (2, "A"): -0.18,  # loss of backbone flexibility
    (2, "C"): -0.20,  # disulfide risk
    (2, "L"): -0.25,  # large hydrophobic insertion
    (2, "F"): -0.22,  # aromatic insertion
    # Position 3 (V54)
    (3, "V"): 0.0,
    (3, "A"): +0.03,  # smaller, stabilizing
    (3, "C"): -0.06,
    (3, "G"): +0.05,  # flexible, stabilizing
    (3, "F"): -0.10,  # aromatic
}

# Stability threshold below which premium mutations become unavailable
_STABILITY_GATE = 0.55


def _load_gb1_data() -> Dict:
    data_path = _DATA_DIR / "gb1_fitness.json"
    if not data_path.exists():
        raise FileNotFoundError(f"GB1 data not found: {data_path}")
    with open(data_path) as f:
        return json.load(f)


class GB1SequenceEnv(BaseConstructiveEditEnv):
    """Semi-real protein sequence editing based on GB1 DMS landscape.

    The agent edits 4 positions in the GB1 domain to maximize binding fitness.
    True fitness comes from real experimental data; observation noise and
    stability mechanics create constructive editing challenges.
    """

    @property
    def task_description(self) -> str:
        return (
            "You are editing a protein sequence (GB1 domain) to maximize binding fitness. "
            f"Your goal: fitness >= {self._fitness_target:.1f} on FINALIZE. "
            "You can edit 4 key positions, one residue at a time. "
            "Fitness estimates shown are based on single-site measurements and may be "
            "MISLEADING — real fitness depends on interactions between positions. "
            "True fitness is revealed only on FINALIZE. "
            "Some positions that appear unimportant alone may be critical in combination. "
            "Structural stability constrains which mutations are available."
        )

    @property
    def constraints(self) -> str:
        return (
            "1. Each step: EDIT <position> <amino_acid> (position 0-3).\n"
            "2. The amino acid must differ from the current residue.\n"
            "3. Each position can be re-edited, but budget is limited.\n"
            "4. If stability drops too low, some mutations become unavailable.\n"
            "5. FINALIZE triggers full evaluation — use when confident.\n"
            "6. Fitness estimates are based on single-site data and may be WRONG\n"
            "   due to interactions between positions (epistasis)."
        )

    @property
    def action_format(self) -> str:
        return (
            "EDIT <position> <amino_acid>\n"
            "Example: EDIT 1 W\n"
            "Or: FINALIZE"
        )

    def __init__(
        self,
        max_steps: int = 6,
        fitness_target: float = 5.0,
        noise_std: float = 0.20,
        stability_target: float = 0.35,
        start_mutations: int = 1,
        evaluator_mode: str = "epistatic",
        stability_gate: float = _STABILITY_GATE,
    ):
        super().__init__(env_name="gb1_sequence", max_steps=max_steps)
        self._fitness_target = fitness_target
        self._noise_std = noise_std
        self._stability_target = stability_target
        self._start_mutations = start_mutations
        self._evaluator_mode = evaluator_mode
        self._stability_gate = stability_gate

        # Load real GB1 data
        data = _load_gb1_data()
        self._fitness_lookup: Dict[str, float] = data["fitness_lookup"]
        self._single_site: Dict[str, Dict[str, float]] = data["single_site_fitness"]
        self._positions_meta: List[Dict] = data["positions"]
        self._wildtype: str = data["wildtype"]

        # Build allowed residues map
        self._allowed: Dict[int, List[str]] = {
            i: pos["allowed"] for i, pos in enumerate(self._positions_meta)
        }

        self._last_raw_action: Optional[str] = None

    # ── helpers ──────────────────────────────────────────────

    def _count_mutations_from_wt(self, sequence: List[str]) -> int:
        return sum(1 for s, w in zip(sequence, self._wildtype) if s != w)

    def _mutation_summary_from_wt(self, sequence: List[str]) -> str:
        pos_labels = ["V39", "D40", "G41", "V54"]
        diffs = []
        for i, (cur, wt) in enumerate(zip(sequence, self._wildtype)):
            if cur != wt:
                diffs.append(f"pos {i} ({pos_labels[i]}): {wt}->{cur}")
        return ", ".join(diffs)

    def _estimated_finalize_ready(self) -> Tuple[bool, bool, List[str]]:
        """Check visible readiness for FINALIZE. Returns (fitness_ready, stability_ready, reasons).

        NOTE: Fitness precheck is intentionally lenient. Because single-site estimates
        are systematically wrong due to epistasis, we do NOT gate FINALIZE on estimated
        fitness. The agent can attempt FINALIZE whenever stability allows — the true
        combinatorial fitness (hidden) determines actual success in finalize().
        This preserves the epistasis trap: the agent must decide when to finalize
        without reliable fitness estimates.
        """
        stability = self.state.latent["stability"]
        reasons = []
        # Fitness precheck: always ready (epistasis makes estimates unreliable)
        fitness_ready = True
        # Stability precheck: must be above stability_target
        stability_ready = stability >= self._stability_target
        if not stability_ready:
            reasons.append("stability_not_ready")
        return fitness_ready, stability_ready, reasons

    def _get_variant_string(self, sequence: List[str]) -> str:
        return "".join(sequence)

    def _true_fitness(self, sequence: List[str]) -> float:
        if self._evaluator_mode == "additive":
            return self._estimated_fitness(sequence)
        variant = self._get_variant_string(sequence)
        return self._fitness_lookup.get(variant, 0.0)

    def _estimated_fitness(self, sequence: List[str]) -> float:
        """Additive single-site estimate (systematically wrong due to epistasis)."""
        wt = list(self._wildtype)
        est = 1.0  # WT baseline
        for pos in range(4):
            if sequence[pos] != wt[pos]:
                single = self._single_site[str(pos)].get(sequence[pos], 0.0)
                est += single - 1.0  # delta from WT
        return est

    def _noisy_estimate(self, sequence: List[str]) -> float:
        est = self._estimated_fitness(sequence)
        return est + float(self.rng.normal(0, self._noise_std))

    def _stability_label(self, stab: float) -> str:
        if stab >= 0.85:
            return "stable"
        elif stab >= 0.65:
            return "moderate"
        elif stab >= self._stability_gate:
            return "borderline"
        elif stab >= self._stability_target:
            return "low"
        else:
            return "critical"

    def _get_available_edits(self) -> Dict[int, List[Tuple[str, float, str]]]:
        """Available (residue, noisy_single_site_delta, stability_hint) per position."""
        sequence = list(self.state.structure)
        stability = self.state.latent["stability"]

        available: Dict[int, List[Tuple[str, float, str]]] = {}
        for pos in range(4):
            opts = []
            for aa in self._allowed[pos]:
                if aa == sequence[pos]:
                    continue

                # Stability gate: if stability is low, block destabilizing mutations
                sd = _STABILITY_DELTAS.get((pos, aa), 0.0)
                if stability < self._stability_gate and sd < -0.05:
                    continue

                # Noisy single-site estimate
                single = self._single_site[str(pos)].get(aa, 0.0)
                wt_single = self._single_site[str(pos)].get(self._wildtype[pos], 1.0)
                delta_est = single - wt_single + float(self.rng.normal(0, self._noise_std * 0.5))

                # Stability hint
                if sd < -0.15:
                    hint = "destabilizing"
                elif sd < -0.05:
                    hint = "mild risk"
                elif sd > 0.02:
                    hint = "stabilizing"
                else:
                    hint = "neutral"

                opts.append((aa, round(delta_est, 3), hint))

            if opts:
                available[pos] = opts

        return available

    def _available_edits_for(self, sequence: List[str], stability: float) -> List[Tuple[int, str]]:
        edits: List[Tuple[int, str]] = []
        for pos in range(4):
            for aa in self._allowed[pos]:
                if aa == sequence[pos]:
                    continue
                sd = _STABILITY_DELTAS.get((pos, aa), 0.0)
                if stability < self._stability_gate and sd < -0.05:
                    continue
                edits.append((pos, aa))
        return edits

    def _next_stability(self, sequence: List[str], stability: float, pos: int, aa: str) -> float:
        old_aa = sequence[pos]
        sd = _STABILITY_DELTAS.get((pos, aa), 0.0)
        if aa == self._wildtype[pos]:
            sd = -_STABILITY_DELTAS.get((pos, old_aa), 0.0) * 0.7
        return round(max(0.0, min(1.0, stability + sd)), 4)

    def _success_for(self, sequence: List[str], stability: float) -> bool:
        return (
            self._true_fitness(sequence) >= self._fitness_target
            and stability >= self._stability_target
        )

    def _local_proxy_score(self, sequence: List[str], stability: float) -> float:
        return (
            (self._estimated_fitness(sequence) - self._fitness_target)
            + 0.5 * (stability - self._stability_target)
        )

    def _recoverability_probe(
        self, sequence: List[str], stability: float, budget_left: int
    ) -> Dict[str, Any]:
        """Exhaustive small-state future-feasibility probe."""
        if self._success_for(sequence, stability):
            return {
                "recoverable": True,
                "best_true_fitness": round(self._true_fitness(sequence), 4),
                "best_proxy_score": round(self._local_proxy_score(sequence, stability), 4),
            }
        frontier = [(tuple(sequence), stability)]
        seen = set(frontier)
        best_true = self._true_fitness(sequence)
        best_proxy = self._local_proxy_score(sequence, stability)
        for _ in range(max(0, budget_left)):
            nxt_frontier = []
            for seq_tuple, stab in frontier:
                seq = list(seq_tuple)
                for pos, aa in self._available_edits_for(seq, stab):
                    new_seq = list(seq)
                    new_seq[pos] = aa
                    new_stab = self._next_stability(seq, stab, pos, aa)
                    key = (tuple(new_seq), new_stab)
                    if key in seen:
                        continue
                    seen.add(key)
                    true_fit = self._true_fitness(new_seq)
                    proxy = self._local_proxy_score(new_seq, new_stab)
                    best_true = max(best_true, true_fit)
                    best_proxy = max(best_proxy, proxy)
                    if self._success_for(new_seq, new_stab):
                        return {
                            "recoverable": True,
                            "best_true_fitness": round(best_true, 4),
                            "best_proxy_score": round(best_proxy, 4),
                        }
                    nxt_frontier.append(key)
            if not nxt_frontier:
                break
            frontier = nxt_frontier
        return {
            "recoverable": False,
            "best_true_fitness": round(best_true, 4),
            "best_proxy_score": round(best_proxy, 4),
        }

    def _diagnostic_snapshot(self) -> Dict[str, Any]:
        sequence = list(self.state.structure)
        stability = float(self.state.latent["stability"])
        probe = self._recoverability_probe(sequence, stability, self.state.budget_left)
        true_fit = self._true_fitness(sequence)
        est_fit = self._estimated_fitness(sequence)
        return {
            "remaining_budget": self.state.budget_left,
            "recoverable": bool(probe["recoverable"]),
            "recoverability_score": float(probe["best_true_fitness"]),
            "local_proxy_score": round(self._local_proxy_score(sequence, stability), 4),
            "true_fitness": round(true_fit, 4),
            "estimated_fitness": round(est_fit, 4),
            "estimation_error": round(est_fit - true_fit, 4),
            "stability": round(stability, 4),
        }

    # ── core interface ──────────────────────────────────────

    def _jittered_start(self) -> List[str]:
        """Generate seed-dependent starting sequence with random mutations."""
        seq = list(self._wildtype)
        if self._start_mutations <= 0:
            return seq
        # Randomly mutate 1-2 positions
        positions = list(range(4))
        self.rng.shuffle(positions)
        n_mut = min(self._start_mutations, 4)
        for pos in positions[:n_mut]:
            alternatives = [aa for aa in self._allowed[pos] if aa != seq[pos]]
            if alternatives:
                seq[pos] = str(self.rng.choice(alternatives))
        return seq

    def reset(self, seed: int, **kwargs) -> str:
        super().reset(seed)

        start_sequence = self._jittered_start()

        self.state = EditState(
            t=0,
            budget_left=self.max_steps,
            structure=start_sequence,
            history=[],
            latent={
                "stability": 1.0,
                "edited_positions": set(),
                "edit_count_per_pos": {i: 0 for i in range(4)},
                "edit_sequence": [],  # list of (pos, aa) in order
            },
            done=False,
        )
        self._last_raw_action = None
        return self.render_observation()

    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        text = raw_action.strip()

        if re.match(r"(?i)^finalize\b", text):
            return {"type": "finalize"}

        m = re.match(r"(?i)^edit\s+(\d+)\s+([A-Za-z])$", text)
        if m:
            pos = int(m.group(1))
            aa = m.group(2).upper()
            return {"type": "edit", "position": pos, "amino_acid": aa}

        return {"type": "invalid", "raw": text}

    def validate_action(self, parsed: Dict[str, Any]) -> Tuple[bool, List[FailureEvent]]:
        events: List[FailureEvent] = []

        if parsed["type"] == "invalid":
            events.append(FailureEvent.MALFORMED_ACTION)
            return False, events

        if parsed["type"] == "finalize":
            fitness_ready, stability_ready, reasons = self._estimated_finalize_ready()
            if not fitness_ready or not stability_ready:
                events.append(FailureEvent.PREMATURE_FINALIZE)
                return False, events
            return True, events

        pos = parsed["position"]
        aa = parsed["amino_acid"]
        sequence = self.state.structure

        if pos < 0 or pos >= 4:
            events.append(FailureEvent.INVALID_POSITION)
            return False, events

        if aa not in self._allowed[pos]:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events

        if sequence[pos] == aa:
            events.append(FailureEvent.ILLEGAL_EDIT)
            return False, events

        # Stability gate
        stability = self.state.latent["stability"]
        sd = _STABILITY_DELTAS.get((pos, aa), 0.0)
        if stability < self._stability_gate and sd < -0.05:
            events.append(FailureEvent.HARD_CONSTRAINT_VIOLATION)
            return False, events

        return True, events

    def apply_edit(self, parsed: Dict[str, Any]) -> List[FailureEvent]:
        events: List[FailureEvent] = []
        pos = parsed["position"]
        aa = parsed["amino_acid"]

        old_seq = list(self.state.structure)
        old_aa = old_seq[pos]
        new_seq = list(old_seq)
        new_seq[pos] = aa
        self.state.structure = new_seq

        lat = self.state.latent

        # Stability update
        sd = _STABILITY_DELTAS.get((pos, aa), 0.0)
        # If reverting to WT, recover stability
        if aa == self._wildtype[pos]:
            sd = -_STABILITY_DELTAS.get((pos, old_aa), 0.0) * 0.7  # partial recovery
        prev_stab = lat["stability"]
        lat["stability"] = round(max(0.0, min(1.0, prev_stab + sd)), 4)

        # Track edits
        lat["edited_positions"].add(pos)
        lat["edit_count_per_pos"][pos] += 1
        lat["edit_sequence"].append((pos, aa))

        # Re-edit detection (oscillation)
        if lat["edit_count_per_pos"][pos] > 1:
            events.append(FailureEvent.OSCILLATION)

        # Stability warning
        if lat["stability"] < self._stability_target:
            events.append(FailureEvent.HARD_CONSTRAINT_VIOLATION)
        elif lat["stability"] < self._stability_gate and prev_stab >= self._stability_gate:
            events.append(FailureEvent.GLOBAL_FEASIBILITY_LOSS)

        # Epistasis trap detection: if estimated fitness went up but true went down
        old_true = self._true_fitness(old_seq)
        new_true = self._true_fitness(new_seq)
        old_est = self._estimated_fitness(old_seq)
        new_est = self._estimated_fitness(new_seq)
        if new_est > old_est and new_true < old_true:
            events.append(FailureEvent.LOCAL_OPTIMUM_TRAP)

        # Record history
        self.state.history.append({
            "t": self.state.t,
            "edit": f"pos {pos}: {old_aa}->{aa}",
            "position": pos,
            "old_aa": old_aa,
            "new_aa": aa,
            "noisy_delta": round(self._noisy_estimate(new_seq) - self._noisy_estimate(old_seq), 3),
            "stability_label": self._stability_label(lat["stability"]),
        })

        return events

    def finalize(self) -> Dict[str, Any]:
        lat = self.state.latent
        true_fit = self._true_fitness(self.state.structure)
        est_fit = self._estimated_fitness(self.state.structure)
        stability = lat["stability"]

        success = true_fit >= self._fitness_target and stability >= self._stability_target

        return {
            "success": success,
            "true_fitness": round(true_fit, 4),
            "estimated_fitness": round(est_fit, 4),
            "estimation_error": round(est_fit - true_fit, 4),
            "stability": round(stability, 4),
            "fitness_ok": true_fit >= self._fitness_target,
            "stability_ok": stability >= self._stability_target,
            "variant": self._get_variant_string(self.state.structure),
            "edits_made": len(lat["edit_sequence"]),
            "positions_edited": len(lat["edited_positions"]),
            "evaluator_mode": self._evaluator_mode,
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

        # Record premature finalize reasons if rejected
        if not valid and parsed["type"] == "finalize":
            _, _, reasons = self._estimated_finalize_ready()
            if reasons:
                info["premature_finalize_reasons"] = reasons

        if valid and parsed["type"] == "finalize":
            terminated = True
            result = self.finalize()
            info.update(result)
            if result["success"]:
                self._success = True
                reward = 1.0

        elif valid and parsed["type"] == "edit":
            edit_events = self.apply_edit(parsed)
            events.extend(edit_events)

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
        info["diagnostics"]["local_improvement_deception"] = (
            info["diagnostics"]["local_proxy_delta"] > 0
            and (
                info["diagnostics"]["recoverability_score_delta"] < 0
                or info["diagnostics"]["recoverability_drop"]
            )
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
            structure_before="".join(structure_before),
            structure_after="".join(self.state.structure),
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
        sequence = self.state.structure
        lat = self.state.latent
        stability = lat["stability"]

        # Noisy fitness estimate
        noisy_fit = self._noisy_estimate(sequence)

        pos_labels = ["V39", "D40", "G41", "V54"]
        seq_display = "  ".join(f"{pos_labels[i]}={sequence[i]}" for i in range(4))

        lines = [
            f"Step {self.state.t + 1}/{self.max_steps}  |  "
            f"Budget remaining: {self.state.budget_left}",
            "",
            f"Current sequence: {seq_display}",
            f"Wild-type:        {' '.join(f'{pos_labels[i]}={self._wildtype[i]}' for i in range(4))}",
            "",
            f"Estimated fitness: {noisy_fit:.2f}  (target: >= {self._fitness_target:.1f})",
            f"  [Based on single-site measurements — may not reflect true combinatorial fitness]",
            f"Stability: {self._stability_label(stability)}",
            "",
        ]

        # Available edits
        available = self._get_available_edits()
        if available:
            lines.append("Available mutations (est. single-site effect, stability):")
            for pos in sorted(available.keys()):
                opts = available[pos]
                opts.sort(key=lambda x: x[1], reverse=True)
                opt_strs = [f"{aa}({delta:+.2f}, {hint})" for aa, delta, hint in opts]
                lines.append(f"  Position {pos} ({pos_labels[pos]}, current={sequence[pos]}): "
                             f"{{{', '.join(opt_strs)}}}")
        else:
            lines.append("No mutations available (stability too low).")

        # Edit history
        lines.append("")
        if lat.get("edit_sequence"):
            lines.append("Edit history:")
            for h in self.state.history[-4:]:
                lines.append(f"  Step {h['t']}: {h['edit']}  "
                             f"stability: {h['stability_label']}")
        else:
            num_mut = self._count_mutations_from_wt(sequence)
            if num_mut == 0:
                lines.append("No edits applied yet. Sequence is at wild-type (fitness ~1.0).")
            else:
                diff_text = self._mutation_summary_from_wt(sequence)
                lines.append(
                    f"No edits applied yet. Starting sequence is pre-mutated "
                    f"relative to wild-type at {num_mut} position(s): {diff_text}."
                )

        return "\n".join(lines)

    def get_episode_summary(self) -> Dict[str, Any]:
        all_events: List[FailureEvent] = []
        for rec in self.trajectory:
            all_events.extend(rec.events)

        violation_by_type = Counter(
            e.name if isinstance(e, FailureEvent) else str(e)
            for e in all_events
        )

        lat = self.state.latent
        true_fit = self._true_fitness(self.state.structure)

        return {
            "success": self._success,
            "steps_taken": self.state.t,
            "max_steps": self.max_steps,
            "true_fitness": round(true_fit, 4),
            "estimated_fitness": round(self._estimated_fitness(self.state.structure), 4),
            "stability": round(lat["stability"], 4),
            "variant": self._get_variant_string(self.state.structure),
            "edits_made": len(lat["edit_sequence"]),
            "positions_edited": len(lat["edited_positions"]),
            "evaluator_mode": self._evaluator_mode,
            "violations_by_type": dict(violation_by_type),
            "total_violations": len(all_events),
            "events": [
                e.name if isinstance(e, FailureEvent) else str(e)
                for e in all_events
            ],
        }
