"""
Protein Sequence — Constructive Editing Environment

The agent edits a protein sequence one residue at a time to optimize fitness.
Each intermediate state must maintain structural stability.

Action format: EDIT <position> <amino_acid>
             or: FINALIZE (triggers full evaluation)

Constructive editing properties:
  C1. Edit locality — one residue change per step
  C2. Path dependence — destabilizing edits lock out premium mutations
  C3. Partial-state semantics — stability gates constrain future edits
  C4. Delayed evaluation — true fitness revealed only on FINALIZE
  C5. Order sensitivity — synergy pairs require correct application order
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv, EditState, StepResult
from benchmark_v4.failure_taxonomy import FailureEvent, StepRecord

# 20 canonical amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Synthetic fitness landscape data structures
# ---------------------------------------------------------------------------

@dataclass
class MutationEffect:
    """Effect of mutating a specific position to a specific amino acid."""
    position: int             # 0-indexed
    amino_acid: str
    true_fitness_delta: float
    true_stability_delta: float
    is_trap: bool
    noisy_fitness_bias: float  # added to noisy estimate (traps have high bias)
    requires_stability: float  # minimum stability to attempt this edit


@dataclass
class SynergyPair:
    """Epistatic pair: editing pos_a before pos_b gives bonus fitness."""
    pos_a: int
    aa_a: str
    pos_b: int
    aa_b: str
    bonus: float


# ---------------------------------------------------------------------------
# Landscape generator
# ---------------------------------------------------------------------------

def _generate_landscape(
    rng: np.random.Generator,
    seq_length: int = 20,
    n_traps: int = 8,
    n_synergies: int = 5,
) -> Tuple[str, Dict[Tuple[int, str], MutationEffect], List[SynergyPair]]:
    """Generate a synthetic fitness landscape.

    Returns:
        wild_type: starting sequence
        effects: dict mapping (position, aa) -> MutationEffect
        synergies: list of synergy pairs
    """
    # Generate wild-type sequence
    wt = "".join(rng.choice(AMINO_ACIDS) for _ in range(seq_length))

    effects: Dict[Tuple[int, str], MutationEffect] = {}

    # For each position, generate effects for 3-6 alternative amino acids
    good_mutations: List[Tuple[int, str]] = []

    for pos in range(seq_length):
        wt_aa = wt[pos]
        n_alts = int(rng.integers(3, 7))
        available = [a for a in AMINO_ACIDS if a != wt_aa]
        rng.shuffle(available)
        alts = available[:n_alts]

        for aa in alts:
            # Most are small positive or neutral
            fitness = float(rng.uniform(-0.15, 0.50))
            stab = float(rng.uniform(-0.08, 0.04))

            # High-fitness mutations require stability >= 0.6
            req_stab = 0.6 if fitness > 0.30 else 0.0

            effects[(pos, aa)] = MutationEffect(
                position=pos,
                amino_acid=aa,
                true_fitness_delta=round(fitness, 3),
                true_stability_delta=round(stab, 3),
                is_trap=False,
                noisy_fitness_bias=0.0,
                requires_stability=req_stab,
            )
            if fitness > 0.1:
                good_mutations.append((pos, aa))

    # Designate some as traps: override their properties
    trap_keys = list(effects.keys())
    rng.shuffle(trap_keys)
    trap_count = 0
    for key in trap_keys:
        if trap_count >= n_traps:
            break
        eff = effects[key]
        if eff.true_fitness_delta > 0.0:
            # Convert to trap: low true fitness, high noisy bias, stability cost
            eff.true_fitness_delta = round(float(rng.uniform(-0.10, 0.10)), 3)
            eff.true_stability_delta = round(float(rng.uniform(-0.22, -0.10)), 3)
            eff.noisy_fitness_bias = round(float(rng.uniform(0.25, 0.45)), 3)
            eff.is_trap = True
            eff.requires_stability = 0.0  # traps are always available
            trap_count += 1

    # Generate synergy pairs from good mutations
    synergies: List[SynergyPair] = []
    rng.shuffle(good_mutations)
    syn_count = 0
    used_positions = set()
    for i in range(0, len(good_mutations) - 1, 2):
        if syn_count >= n_synergies:
            break
        pos_a, aa_a = good_mutations[i]
        pos_b, aa_b = good_mutations[i + 1]
        if pos_a == pos_b or pos_a in used_positions or pos_b in used_positions:
            continue
        bonus = round(float(rng.uniform(0.20, 0.40)), 3)
        synergies.append(SynergyPair(
            pos_a=pos_a, aa_a=aa_a,
            pos_b=pos_b, aa_b=aa_b,
            bonus=bonus,
        ))
        used_positions.add(pos_a)
        used_positions.add(pos_b)
        syn_count += 1

    return wt, effects, synergies


class ProteinEnv(BaseConstructiveEditEnv):
    """Constructive protein sequence editing: mutate residues to optimize fitness.

    The agent decides WHICH position to edit and WHAT amino acid to place.
    Must achieve fitness threshold while maintaining structural stability.
    """

    # ── prompt-builder properties ─────────────────────────────

    @property
    def task_description(self) -> str:
        return (
            "You are editing a protein sequence one residue at a time to maximize fitness. "
            f"Your goal: cumulative fitness >= {self._fitness_target:.1f} "
            f"AND stability >= {self._stability_target:.1f}. "
            "Each edit changes one amino acid at a position. "
            "Fitness estimates are noisy — true values revealed only on FINALIZE. "
            "Some mutations that appear beneficial may actually be traps "
            "(low true fitness + stability cost). "
            "If stability drops too low, the best mutations become unavailable. "
            "Some mutation pairs give synergy bonuses when applied in correct order."
        )

    @property
    def constraints(self) -> str:
        return (
            "1. Each step: EDIT <position> <amino_acid> (0-indexed).\n"
            "2. The amino acid must be different from the current residue.\n"
            "3. Each position can only be edited once.\n"
            "4. Stability must stay above threshold to access premium mutations.\n"
            "5. FINALIZE triggers full evaluation — use when you believe targets are met.\n"
            "6. Fitness estimates shown are NOISY. True values on FINALIZE only."
        )

    @property
    def action_format(self) -> str:
        return (
            "EDIT <position> <amino_acid>\n"
            "Example: EDIT 3 G\n"
            "Or: FINALIZE"
        )

    # ── construction ──────────────────────────────────────────

    def __init__(
        self,
        max_steps: int = 8,
        seq_length: int = 20,
        noise_std: float = 0.15,
        fitness_target: float = 2.0,
        stability_target: float = 0.4,
        n_traps: int = 8,
        n_synergies: int = 5,
    ):
        super().__init__(env_name="protein", max_steps=max_steps)
        self._seq_length = seq_length
        self._noise_std = noise_std
        self._fitness_target = fitness_target
        self._stability_target = stability_target
        self._n_traps = n_traps
        self._n_synergies = n_synergies

        self._last_raw_action: Optional[str] = None
        self._wild_type: str = ""
        self._effects: Dict[Tuple[int, str], MutationEffect] = {}
        self._synergies: List[SynergyPair] = []

    # ── helpers ──────────────────────────────────────────────

    def _stability_label(self, stab: float) -> str:
        if stab >= 0.85:
            return "stable"
        elif stab >= 0.65:
            return "mild risk"
        elif stab >= 0.45:
            return "concerning"
        else:
            return "critical"

    def _get_available_edits(self) -> Dict[int, List[Tuple[str, float, str]]]:
        """For each editable position, return available (aa, noisy_estimate, stability_hint).

        Only includes positions not yet edited and mutations meeting stability requirement.
        """
        edited_positions = self.state.latent["edited_positions"]
        stability = self.state.latent["true_stability"]
        sequence = list(self.state.structure)

        available: Dict[int, List[Tuple[str, float, str]]] = {}

        for (pos, aa), eff in self._effects.items():
            if pos in edited_positions:
                continue
            if sequence[pos] == aa:
                continue  # already this aa (from wild type)
            if stability < eff.requires_stability:
                continue

            # Noisy estimate
            noisy = eff.true_fitness_delta + eff.noisy_fitness_bias + float(
                self.rng.normal(0, self._noise_std)
            )

            # Stability hint
            if eff.is_trap:
                # Misleading hints for traps
                roll = float(self.rng.random())
                if roll < 0.5:
                    hint = "neutral"
                elif roll < 0.8:
                    hint = "mild risk"
                else:
                    hint = "stabilizing"
            else:
                sd = eff.true_stability_delta
                if sd < -0.12:
                    hint = "destabilizing"
                elif sd < -0.04:
                    hint = "mild risk"
                elif sd > 0.02:
                    hint = "stabilizing"
                else:
                    hint = "neutral"

            available.setdefault(pos, []).append((aa, round(noisy, 3), hint))

        return available

    def _count_remaining_synergies(self) -> int:
        applied = self.state.latent.get("applied_edits", [])
        applied_set = set(applied)
        count = 0
        for sp in self._synergies:
            b_key = (sp.pos_b, sp.aa_b)
            if b_key not in applied_set:
                count += 1
        return count

    # ── core interface ────────────────────────────────────────

    def reset(self, seed: int, **kwargs) -> str:
        super().reset(seed)

        # Generate landscape
        self._wild_type, self._effects, self._synergies = _generate_landscape(
            self.rng,
            seq_length=self._seq_length,
            n_traps=self._n_traps,
            n_synergies=self._n_synergies,
        )

        self.state = EditState(
            t=0,
            budget_left=self.max_steps,
            structure=self._wild_type,  # current sequence as string
            history=[],
            latent={
                "true_cumulative_fitness": 0.0,
                "true_stability": 1.0,
                "noisy_cumulative": 0.0,
                "edited_positions": set(),
                "applied_edits": [],  # list of (pos, aa) tuples
                "synergies_earned": 0.0,
                "synergies_lost": 0,
                "trap_count": 0,
            },
            done=False,
        )
        self._last_raw_action = None
        return self.render_observation()

    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        import re
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
            fitness = self.state.latent["true_cumulative_fitness"]
            stability = self.state.latent["true_stability"]
            if fitness < self._fitness_target or stability < self._stability_target:
                events.append(FailureEvent.PREMATURE_FINALIZE)
                return False, events
            return True, events

        # type == "edit"
        pos = parsed["position"]
        aa = parsed["amino_acid"]
        sequence = self.state.structure

        # Position range
        if pos < 0 or pos >= len(sequence):
            events.append(FailureEvent.INVALID_POSITION)
            return False, events

        # Valid amino acid
        if aa not in AMINO_ACIDS:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events

        # Same residue
        if sequence[pos] == aa:
            events.append(FailureEvent.ILLEGAL_EDIT)
            return False, events

        # Position already edited
        if pos in self.state.latent["edited_positions"]:
            events.append(FailureEvent.OSCILLATION)
            return False, events

        # Check if this mutation exists in landscape
        if (pos, aa) not in self._effects:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events

        # Stability gate
        eff = self._effects[(pos, aa)]
        stability = self.state.latent["true_stability"]
        if stability < eff.requires_stability:
            events.append(FailureEvent.HARD_CONSTRAINT_VIOLATION)
            return False, events

        return True, events

    def apply_edit(self, parsed: Dict[str, Any]) -> List[FailureEvent]:
        events: List[FailureEvent] = []
        pos = parsed["position"]
        aa = parsed["amino_acid"]
        eff = self._effects[(pos, aa)]

        old_seq = self.state.structure
        seq = list(old_seq)
        old_aa = seq[pos]
        seq[pos] = aa
        self.state.structure = "".join(seq)

        # Update latent state
        lat = self.state.latent
        prev_fitness = lat["true_cumulative_fitness"]
        prev_stability = lat["true_stability"]

        # Fitness gain
        fitness_gain = eff.true_fitness_delta

        # Synergy check
        synergy_bonus = 0.0
        for sp in self._synergies:
            if sp.pos_b == pos and sp.aa_b == aa:
                # Check if pos_a was already edited with aa_a
                if (sp.pos_a, sp.aa_a) in [(p, a) for p, a in lat["applied_edits"]]:
                    synergy_bonus += sp.bonus

        # Lost synergy check (editing B before A)
        lost = 0
        for sp in self._synergies:
            if sp.pos_a == pos and sp.aa_a == aa:
                # We're editing A; if B was already done, synergy is lost
                if (sp.pos_b, sp.aa_b) in [(p, a) for p, a in lat["applied_edits"]]:
                    lost += 1

        total_gain = fitness_gain + synergy_bonus
        lat["true_cumulative_fitness"] = round(prev_fitness + total_gain, 4)
        lat["true_stability"] = round(prev_stability + eff.true_stability_delta, 4)

        # Noisy observation
        noisy_delta = eff.true_fitness_delta + eff.noisy_fitness_bias + float(
            self.rng.normal(0, self._noise_std)
        )
        lat["noisy_cumulative"] = round(lat["noisy_cumulative"] + noisy_delta, 4)

        # Track
        lat["edited_positions"].add(pos)
        lat["applied_edits"].append((pos, aa))
        lat["synergies_earned"] = round(lat["synergies_earned"] + synergy_bonus, 4)
        lat["synergies_lost"] += lost

        if eff.is_trap:
            lat["trap_count"] += 1
            events.append(FailureEvent.LOCAL_OPTIMUM_TRAP)

        # Detect motif break: if synergy was lost
        if lost > 0:
            events.append(FailureEvent.SYNERGY_MISSED)

        # Stability check
        new_stab = lat["true_stability"]
        if new_stab < self._stability_target:
            events.append(FailureEvent.HARD_CONSTRAINT_VIOLATION)

        # Stability gate crossed (was >= 0.6, now < 0.6)
        if prev_stability >= 0.6 and new_stab < 0.6:
            events.append(FailureEvent.GLOBAL_FEASIBILITY_LOSS)

        # Repeated exact edit detection
        if self._last_raw_action is not None:
            last_parsed = self.parse_action(self._last_raw_action)
            if (last_parsed.get("type") == "edit"
                    and last_parsed.get("position") == pos
                    and last_parsed.get("amino_acid") == aa):
                events.append(FailureEvent.REPEATED_EXACT_EDIT)

        # Record history
        self.state.history.append({
            "t": self.state.t,
            "edit": f"pos {pos}: {old_aa}->{aa}",
            "position": pos,
            "old_aa": old_aa,
            "new_aa": aa,
            "noisy_delta": round(noisy_delta, 3),
            "true_delta": eff.true_fitness_delta,
            "stability_label": self._stability_label(new_stab),
            "was_trap": eff.is_trap,
            "synergy_bonus": synergy_bonus,
        })

        return events

    def finalize(self) -> Dict[str, Any]:
        lat = self.state.latent
        true_fit = lat["true_cumulative_fitness"]
        true_stab = lat["true_stability"]

        fitness_ok = true_fit >= self._fitness_target
        stability_ok = true_stab >= self._stability_target
        success = fitness_ok and stability_ok

        return {
            "success": success,
            "true_cumulative_fitness": round(true_fit, 3),
            "true_stability": round(true_stab, 3),
            "fitness_ok": fitness_ok,
            "stability_ok": stability_ok,
            "synergies_earned": round(lat["synergies_earned"], 3),
            "synergies_lost": lat["synergies_lost"],
            "trap_count": lat["trap_count"],
            "mutations_applied": len(lat["applied_edits"]),
            "final_sequence": self.state.structure,
        }

    def step(self, raw_action: str) -> StepResult:
        structure_before = self.state.structure
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
            edit_events = self.apply_edit(parsed)
            events.extend(edit_events)

            # Budget check
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
        sequence = self.state.structure
        lat = self.state.latent
        noisy_cum = lat["noisy_cumulative"]
        true_stab = lat["true_stability"]
        stab_label = self._stability_label(true_stab)
        n_applied = len(lat["applied_edits"])

        # Add noise to displayed cumulative
        displayed_cum = noisy_cum + float(self.rng.normal(0, self._noise_std * 0.3))

        # Format sequence with position markers
        seq_line = " ".join(sequence)
        pos_line = " ".join(f"{i}" if i < 10 else str(i) for i in range(len(sequence)))

        # Mark edited positions
        edited = lat["edited_positions"]
        edit_markers = " ".join(
            "*" if i in edited else " "
            for i in range(len(sequence))
        )

        lines = [
            f"Step {self.state.t + 1}/{self.max_steps}  |  "
            f"Budget remaining: {self.state.budget_left}",
            "",
            f"Sequence:  {seq_line}",
            f"Position:  {pos_line}",
            f"Edited:    {edit_markers}",
            "",
            f"Estimated cumulative fitness: {displayed_cum:+.3f}  "
            f"(target: >= {self._fitness_target:.1f})",
            f"Stability: {stab_label}  "
            f"(must stay above threshold for premium mutations)",
            f"Mutations applied: {n_applied}",
            "",
        ]

        # Synergy hints
        remaining_syn = self._count_remaining_synergies()
        if remaining_syn > 0:
            lines.append(
                f"[Synergy opportunities: ~{remaining_syn} mutation pairs "
                f"may give bonus if applied in correct order]"
            )
            lines.append("")

        # Available edits per position (show top options)
        available = self._get_available_edits()
        if available:
            lines.append("Editable positions and options (est. fitness, stability hint):")
            for pos in sorted(available.keys()):
                opts = available[pos]
                # Sort by noisy estimate descending, show top 4
                opts.sort(key=lambda x: x[1], reverse=True)
                shown = opts[:4]
                opt_strs = [
                    f"{aa}({est:+.2f},{hint})"
                    for aa, est, hint in shown
                ]
                if len(opts) > 4:
                    opt_strs.append(f"+{len(opts)-4} more")
                lines.append(f"  {pos}: {{{', '.join(opt_strs)}}}")
        else:
            lines.append("No available edits (all positions edited or stability too low).")

        # Recent history
        lines.append("")
        if lat.get("applied_edits"):
            lines.append("Recent edits:")
            for h in self.state.history[-4:]:
                syn_str = f" [+synergy {h['synergy_bonus']:.2f}]" if h['synergy_bonus'] > 0 else ""
                lines.append(
                    f"  Step {h['t']}: {h['edit']}  "
                    f"observed: {h['noisy_delta']:+.3f}  "
                    f"stability: {h['stability_label']}{syn_str}"
                )
        else:
            lines.append("No edits applied yet.")

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
        return {
            "success": self._success,
            "steps_taken": self.state.t,
            "max_steps": self.max_steps,
            "true_cumulative_fitness": lat["true_cumulative_fitness"],
            "true_stability": lat["true_stability"],
            "fitness_ok": lat["true_cumulative_fitness"] >= self._fitness_target,
            "stability_ok": lat["true_stability"] >= self._stability_target,
            "mutations_applied": len(lat["applied_edits"]),
            "synergies_earned": lat["synergies_earned"],
            "synergies_lost": lat["synergies_lost"],
            "trap_count": lat["trap_count"],
            "edited_positions": len(lat["edited_positions"]),
            "violations_by_type": dict(violation_by_type),
            "total_violations": len(all_events),
            "events": [
                e.name if isinstance(e, FailureEvent) else str(e)
                for e in all_events
            ],
        }
