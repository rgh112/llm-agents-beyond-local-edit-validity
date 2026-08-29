"""
Alloy Composition — Constructive Editing Environment

The agent edits the alloy composition one swap at a time to reach target properties.
Each edit transfers atomic percent from one element to another.

Action format: EDIT <inc_element> <dec_element> <delta>
             or: FINALIZE (triggers full evaluation)

Constructive editing properties:
  C1. Edit locality — one element-pair swap per step
  C2. Path dependence — recovery cost accumulates from inefficient paths
  C3. Partial-state semantics — composition constrains available moves
  C4. Delayed evaluation — true UTS and recovery cost revealed only on FINALIZE
  C5. Order sensitivity — coarse/fine phases limit available deltas
"""
from __future__ import annotations

import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv, EditState, StepResult
from benchmark_v4.failure_taxonomy import FailureEvent, StepRecord

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_SKLEARN = True
except ImportError:
    GradientBoostingRegressor = None  # type: ignore[assignment,misc]
    _HAS_SKLEARN = False

try:
    from pymatgen.core import Composition as PymatgenComposition
    _HAS_PYMATGEN = True
except ImportError:
    PymatgenComposition = None  # type: ignore[assignment,misc]
    _HAS_PYMATGEN = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def _find_project_root() -> pathlib.Path:
    """Find the repository root from either live or supplementary code paths."""
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "alloy_data" / "mpea_mech.csv").exists():
            return candidate
    return pathlib.Path(__file__).resolve().parent.parent.parent


_PROJECT_ROOT = _find_project_root()
_DEFAULT_CSV = _PROJECT_ROOT / "alloy_data" / "mpea_mech.csv"

ELEMENTS: List[str] = ["Fe", "Ni", "Cr", "Co", "Mn", "Mo"]

DENSITY_TABLE: Dict[str, float] = {
    "Fe": 7.87, "Ni": 8.90, "Cr": 7.20,
    "Co": 8.90, "Mn": 7.40, "Mo": 10.20,
}

DEFAULT_UTS_TARGET: float = 2000.0   # MPa
DEFAULT_DENSITY_TARGET: float = 8.20  # g/cm^3

# Multiplicative UTS scaling from heavy elements — creates genuine UTS-density tradeoff.
# Without heavy elements, UTS is capped at floor_fraction of base.
# Need heavy_pct >= heavy_threshold for full UTS.
# This forces the agent to include some Mo/Co (heavy, density-increasing) to reach UTS target.
DEFAULT_UTS_HEAVY_FLOOR: float = 0.30       # UTS floor fraction without heavy elements
DEFAULT_UTS_HEAVY_THRESHOLD: float = 18.0   # weighted heavy% needed for full UTS
DEFAULT_UTS_HEAVY_WEIGHTS: Dict[str, float] = {
    "Mo": 1.0,    # Mo fully counts toward heavy requirement
    "Co": 0.6,    # Co partially counts
    "Ni": 0.3,    # Ni slightly counts
}

# Per-element concentration constraints [min%, max%]
ELEMENT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Fe": (0.0, 40.0),
    "Ni": (0.0, 35.0),
    "Cr": (0.0, 40.0),
    "Co": (0.0, 35.0),
    "Mn": (0.0, 35.0),
    "Mo": (0.0, 30.0),  # Mo capped lower — very heavy
}

DEFAULT_START_COMPOSITION: Dict[str, float] = {
    "Fe": 25.0, "Ni": 15.0, "Cr": 20.0,
    "Co": 10.0, "Mn": 20.0, "Mo": 10.0,
}

# Starting composition jitter: each element perturbed by ±jitter%
# This creates seed-dependent starting conditions so greedy baselines
# don't always follow the exact same path → intermediate success rates.
DEFAULT_START_JITTER: float = 8.0

# Known-good composition region center (UTS > 2000, density < 8.5)
_GOOD_REGION_CENTER: Dict[str, float] = {
    "Fe": 20.0, "Ni": 20.0, "Cr": 25.0,
    "Co": 10.0, "Mn": 10.0, "Mo": 15.0,
}

_RECOVERY_COST_RATE: float = 0.22
_RECOVERY_COST_THRESHOLD: float = 3.8

# Phase-dependent allowed deltas
COARSE_DELTAS: List[int] = [10, 15]
FINE_DELTAS: List[int] = [2, 5]


def _parse_formula(formula: str) -> Dict[str, float]:
    """Parse a chemical formula string into {element: atomic_percent}."""
    if _HAS_PYMATGEN:
        try:
            comp = PymatgenComposition(formula).fractional_composition
            return {
                str(el): frac * 100.0
                for el, frac in comp.get_el_amt_dict().items()
            }
        except Exception:
            pass
    result: Dict[str, float] = {}
    for match in re.finditer(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        elem, num = match.groups()
        result[elem] = float(num) if num else 1.0
    total = sum(result.values())
    if total > 0:
        return {k: v / total * 100.0 for k, v in result.items()}
    return result


class AlloyEnv(BaseConstructiveEditEnv):
    """Constructive alloy composition editing: swap elements to reach targets.

    The agent decides WHICH elements to swap and by HOW MUCH.
    Must satisfy UTS and density targets while keeping recovery cost low.
    """

    # ── prompt-builder properties ─────────────────────────────

    @property
    def task_description(self) -> str:
        return (
            "You are editing an alloy composition to meet performance targets. "
            f"Your goal: UTS >= {self._uts_target:.0f} MPa AND "
            f"density <= {self._density_target:.1f} g/cm^3. "
            f"Elements: {', '.join(ELEMENTS)}. "
            "Each edit transfers atomic percent from one element to another. "
            "The path you take matters — inefficient paths accumulate drift "
            "that can cause failure even if targets appear met."
        )

    @property
    def constraints(self) -> str:
        return (
            "1. Each step: EDIT <inc_element> <dec_element> <delta> "
            "(transfers delta% from dec to inc).\n"
            "2. Delta must match allowed values for the current phase.\n"
            "3. All element percentages must stay in [0, 100].\n"
            "4. Total composition always sums to 100%.\n"
            "5. Coarse phase (first half): deltas {10, 15}.\n"
            "6. Fine phase (second half): deltas {2, 5}.\n"
            "7. FINALIZE triggers full evaluation — use when you believe targets are met."
        )

    @property
    def action_format(self) -> str:
        return (
            "EDIT <inc_element> <dec_element> <delta>\n"
            "Example: EDIT Ni Mn 10\n"
            "Or: FINALIZE"
        )

    # ── construction ──────────────────────────────────────────

    def __init__(
        self,
        csv_path: str | pathlib.Path | None = None,
        max_steps: int = 8,
        uts_target: float = DEFAULT_UTS_TARGET,
        density_target: float = DEFAULT_DENSITY_TARGET,
        uts_noise_std: float = 30.0,
        element_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        rc_rate: Optional[float] = None,
        rc_threshold: Optional[float] = None,
        start_jitter: float = DEFAULT_START_JITTER,
        heavy_floor: float = DEFAULT_UTS_HEAVY_FLOOR,
        heavy_threshold: float = DEFAULT_UTS_HEAVY_THRESHOLD,
        heavy_weights: Optional[Dict[str, float]] = None,
        overshoot_penalty: float = 0.0,
    ):
        super().__init__(env_name="alloy", max_steps=max_steps)
        self._uts_target = uts_target
        self._density_target = density_target
        self._uts_noise_std = uts_noise_std
        self._element_bounds = element_bounds if element_bounds is not None else dict(ELEMENT_BOUNDS)
        self._rc_rate = rc_rate if rc_rate is not None else _RECOVERY_COST_RATE
        self._rc_threshold = rc_threshold if rc_threshold is not None else _RECOVERY_COST_THRESHOLD
        self._start_jitter = start_jitter
        self._heavy_floor = heavy_floor
        self._heavy_threshold = heavy_threshold
        self._heavy_weights = heavy_weights if heavy_weights is not None else dict(DEFAULT_UTS_HEAVY_WEIGHTS)
        self._overshoot_penalty = overshoot_penalty  # UTS multiplier per density overshoot
        self._recoverability_probe_depth = 3
        self._recoverability_beam_width = 128

        self._csv_path = pathlib.Path(csv_path) if csv_path else _DEFAULT_CSV
        self._uts_model = self._train_surrogate()

        self._last_raw_action: Optional[str] = None

    # ── surrogate model ──────────────────────────────────────

    def _train_surrogate(self):
        if not _HAS_PANDAS:
            raise ImportError("pandas required: pip install pandas")
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        if not self._csv_path.exists():
            raise FileNotFoundError(f"MPEA data not found: {self._csv_path}")

        df = pd.read_csv(self._csv_path).fillna(0.0)
        col_map = {
            "PROPERTY: UTS (MPa)": "UTS",
            "PROPERTY: Exp. Density (g/cm$^3$)": "density_exp",
            "PROPERTY: Calculated Density (g/cm$^3$)": "density_calc",
            "FORMULA": "formula",
        }
        df = df.rename(columns=col_map)

        if "density_exp" in df.columns:
            df["density_exp"] = df["density_exp"].replace(0, np.nan)
            df["density"] = df["density_exp"].fillna(df.get("density_calc", 0))
        elif "density_calc" in df.columns:
            df["density"] = df["density_calc"]
        else:
            df["density"] = 0.0

        df = df.dropna(subset=["UTS"])
        df = df[df["UTS"] > 0]

        for el in ELEMENTS:
            df[el] = 0.0
        for idx, row in df.iterrows():
            parsed = _parse_formula(str(row.get("formula", "")))
            for el, pct in parsed.items():
                if el in ELEMENTS:
                    df.at[idx, el] = pct

        X = df[ELEMENTS].values
        y = df["UTS"].values

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=0,
        )
        model.fit(X, y)
        return model

    # ── physics helpers ──────────────────────────────────────

    def _predict_uts(self, comp: Dict[str, float]) -> float:
        x = np.array([[comp.get(el, 0.0) for el in ELEMENTS]])
        base = float(self._uts_model.predict(x)[0])
        # Multiplicative scaling: need heavy elements for full UTS
        heavy_pct = sum(
            comp.get(el, 0.0) * w
            for el, w in self._heavy_weights.items()
        )
        # Scale from floor to 1.0 as heavy_pct goes from 0 to threshold
        scale = self._heavy_floor + (1.0 - self._heavy_floor) * min(1.0, heavy_pct / self._heavy_threshold)
        return base * scale

    def _predict_uts_noisy(self, comp: Dict[str, float]) -> float:
        base = self._predict_uts(comp)
        noise = float(self.rng.normal(0, self._uts_noise_std))
        return base + noise

    @staticmethod
    def _calculate_density(comp: Dict[str, float]) -> float:
        return sum(DENSITY_TABLE[el] * comp.get(el, 0.0) / 100.0 for el in ELEMENTS)

    def _distance_from_good_region(self, comp: Dict[str, float]) -> float:
        return sum(
            (comp.get(el, 0.0) - _GOOD_REGION_CENTER.get(el, 0.0)) ** 2
            for el in ELEMENTS
        ) ** 0.5

    def _state_success(self, comp: Dict[str, float], recovery_cost: float,
                       density_trajectory: List[float]) -> bool:
        true_uts = self._predict_uts(comp)
        true_density = self._calculate_density(comp)
        max_density_seen = max(density_trajectory) if density_trajectory else true_density
        overshoot = max(0, max_density_seen - self._density_target)
        effective_uts_target = self._uts_target * (1.0 + overshoot * self._overshoot_penalty)
        return (
            true_uts >= effective_uts_target
            and true_density <= self._density_target
            and recovery_cost < self._rc_threshold
        )

    def _local_proxy_score(self, comp: Dict[str, float], recovery_cost: float) -> float:
        uts_margin = (self._predict_uts(comp) - self._uts_target) / max(self._uts_target, 1.0)
        density_margin = (self._density_target - self._calculate_density(comp)) / max(self._density_target, 1.0)
        rc_margin = (self._rc_threshold - recovery_cost) / max(self._rc_threshold, 1.0)
        return uts_margin + density_margin + 0.25 * rc_margin

    def _phase_at(self, t: int) -> str:
        return "coarse" if t < self.max_steps // 2 else "fine"

    def _allowed_deltas_at(self, t: int) -> List[int]:
        return list(COARSE_DELTAS) if self._phase_at(t) == "coarse" else list(FINE_DELTAS)

    def _simulate_valid_edits(self, comp: Dict[str, float], t: int) -> List[Tuple[str, str, int]]:
        edits: List[Tuple[str, str, int]] = []
        for inc_el in ELEMENTS:
            for dec_el in ELEMENTS:
                if inc_el == dec_el:
                    continue
                for delta in self._allowed_deltas_at(t):
                    new_inc = comp.get(inc_el, 0.0) + delta
                    new_dec = comp.get(dec_el, 0.0) - delta
                    if new_inc > 100.0 or new_dec < 0.0:
                        continue
                    inc_lo, inc_hi = self._element_bounds.get(inc_el, (0.0, 100.0))
                    dec_lo, dec_hi = self._element_bounds.get(dec_el, (0.0, 100.0))
                    if new_inc <= inc_hi and new_dec >= dec_lo:
                        edits.append((inc_el, dec_el, delta))
        return edits

    def _simulate_edit_state(
        self,
        comp: Dict[str, float],
        recovery_cost: float,
        density_trajectory: List[float],
        edit: Tuple[str, str, int],
    ) -> Tuple[Dict[str, float], float, List[float]]:
        inc_el, dec_el, delta = edit
        old_comp = dict(comp)
        new_comp = dict(comp)
        new_comp[inc_el] = new_comp.get(inc_el, 0.0) + delta
        new_comp[dec_el] = new_comp.get(dec_el, 0.0) - delta
        prev_distance = self._distance_from_good_region(old_comp)
        new_distance = self._distance_from_good_region(new_comp)
        rc = recovery_cost
        if new_distance > prev_distance:
            rc += self._rc_rate * (new_distance - prev_distance)
        elif rc > 0:
            rc = max(0, rc - self._rc_rate * (prev_distance - new_distance) * 0.3)
        return new_comp, rc, list(density_trajectory) + [self._calculate_density(new_comp)]

    def _recoverability_probe(
        self,
        comp: Dict[str, float],
        recovery_cost: float,
        density_trajectory: List[float],
        t: int,
        budget_left: int,
    ) -> Dict[str, Any]:
        """Beam-search proxy for future feasibility under remaining edit budget."""
        if self._state_success(comp, recovery_cost, density_trajectory):
            return {"recoverable": True, "best_proxy_score": round(self._local_proxy_score(comp, recovery_cost), 4)}
        frontier = [(self._local_proxy_score(comp, recovery_cost), dict(comp), recovery_cost, list(density_trajectory), t)]
        best_score = frontier[0][0]
        max_depth = min(max(0, budget_left), self._recoverability_probe_depth)
        for _ in range(max_depth):
            candidates = []
            for _, cur_comp, cur_rc, cur_density_traj, cur_t in frontier:
                for edit in self._simulate_valid_edits(cur_comp, cur_t):
                    nxt_comp, nxt_rc, nxt_density_traj = self._simulate_edit_state(
                        cur_comp, cur_rc, cur_density_traj, edit
                    )
                    score = self._local_proxy_score(nxt_comp, nxt_rc)
                    best_score = max(best_score, score)
                    if self._state_success(nxt_comp, nxt_rc, nxt_density_traj):
                        return {"recoverable": True, "best_proxy_score": round(best_score, 4)}
                    candidates.append((score, nxt_comp, nxt_rc, nxt_density_traj, cur_t + 1))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            frontier = candidates[:self._recoverability_beam_width]
        return {"recoverable": False, "best_proxy_score": round(best_score, 4)}

    def _diagnostic_snapshot(self) -> Dict[str, Any]:
        comp = dict(self.state.structure)
        rc = float(self.state.latent["recovery_cost"])
        density_traj = list(self.state.latent.get("density_trajectory", []))
        probe = self._recoverability_probe(
            comp, rc, density_traj, self.state.t, self.state.budget_left
        )
        return {
            "remaining_budget": self.state.budget_left,
            "recoverable": bool(probe["recoverable"]),
            "recoverability_score": float(probe["best_proxy_score"]),
            "local_proxy_score": round(self._local_proxy_score(comp, rc), 4),
            "uts": round(self._predict_uts(comp), 4),
            "density": round(self._calculate_density(comp), 4),
            "recovery_cost": round(rc, 4),
        }

    # ── phase helpers ────────────────────────────────────────

    def _current_phase(self) -> str:
        if self.state.t < self.max_steps // 2:
            return "coarse"
        return "fine"

    def _allowed_deltas(self) -> List[int]:
        if self._current_phase() == "coarse":
            return list(COARSE_DELTAS)
        return list(FINE_DELTAS)

    # ── composition string helper ────────────────────────────

    @staticmethod
    def _comp_str(comp: Dict[str, float]) -> str:
        return "  ".join(f"{el}: {comp.get(el, 0.0):.1f}%" for el in ELEMENTS)

    # ── core interface ────────────────────────────────────────

    def _jittered_start(self) -> Dict[str, float]:
        """Generate a seed-dependent starting composition with jitter."""
        if self._start_jitter <= 0:
            return dict(DEFAULT_START_COMPOSITION)

        base = dict(DEFAULT_START_COMPOSITION)
        # Add random jitter to each element
        perturbations = {el: float(self.rng.uniform(-self._start_jitter, self._start_jitter))
                         for el in ELEMENTS}
        raw = {el: max(0.0, base[el] + perturbations[el]) for el in ELEMENTS}
        # Renormalize to sum to 100
        total = sum(raw.values())
        comp = {el: round(raw[el] / total * 100.0) for el in ELEMENTS}
        # Fix rounding to exactly 100
        diff = 100 - sum(comp.values())
        if diff != 0:
            # Add diff to the element closest to its target
            comp["Fe"] += diff
        # Clamp to element bounds
        for el in ELEMENTS:
            lo, hi = self._element_bounds.get(el, (0.0, 100.0))
            comp[el] = max(lo, min(hi, comp[el]))
        return comp

    def reset(self, seed: int, **kwargs) -> str:
        super().reset(seed)

        start_comp = self._jittered_start()
        self.state = EditState(
            t=0,
            budget_left=self.max_steps,
            structure=start_comp,
            history=[],
            latent={
                "recovery_cost": 0.0,
                "density_trajectory": [self._calculate_density(start_comp)],
                "composition_trajectory": [dict(start_comp)],
                "bad_moves_count": 0,
            },
            done=False,
        )
        self._last_raw_action = None
        return self.render_observation()

    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        text = raw_action.strip()

        if re.match(r"(?i)^finalize\b", text):
            return {"type": "finalize"}

        # EDIT <inc_el> <dec_el> <delta>
        m = re.match(
            r"(?i)^edit\s+([A-Za-z]{1,2})\s+([A-Za-z]{1,2})\s+(\d+)$", text
        )
        if m:
            inc_el = m.group(1).capitalize()
            dec_el = m.group(2).capitalize()
            delta = int(m.group(3))
            return {
                "type": "edit",
                "inc_element": inc_el,
                "dec_element": dec_el,
                "delta": delta,
            }

        return {"type": "invalid", "raw": text}

    def validate_action(self, parsed: Dict[str, Any]) -> Tuple[bool, List[FailureEvent]]:
        events: List[FailureEvent] = []

        if parsed["type"] == "invalid":
            events.append(FailureEvent.MALFORMED_ACTION)
            return False, events

        if parsed["type"] == "finalize":
            # Check if targets are plausibly met
            comp = self.state.structure
            uts = self._predict_uts(comp)
            density = self._calculate_density(comp)
            if uts < self._uts_target or density > self._density_target:
                events.append(FailureEvent.PREMATURE_FINALIZE)
                return False, events
            return True, events

        # type == "edit"
        inc_el = parsed["inc_element"]
        dec_el = parsed["dec_element"]
        delta = parsed["delta"]
        comp = self.state.structure

        # Element validity
        if inc_el not in ELEMENTS:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events
        if dec_el not in ELEMENTS:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events

        # Same element check
        if inc_el == dec_el:
            events.append(FailureEvent.ILLEGAL_EDIT)
            return False, events

        # Delta validity (must match phase-allowed deltas)
        allowed = self._allowed_deltas()
        if delta not in allowed:
            events.append(FailureEvent.INVALID_VALUE)
            return False, events

        # Range check (basic)
        new_inc = comp.get(inc_el, 0.0) + delta
        new_dec = comp.get(dec_el, 0.0) - delta
        if new_inc > 100.0 or new_dec < 0.0:
            events.append(FailureEvent.HARD_CONSTRAINT_VIOLATION)
            return False, events

        # Per-element bounds check
        if self._element_bounds:
            inc_lo, inc_hi = self._element_bounds.get(inc_el, (0.0, 100.0))
            dec_lo, dec_hi = self._element_bounds.get(dec_el, (0.0, 100.0))
            if new_inc > inc_hi or new_dec < dec_lo:
                events.append(FailureEvent.HARD_CONSTRAINT_VIOLATION)
                return False, events

        return True, events

    def apply_edit(self, parsed: Dict[str, Any]) -> List[FailureEvent]:
        events: List[FailureEvent] = []
        comp = self.state.structure
        inc_el = parsed["inc_element"]
        dec_el = parsed["dec_element"]
        delta = parsed["delta"]

        old_comp = dict(comp)
        prev_uts = self._predict_uts(comp)
        prev_density = self._calculate_density(comp)
        prev_distance = self._distance_from_good_region(comp)

        # Apply edit
        comp[inc_el] = comp.get(inc_el, 0.0) + delta
        comp[dec_el] = comp.get(dec_el, 0.0) - delta

        new_uts = self._predict_uts(comp)
        new_density = self._calculate_density(comp)
        new_distance = self._distance_from_good_region(comp)

        # Recovery cost (C2: path dependence)
        rc = self.state.latent["recovery_cost"]
        if new_distance > prev_distance:
            distance_delta = new_distance - prev_distance
            rc += self._rc_rate * distance_delta
        elif rc > 0:
            distance_delta = prev_distance - new_distance
            recovery_amount = self._rc_rate * distance_delta * 0.3
            rc = max(0, rc - recovery_amount)
        self.state.latent["recovery_cost"] = rc

        # Detect oscillation (reversal of previous move)
        if self._last_raw_action is not None:
            last_parsed = self.parse_action(self._last_raw_action)
            if (last_parsed.get("type") == "edit"
                    and last_parsed.get("inc_element") == dec_el
                    and last_parsed.get("dec_element") == inc_el
                    and last_parsed.get("delta") == delta):
                events.append(FailureEvent.OSCILLATION)

        # Detect repeated exact edit
        if self._last_raw_action is not None:
            last_parsed = self.parse_action(self._last_raw_action)
            if (last_parsed.get("type") == "edit"
                    and last_parsed.get("inc_element") == inc_el
                    and last_parsed.get("dec_element") == dec_el
                    and last_parsed.get("delta") == delta):
                events.append(FailureEvent.REPEATED_EXACT_EDIT)

        # Detect objective tradeoff failure
        uts_improved = new_uts > prev_uts
        density_improved = new_density < prev_density
        uts_worsened = new_uts < prev_uts
        density_worsened = new_density > prev_density
        if (uts_improved and density_worsened) or (density_improved and uts_worsened):
            events.append(FailureEvent.OBJECTIVE_TRADEOFF_FAILURE)

        # Detect trap: increasing heavy elements (Mo, Co, Ni) at expense of light
        heavy_elements = {"Mo", "Co", "Ni"}
        light_elements = {"Fe", "Cr", "Mn"}
        if inc_el in heavy_elements and dec_el in light_elements:
            if new_density > self._density_target:
                events.append(FailureEvent.LOCAL_OPTIMUM_TRAP)
                self.state.latent["bad_moves_count"] += 1

        # Recovery cost explosion check
        if rc >= self._rc_threshold:
            events.append(FailureEvent.RECOVERY_COST_EXPLOSION)

        # Record trajectory
        self.state.latent["composition_trajectory"].append(dict(comp))
        self.state.latent["density_trajectory"].append(new_density)

        # Record history
        self.state.history.append({
            "t": self.state.t,
            "edit": f"{inc_el}+{delta}%, {dec_el}-{delta}%",
            "comp_before": old_comp,
            "comp_after": dict(comp),
            "uts_before": prev_uts,
            "uts_after": new_uts,
            "density_before": prev_density,
            "density_after": new_density,
        })

        return events

    def finalize(self) -> Dict[str, Any]:
        comp = self.state.structure
        true_uts = self._predict_uts(comp)
        true_density = self._calculate_density(comp)
        rc = self.state.latent["recovery_cost"]

        # Density overshoot penalty: if density EVER exceeded target during
        # the trajectory, the effective UTS target increases. This rewards
        # strategies that keep density under control throughout the path.
        density_traj = self.state.latent.get("density_trajectory", [])
        max_density_seen = max(density_traj) if density_traj else true_density
        overshoot = max(0, max_density_seen - self._density_target)
        effective_uts_target = self._uts_target * (1.0 + overshoot * self._overshoot_penalty)

        uts_met = true_uts >= effective_uts_target
        density_met = true_density <= self._density_target
        rc_ok = rc < self._rc_threshold
        success = uts_met and density_met and rc_ok

        return {
            "success": success,
            "true_uts": true_uts,
            "true_density": true_density,
            "effective_uts_target": effective_uts_target,
            "uts_met": uts_met,
            "density_met": density_met,
            "recovery_cost": rc,
            "recovery_cost_ok": rc_ok,
            "max_density_seen": max_density_seen,
            "density_overshoot": overshoot,
            "final_composition": dict(comp),
        }

    def step(self, raw_action: str) -> StepResult:
        structure_before = self._comp_str(self.state.structure)
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
            edit_events = self.apply_edit(parsed)
            events.extend(edit_events)

            # Budget check
            if self.state.budget_left <= 0:
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
            structure_before=structure_before,
            structure_after=self._comp_str(self.state.structure),
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
        comp = self.state.structure
        density = self._calculate_density(comp)
        noisy_uts = self._predict_uts_noisy(comp)
        phase = self._current_phase()
        allowed = self._allowed_deltas()
        rc = self.state.latent["recovery_cost"]

        # Density status
        if density <= self._density_target - 0.3:
            density_status = "well within limit"
        elif density <= self._density_target:
            density_status = "near limit"
        else:
            density_status = "ABOVE limit"

        # Path efficiency proxy (observable signal for recovery cost)
        if rc < 0.5:
            path_status = "efficient"
        elif rc < 1.5:
            path_status = "some drift from optimal region"
        elif rc < self._rc_threshold:
            path_status = "significant drift — past moves may limit final feasibility"
        else:
            path_status = "WARNING: heavy drift — even if targets appear met, process penalty may cause failure"

        # Composition display
        comp_line = self._comp_str(comp)

        # Available edits: show which elements can be inc/dec given constraints
        edit_options = []
        for inc in ELEMENTS:
            for dec in ELEMENTS:
                if inc == dec:
                    continue
                for d in allowed:
                    new_inc = comp.get(inc, 0.0) + d
                    new_dec = comp.get(dec, 0.0) - d
                    if new_inc > 100.0 or new_dec < 0.0:
                        continue
                    # Check element bounds
                    if self._element_bounds:
                        inc_lo, inc_hi = self._element_bounds.get(inc, (0.0, 100.0))
                        dec_lo, dec_hi = self._element_bounds.get(dec, (0.0, 100.0))
                        if new_inc > inc_hi or new_dec < dec_lo:
                            continue
                    edit_options.append(f"EDIT {inc} {dec} {d}")
        # Don't show all — just count
        n_valid = len(edit_options)

        # Recent moves
        if self.state.history:
            recent = self.state.history[-3:]
            hist_lines = []
            for h in recent:
                hist_lines.append(
                    f"  Step {h['t']}: {h['edit']}  "
                    f"(UTS: {h['uts_before']:.0f}->{h['uts_after']:.0f}, "
                    f"density: {h['density_before']:.2f}->{h['density_after']:.2f})"
                )
            history_block = "\n".join(hist_lines)
        else:
            history_block = "  (none)"

        lines = [
            f"Step {self.state.t + 1}/{self.max_steps}  |  "
            f"Budget remaining: {self.state.budget_left}  |  Phase: {phase}",
            "",
            f"Composition (at.%):  {comp_line}",
            "",
            "Performance estimates:",
            f"  UTS:     ~{noisy_uts:.0f} MPa   (target: >= {self._uts_target:.0f})",
            f"  Density:  {density:.2f} g/cm^3 (target: <= {self._density_target:.1f}) "
            f"[{density_status}]",
            f"  Path efficiency: {path_status}",
            "",
            f"Phase: {phase}  |  Allowed deltas: {{{', '.join(str(d) for d in allowed)}}}",
            f"Valid edits available: {n_valid}",
            "",
            "Recent moves:",
            history_block,
            "",
            f"Elements: {', '.join(ELEMENTS)}",
            f"Density reference: "
            + ", ".join(f"{el}={DENSITY_TABLE[el]}" for el in ELEMENTS),
            f"Element limits: "
            + ", ".join(
                f"{el}:[{self._element_bounds.get(el, (0,100))[0]:.0f}-{self._element_bounds.get(el, (0,100))[1]:.0f}%]"
                for el in ELEMENTS
            ) if self._element_bounds else "",
        ]

        return "\n".join(lines)

    def get_episode_summary(self) -> Dict[str, Any]:
        all_events: List[FailureEvent] = []
        for rec in self.trajectory:
            all_events.extend(rec.events)

        violation_by_type = Counter(
            e.name if isinstance(e, FailureEvent) else str(e)
            for e in all_events
        )

        comp = self.state.structure
        true_uts = self._predict_uts(comp)
        density = self._calculate_density(comp)
        rc = self.state.latent["recovery_cost"]

        return {
            "success": self._success,
            "steps_taken": self.state.t,
            "max_steps": self.max_steps,
            "final_composition": dict(comp),
            "final_uts": true_uts,
            "final_density": density,
            "uts_met": true_uts >= self._uts_target,
            "density_met": density <= self._density_target,
            "recovery_cost": rc,
            "recovery_cost_ok": rc < self._rc_threshold,
            "bad_moves_count": self.state.latent.get("bad_moves_count", 0),
            "violations_by_type": dict(violation_by_type),
            "total_violations": len(all_events),
            "composition_trajectory": self.state.latent.get("composition_trajectory", []),
            "density_trajectory": self.state.latent.get("density_trajectory", []),
            "events": [
                e.name if isinstance(e, FailureEvent) else str(e)
                for e in all_events
            ],
        }
