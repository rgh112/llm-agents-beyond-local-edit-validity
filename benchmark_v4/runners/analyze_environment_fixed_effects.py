#!/usr/bin/env python3
"""Fixed-effect logistic robustness check for environment separation.

The matched-cell bootstrap asks whether surface-clean success gaps recur over
model x prompt-class cells. This companion audit fits a grouped-binomial
logistic model with environment indicators while adjusting for model and
prompt-class fixed effects. It is intentionally a small robustness check, not a
full mixed-effects model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


ENVS = ("word_ladder", "alloy", "gb1_sequence")
ENV_LABELS = {
    "word_ladder": "Word Ladder",
    "alloy": "Alloy-like",
    "gb1_sequence": "GB1",
}
PROMPT_CLASSES = ("zero_shot", "few_shot_format", "structural")


def prompt_class(prompt: str) -> str:
    if prompt in {"zero_shot", "few_shot_format"}:
        return prompt
    if prompt in {"scaffold", "self_check", "few_shot_strategy"}:
        return "structural"
    return prompt


def pct(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{100.0 * x:.1f}"


def fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        rows.extend(data.get("results", []))
    return [r for r in rows if r.get("env") in ENVS]


def build_matched_observations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        pc = prompt_class(str(row["prompt"]))
        if pc not in PROMPT_CLASSES:
            continue
        by_key[(str(row["model"]), pc, str(row["env"]))] = row

    observations: List[Dict[str, Any]] = []
    models = sorted({key[0] for key in by_key})
    for model in models:
        for pc in PROMPT_CLASSES:
            if not all((model, pc, env) in by_key for env in ENVS):
                continue
            cell_id = f"{model}|{pc}"
            for env in ENVS:
                row = by_key[(model, pc, env)]
                n_clean = int(row.get("n_surface_clean_episodes", 0))
                if n_clean <= 0:
                    continue
                observations.append(
                    {
                        "cell_id": cell_id,
                        "model": model,
                        "prompt_class": pc,
                        "env": env,
                        "n": n_clean,
                        "y": int(row.get("n_surface_clean_successes", 0)),
                    }
                )
    return observations


def design_matrix(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    models = sorted({str(obs["model"]) for obs in observations})
    prompt_classes = [pc for pc in PROMPT_CLASSES if pc in {obs["prompt_class"] for obs in observations}]

    columns = ["intercept", "env=alloy", "env=gb1_sequence"]
    columns.extend(f"model={model}" for model in models[1:])
    columns.extend(f"prompt_class={pc}" for pc in prompt_classes if pc != "zero_shot")

    x_rows: List[List[float]] = []
    y: List[int] = []
    n: List[int] = []
    for obs in observations:
        row = [1.0]
        row.append(1.0 if obs["env"] == "alloy" else 0.0)
        row.append(1.0 if obs["env"] == "gb1_sequence" else 0.0)
        row.extend(1.0 if obs["model"] == model else 0.0 for model in models[1:])
        row.extend(
            1.0 if obs["prompt_class"] == pc else 0.0
            for pc in prompt_classes
            if pc != "zero_shot"
        )
        x_rows.append(row)
        y.append(int(obs["y"]))
        n.append(int(obs["n"]))
    return (
        np.asarray(x_rows, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(n, dtype=float),
        columns,
        models,
        prompt_classes,
    )


def fit_grouped_logit(
    x: np.ndarray,
    y: np.ndarray,
    n: np.ndarray,
    clusters: Sequence[str],
) -> Dict[str, Any]:
    def nll(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(-np.sum(y * eta - n * np.logaddexp(0.0, eta)))

    def grad(beta: np.ndarray) -> np.ndarray:
        p = expit(x @ beta)
        return -(x.T @ (y - n * p))

    result = minimize(nll, np.zeros(x.shape[1]), jac=grad, method="BFGS")
    if not result.success:
        result = minimize(nll, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B")
    if not result.success:
        raise SystemExit(f"Fixed-effect logistic fit failed: {result.message}")

    beta = np.asarray(result.x, dtype=float)
    p = expit(x @ beta)
    weights = n * p * (1.0 - p)
    hessian = x.T @ (weights[:, None] * x)
    cov_model = np.linalg.pinv(hessian)

    score_rows = x * (y - n * p)[:, None]
    meat = np.zeros_like(cov_model)
    cluster_ids = sorted(set(clusters))
    for cluster_id in cluster_ids:
        score = np.sum(
            score_rows[[i for i, cid in enumerate(clusters) if cid == cluster_id]],
            axis=0,
        )
        meat += np.outer(score, score)
    n_obs = x.shape[0]
    k = x.shape[1]
    g = len(cluster_ids)
    correction = (g / (g - 1.0)) * ((n_obs - 1.0) / (n_obs - k)) if g > 1 and n_obs > k else 1.0
    cov_cluster = correction * cov_model @ meat @ cov_model

    return {
        "beta": beta,
        "p": p,
        "cov_model": cov_model,
        "cov_cluster": cov_cluster,
        "nll": nll(beta),
        "converged": bool(result.success),
        "optimizer_message": str(result.message),
    }


def coefficient_rows(columns: Sequence[str], fit: Dict[str, Any]) -> List[Dict[str, Any]]:
    beta = fit["beta"]
    cov_model = fit["cov_model"]
    cov_cluster = fit["cov_cluster"]
    rows = []
    for idx, name in enumerate(columns):
        model_se = float(math.sqrt(max(cov_model[idx, idx], 0.0)))
        cluster_se = float(math.sqrt(max(cov_cluster[idx, idx], 0.0)))
        se = cluster_se if math.isfinite(cluster_se) and cluster_se > 0 else model_se
        lo = float(beta[idx] - 1.96 * se)
        hi = float(beta[idx] + 1.96 * se)
        rows.append(
            {
                "term": name,
                "coef": float(beta[idx]),
                "model_se": model_se,
                "cluster_se": cluster_se,
                "cluster_95_ci": [lo, hi],
                "odds_ratio": float(math.exp(beta[idx])),
                "odds_ratio_cluster_95_ci": [float(math.exp(lo)), float(math.exp(hi))],
                "wald_z_cluster": float(beta[idx] / se) if se > 0 else None,
                "wald_p_cluster": float(2.0 * (1.0 - norm.cdf(abs(beta[idx] / se)))) if se > 0 else None,
            }
        )
    return rows


def contrast_row(
    name: str,
    vector: np.ndarray,
    fit: Dict[str, Any],
) -> Dict[str, Any]:
    beta = fit["beta"]
    cov_model = fit["cov_model"]
    cov_cluster = fit["cov_cluster"]
    coef = float(vector @ beta)
    model_se = float(math.sqrt(max(float(vector @ cov_model @ vector), 0.0)))
    cluster_se = float(math.sqrt(max(float(vector @ cov_cluster @ vector), 0.0)))
    se = cluster_se if math.isfinite(cluster_se) and cluster_se > 0 else model_se
    lo = coef - 1.96 * se
    hi = coef + 1.96 * se
    return {
        "contrast": name,
        "log_odds": coef,
        "model_se": model_se,
        "cluster_se": cluster_se,
        "cluster_95_ci": [lo, hi],
        "odds_ratio": float(math.exp(coef)),
        "odds_ratio_cluster_95_ci": [float(math.exp(lo)), float(math.exp(hi))],
        "wald_z_cluster": float(coef / se) if se > 0 else None,
        "wald_p_cluster": float(2.0 * (1.0 - norm.cdf(abs(coef / se)))) if se > 0 else None,
    }


def make_design_row(
    env: str,
    model: str,
    prompt_class_name: str,
    models: Sequence[str],
    prompt_classes: Sequence[str],
) -> np.ndarray:
    row = [1.0, 1.0 if env == "alloy" else 0.0, 1.0 if env == "gb1_sequence" else 0.0]
    row.extend(1.0 if model == m else 0.0 for m in models[1:])
    row.extend(
        1.0 if prompt_class_name == pc else 0.0
        for pc in prompt_classes
        if pc != "zero_shot"
    )
    return np.asarray(row, dtype=float)


def adjusted_environment_rates(
    observations: Sequence[Dict[str, Any]],
    fit: Dict[str, Any],
    models: Sequence[str],
    prompt_classes: Sequence[str],
) -> List[Dict[str, Any]]:
    cells = sorted({(obs["model"], obs["prompt_class"]) for obs in observations})
    out = []
    beta = fit["beta"]
    for env in ENVS:
        probs = [
            float(expit(make_design_row(env, model, pc, models, prompt_classes) @ beta))
            for model, pc in cells
        ]
        out.append(
            {
                "env": env,
                "n_cells": len(cells),
                "adjusted_mean_probability": float(np.mean(probs)),
                "min_cell_probability": float(np.min(probs)),
                "max_cell_probability": float(np.max(probs)),
            }
        )
    return out


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Fixed-Effect Logistic Environment-Separation Audit",
        "",
        (
            "Grouped-binomial logistic regression on matched model x prompt-class "
            "cells. The outcome is surface-clean success among surface-clean "
            "episodes. Predictors are environment indicators plus model and "
            "prompt-class fixed effects; uncertainty uses a cluster-robust "
            "sandwich estimate over matched cells. This is a robustness check, "
            "not a full mixed-effects model."
        ),
        "",
        "## Adjusted Environment Rates",
        "",
        "| Environment | Cells | Adjusted mean surface-clean success | Cell range |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in result["adjusted_environment_rates"]:
        lines.append(
            "| {env} | {cells} | {mean}% | [{lo}, {hi}]% |".format(
                env=ENV_LABELS[row["env"]],
                cells=row["n_cells"],
                mean=pct(row["adjusted_mean_probability"]),
                lo=pct(row["min_cell_probability"]),
                hi=pct(row["max_cell_probability"]),
            )
        )
    lines.extend(
        [
            "",
            "## Adjusted Odds-Ratio Contrasts",
            "",
            "| Contrast | Log-odds | Odds ratio | Cluster 95% CI | p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["contrasts"]:
        lo, hi = row["odds_ratio_cluster_95_ci"]
        lines.append(
            "| {name} | {log_odds} | {or_} | [{lo}, {hi}] | {p} |".format(
                name=row["contrast"],
                log_odds=fmt_float(row["log_odds"]),
                or_=fmt_float(row["odds_ratio"]),
                lo=fmt_float(lo),
                hi=fmt_float(hi),
                p=fmt_float(row["wald_p_cluster"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="cross_model_summary.json files")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    rows = load_rows(args.summaries)
    observations = build_matched_observations(rows)
    if len(observations) != 54:
        raise SystemExit(f"Expected 54 matched observations, found {len(observations)}")
    x, y, n, columns, models, prompt_classes = design_matrix(observations)
    clusters = [str(obs["cell_id"]) for obs in observations]
    fit = fit_grouped_logit(x, y, n, clusters)

    contrast_vectors: List[Tuple[str, np.ndarray]] = []
    alloy = np.zeros(len(columns))
    alloy[columns.index("env=alloy")] = 1.0
    gb1 = np.zeros(len(columns))
    gb1[columns.index("env=gb1_sequence")] = 1.0
    contrast_vectors.append(("Alloy-like vs Word Ladder", alloy))
    contrast_vectors.append(("GB1 vs Word Ladder", gb1))
    contrast_vectors.append(("Alloy-like vs GB1", alloy - gb1))

    result: Dict[str, Any] = {
        "note": (
            "Grouped-binomial fixed-effect logistic robustness check over "
            "matched model x prompt-class cells; not a full mixed-effects model."
        ),
        "n_rows_loaded": len(rows),
        "n_observations": len(observations),
        "n_cells": len(set(clusters)),
        "outcome": "surface_clean_success_given_surface_clean",
        "columns": columns,
        "models": models,
        "prompt_classes": prompt_classes,
        "model_log_likelihood": -float(fit["nll"]),
        "coefficients": coefficient_rows(columns, fit),
        "contrasts": [contrast_row(name, vector, fit) for name, vector in contrast_vectors],
        "adjusted_environment_rates": adjusted_environment_rates(
            observations, fit, models, prompt_classes
        ),
        "observations": observations,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(result, Path(args.output_md))
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))
    print(f"Saved to {out_json}")
    print(f"Saved to {args.output_md}")


if __name__ == "__main__":
    main()
