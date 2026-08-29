# Evaluating LLM Agents Beyond Local Edit Validity

Official implementation and released experimental results for the paper
*Evaluating LLM Agents Beyond Local Edit Validity*, accepted to the Findings
of the Association for Computational Linguistics: EMNLP 2026.

This repository implements the paper's recoverability-centered evaluation
protocol and three matched constructive-editing environments:

- **Word Ladder:** graph-constrained lexical editing.
- **Alloy-like composition editing:** a controlled composition-optimization
  proxy with coupled objectives and hard constraints.
- **GB1 landscape editing:** a restricted protein-sequence landscape used to
  study recoverability and local-optimum traps.

The scientific-design-inspired environments are controlled diagnostic proxies,
not deployable materials- or protein-design systems. The repository contains
no wet-lab protocols, synthesis instructions, or biological design guidance.

## Repository layout

```text
benchmark_v4/   Environments, prompts, memory modules, controllers, and runners
scripts/        Result aggregation and visualization utilities
results/        Released per-episode logs and machine-readable summaries
figures/        Figures generated from the released results
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS or Linux
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` before installing the requirements.

The package can be imported directly from the repository root:

```bash
python -c "from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv; env = GB1SequenceEnv(); print(env.reset(seed=0).splitlines()[0])"
```

## Data

The restricted 625-state GB1 lookup used by the environment is included in
`benchmark_v4/data/gb1_fitness.json`.

The original lexical and alloy source datasets are not redistributed. To rerun
those environments, obtain the datasets under their original terms and place
the required files at the following paths:

```text
wordladder_data/enable1.txt
wordladder_data/pairs_500_len3.csv
wordladder_data/pairs_500_len4.csv
wordladder_data/pairs_500_len5.csv
alloy_data/mpea_mech.csv
```

`enable1.txt` is the ENABLE word list. `mpea_mech.csv` is the MPEA
measured-property table described and cited in the paper. Please cite the
original data sources when using either environment.

## Running experiments

The main controlled experiments use `Qwen/Qwen2.5-7B-Instruct-Turbo` through
the Together API. Additional runners support the open-family analyses reported
in the appendix through OpenRouter. Hosted-model runners read credentials from
the `TOGETHER_API_KEY` or `OPENROUTER_API_KEY` environment variable. Never put
credentials in source files, command-line arguments, logs, or commits.

The commands below launch full experiment configurations and may incur API
costs. Review the model, seed count, call budget, and output path in each runner
before execution. Several original experiment runners use module-level
configuration rather than a complete command-line interface.

Examples:

```bash
# Hosted main prompt sweep
python -m benchmark_v4.runners.run_main_prompt

# Hosted history ablation
python -m benchmark_v4.runners.run_history_ablation

# Non-LLM baselines
python -m benchmark_v4.runners.run_baselines

# Deterministic landscape analyses
python -m benchmark_v4.runners.analyze_gb1_landscape \
  --output results/gb1_landscape_stats.json
python -m benchmark_v4.runners.analyze_alloy_mechanism \
  --output results/alloy_mechanism_stats.json
```

Runners with a command-line interface document their options under `--help`.
The original fixed-grid runners should be inspected directly before use.

## Released results

`results/` contains released logs and summaries for the central prompt sweep,
GB1 rerun, history ablation, non-LLM baselines, Alloy-like exemplar ablation,
GB1 calibration, and Word Ladder length ablation. Compact aggregate summaries
for the broader open-family analyses are also included. These files support
inspection of the reported results without making hosted-model calls. The much
larger development log directory is not part of this repository.

## Verification

To check that all Python files parse correctly:

```bash
python -m compileall -q benchmark_v4 scripts
```

## License

Code is released under the MIT License. Source datasets retain their original
licenses and citation requirements. Released result logs are provided for
research verification of the accompanying paper.

## Citation

Please cite the accompanying paper:

```bibtex
@inproceedings{ryu-etal-2026-local-edit-validity,
  title     = {Evaluating {LLM} Agents Beyond Local Edit Validity},
  author    = {Ryu, Kunhee and Lee, Chi-Guhn and Lee, Keeheon},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

The entry will be updated with ACL Anthology metadata when the proceedings are
published. Machine-readable software citation metadata is available in
`CITATION.cff`.
