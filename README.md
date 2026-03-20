# PhysioNet 2026 Survival Benchmarking Framework

This repository extends the minimal George B. Moody PhysioNet Challenge 2026 example into a modular, reproducible benchmarking framework for predicting time to cognitive impairment diagnosis from polysomnography and tabular metadata.

The framework supports:

- survival analysis with censoring;
- multimodal inputs spanning raw PSG, engineered PSG features, and demographics;
- classical, deep, signal-only, and multimodal model classes;
- config-driven experiments and benchmark summaries; and
- backward-compatible Challenge entry points through `train_model.py`, `run_model.py`, and `team_code.py`.

## Repository layout

```text
python-example-2026/
    configs/
    data/
    experiments/
    notebooks/
    src/
        data/
        evaluation/
        features/
        models/
        training/
        utils/
    tests/
    run_experiment.py
    team_code.py
```

## Implemented components

### Data pipeline

- deterministic patient-level dataset indexing from `demographics.csv`;
- loaders for raw PSG EDF files, algorithmic annotations, and human annotations when available;
- survival target extraction using `Time_to_Event` and `Time_to_Last_Visit`;
- grouped train/validation/test splitting and grouped cross-validation utilities.

### Feature engineering

The main engineered feature extractor is implemented in [src/features/psg_features.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/features/psg_features.py). It provides cached, reproducible feature extraction for:

- sleep efficiency;
- REM latency;
- % REM and % N3;
- AHI;
- oxygen desaturation metrics;
- arousal index;
- sleep fragmentation metrics;
- EEG spectral band powers for delta, theta, alpha, and beta;
- sleep stage transition statistics.

Demographic feature extraction lives in [src/features/tabular.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/features/tabular.py).

### Models

Implemented model families include:

- `cox_ph`
- `regularized_cox`
- `random_survival_forest`
- `xgboost_survival`
- `deep_surv`
- `discrete_time`
- `deep_hit`
- `cnn_survival`
- `transformer_survival`
- `multimodal_survival`

Classical model wrappers are in [src/models/classical.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/classical.py), deep tabular survival models in [src/models/deep.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/deep.py), signal encoders in [src/models/signal.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/signal.py), and multimodal fusion in [src/models/multimodal.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/multimodal.py).

### Training and evaluation

- unified neural training loop with early stopping, checkpointing, and mixed precision in [src/training/trainer.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/training/trainer.py);
- Cox and discrete-time survival losses in [src/training/losses.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/training/losses.py);
- C-index, integrated Brier score, time-dependent AUC, bootstrap confidence intervals, and subgroup evaluation in [src/evaluation/metrics.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/evaluation/metrics.py).

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset format

The framework expects a dataset folder with at least:

```text
<dataset_root>/
    demographics.csv
    physiological_data/<SiteID>/<BidsFolder>_ses-<SessionID>.edf
    algorithmic_annotations/<SiteID>/<BidsFolder>_ses-<SessionID>_caisr_annotations.edf
    human_annotations/<SiteID>/<BidsFolder>_ses-<SessionID>_expert_annotations.edf
```

Required columns for survival training in `demographics.csv` are:

- `BidsFolder`
- `SessionID`
- `SiteID`
- `Time_to_Event`
- `Time_to_Last_Visit`
- `Cognitive_Impairment`

Demographic columns such as `Age`, `Sex`, `Race`, `Ethnicity`, and `BMI` are used when present.

## Running experiments

Single-model experiments:

```bash
python run_experiment.py --config configs/cox_baseline.yaml
python run_experiment.py --config configs/random_survival_forest.yaml
python run_experiment.py --config configs/deep_surv.yaml
python run_experiment.py --config configs/multimodal_cnn.yaml
```

Benchmark suite:

```bash
python run_experiment.py --config configs/benchmark_suite.yaml
```

Each run writes an experiment directory under `experiments/` containing:

- the resolved config;
- cached features;
- per-model metrics;
- subgroup metrics;
- `benchmark_summary.csv` and a Markdown summary table.

The benchmark summary table follows this format:

| Model | C-index | IBS | Notes |
|---|---:|---:|---|
| cox_ph | ... | ... | ... |

## Challenge-compatible usage

The root Challenge scripts still work. The new [team_code.py](/Users/hsi/Documents/Code/Moody/python-example-2026/team_code.py) delegates to the modular baseline defined in [src/utils/challenge_baseline.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/utils/challenge_baseline.py).

Train the baseline submission model:

```bash
python train_model.py -d data/training_data -m model -v
```

Run inference:

```bash
python run_model.py -d /path/to/holdout_data -m model -o /path/to/outputs -v
```

The convenience pipeline remains available:

```bash
python cox_pipeline.py --train-data data/training_data --test-data /path/to/holdout_data --model-folder model --output-folder outputs -v
```

## Benchmarking plan

The provided configs cover the requested baseline families:

1. Cox PH baseline
2. Random Survival Forest
3. DeepSurv
4. Multimodal CNN survival

The benchmark suite config also includes XGBoost survival. The remaining models are implemented in the framework and can be enabled with additional configs.

## Reproducibility

- global seeds are set through [src/utils/seed.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/utils/seed.py);
- splits are deterministic and group-aware;
- engineered features are cached to disk;
- experiment configs are copied into each run directory.

## Tests

Run the lightweight tests with:

```bash
pytest tests
```

## Important note for this workspace

The checked-in [data/training_data](/Users/hsi/Documents/Code/Moody/python-example-2026/data/training_data) directory is empty in the current workspace snapshot, so real benchmark scores cannot be produced until labeled training data is added there. The framework, configs, and tests are set up so experiments become runnable as soon as the data is populated.
