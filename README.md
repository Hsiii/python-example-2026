# PhysioNet 2026 Survival Benchmarking Framework

This README documents the benchmarking framework that was added on top of the original PhysioNet Challenge template.

The original template README and Challenge-specific repository guidance have been preserved in [README.md.bak](/Users/hsi/Documents/Code/Moody/python-example-2026/README.md.bak).

## What was added

This repository now includes a modular, reproducible framework for predicting time to cognitive impairment diagnosis from polysomnography and tabular metadata.

Added capabilities:

- survival analysis with censoring;
- multimodal inputs spanning raw PSG, engineered PSG features, and demographics;
- classical, deep, signal-only, and multimodal model classes;
- config-driven experiments and benchmark summaries; and
- a Challenge-compatible baseline wrapper backed by the modular framework.

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
```

## Added modules

### Data pipeline

- patient-level dataset indexing and survival target extraction in [src/data/loaders.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/data/loaders.py)
- grouped train, validation, test splitting and grouped cross-validation in [src/data/splits.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/data/splits.py)
- tensor dataset support for deep models in [src/data/datasets.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/data/datasets.py)

### Feature engineering

Engineered PSG feature extraction is implemented in [src/features/psg_features.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/features/psg_features.py).

Extracted features include:

- sleep efficiency
- REM latency
- REM fraction and N3 fraction
- AHI
- oxygen desaturation metrics
- arousal index
- sleep fragmentation metrics
- EEG spectral delta, theta, alpha, and beta power
- sleep stage transition statistics

Demographic features are implemented in [src/features/tabular.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/features/tabular.py).

### Models

Implemented model families:

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

Key model files:

- [src/models/classical.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/classical.py)
- [src/models/deep.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/deep.py)
- [src/models/signal.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/signal.py)
- [src/models/multimodal.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/multimodal.py)
- [src/models/ensemble.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/models/ensemble.py)

### Training and evaluation

- unified neural training loop in [src/training/trainer.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/training/trainer.py)
- Cox and discrete-time survival losses in [src/training/losses.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/training/losses.py)
- C-index, integrated Brier score, time-dependent AUC, bootstrap confidence intervals, and subgroup analysis in [src/evaluation/metrics.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/evaluation/metrics.py)

## Experiment runner

Config-driven experiments run through [run_experiment.py](/Users/hsi/Documents/Code/Moody/python-example-2026/run_experiment.py).

Example configs:

- [configs/cox_baseline.yaml](/Users/hsi/Documents/Code/Moody/python-example-2026/configs/cox_baseline.yaml)
- [configs/random_survival_forest.yaml](/Users/hsi/Documents/Code/Moody/python-example-2026/configs/random_survival_forest.yaml)
- [configs/deep_surv.yaml](/Users/hsi/Documents/Code/Moody/python-example-2026/configs/deep_surv.yaml)
- [configs/multimodal_cnn.yaml](/Users/hsi/Documents/Code/Moody/python-example-2026/configs/multimodal_cnn.yaml)
- [configs/benchmark_suite.yaml](/Users/hsi/Documents/Code/Moody/python-example-2026/configs/benchmark_suite.yaml)

Run examples:

```bash
python run_experiment.py --config configs/cox_baseline.yaml
python run_experiment.py --config configs/benchmark_suite.yaml
```

Outputs are written under `experiments/` and include:

- resolved configs
- cached engineered features
- per-model metrics
- subgroup evaluation tables
- benchmark summary tables

## Challenge compatibility

The modular framework is connected to the Challenge entry points through [team_code.py](/Users/hsi/Documents/Code/Moody/python-example-2026/team_code.py) and [src/utils/challenge_baseline.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/utils/challenge_baseline.py).

That preserves compatibility with the existing root scripts while keeping implementation details in `src/`.

## Reproducibility

- seeded execution in [src/utils/seed.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/utils/seed.py)
- config loading in [src/utils/config.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/utils/config.py)
- run artifact writing in [src/utils/io.py](/Users/hsi/Documents/Code/Moody/python-example-2026/src/utils/io.py)
- cached deterministic engineered feature extraction
- group-aware data splitting

## Tests

Lightweight checks live in [tests/test_metrics.py](/Users/hsi/Documents/Code/Moody/python-example-2026/tests/test_metrics.py), [tests/test_tabular_features.py](/Users/hsi/Documents/Code/Moody/python-example-2026/tests/test_tabular_features.py), and [tests/test_config_loading.py](/Users/hsi/Documents/Code/Moody/python-example-2026/tests/test_config_loading.py).

Run them with:

```bash
pytest tests
```

## Workspace note

The current workspace snapshot still has an empty [data/training_data](/Users/hsi/Documents/Code/Moody/python-example-2026/data/training_data) directory, so real benchmark outputs cannot be generated until labeled training data is placed there.
