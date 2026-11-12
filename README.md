# active-testing-benchmark

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Benchmarking of active-testing approaches on medical imaging

## Project commands

Install [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), then
use `just` to list the available commands. Commands run Python tools through uv.

```bash
just requirements              # Sync dependencies from uv.lock
just test                      # Run the test suite
just test tests/test_runner_subgroups.py -q
just lint                      # Check formatting and lint rules
just format                    # Apply formatting and lint fixes
```

## FairVision raw data

The FairVision downloader retains the standalone, original-resolution SLO fundus
JPEGs and the disease metadata CSVs only. It does not download or retain the 3D
OCT-containing NPZ files. For example:

```bash
just fairvision-metadata glaucoma
just data glaucoma
```

The disease defaults to `dr`. Pass `amd`, `glaucoma`, or `all` as the first
argument. Additional downloader flags follow it, for example
`just data glaucoma --output-dir /path/to/fairvision`.

Files are written below `data/raw/fairvision/` by default. The image command
uses HTTP byte ranges to retrieve selected JPEG members from the upstream ZIP
rather than transferring the complete 40--63 GB disease archive. See the
[upstream FairVision dataset](https://huggingface.co/datasets/harvardairobotics/FairVision)
for its CC BY-NC-ND 4.0 license. It is for non-commercial research and not for
clinical decisions or patient care.

## Image-classification experiments

Experiments run through `active_testing_benchmark.experiment`. The dataset reads
FairVision's metadata and SLO images from `data/raw/fairvision/<disease>/`,
preserving its `training`, `validation`, and `test` splits. Raw images are never
copied or changed.

Run the development benchmark (the recipe first syncs dependencies):

```bash
just experiment --seed 42
```

Development AT experiment trains ResNet-18 on a stratified 80% of FairVision's
training split. The remaining 20% selects the checkpoint with the highest ROC
AUC. Ties keep the earliest checkpoint. The restored model is evaluated and
benchmarked on FairVision validation. Training defaults to 10 epochs, batch size
32, pretrained ImageNet weights, unweighted cross-entropy, AdamW, and cosine
learning rate scheduling. A random seed is generated and logged unless `--seed`
is given.

A seperate model selection experiment comparing ResNet-18 and EfficientNet-B0
can be run:

```bash
just model-selection --seed 42

# Equivalent to:
just experiment --experiment model_selection --seed 42
```

Both models use the same 80/20 training split. Their best checkpoints are
compared on FairVision validation. This command does not run AT methods or use
test data.

AT benchmarking runs uniform, balanced/proportional stratified, Kossen,
active-surrogate, and prediction-powered methods. It estimates unweighted
cross-entropy risk overall and by subgroup, including `age_group`. Each method
runs for `--repetitions` (default 25), with a max label budget equal to the
full evaluation pool.

Kossen and PPAT use the same average predictive probabilities from both
surrogate forests. ASE starts with that ensemble and uses its individual
members for XWED.

ASE refits its surrogate forests on the original development data plus all
acquired labels every 50 labels. Set `--ase-update-every N` to change that
frequency (positive integers only). Acquisition batches stop at update
thresholds so refitting happens at the requested label counts. Each objective
and repetition starts with fresh copies of the development-trained ASE
surrogates.

The recipes forward CLI flags, for example:

```bash
just experiment --repetitions 5 --seed 42 --epochs 20
just experiment --ase-update-every 100 --seed 42
just experiment --protected-attribute gender --seed 42
just model-selection --epochs 10 --batch-size 16 --device cpu
just experiment --help
```

By default, AT benchmarking reports all marginal subgroup attributes except raw
`age`. Uniform sampling, Kossen, ASE, and PPAT each share one sampling run per
repetition and objective across all attributes. The two stratified baselines run
separately for each sampling attribute, and each run also reports every requested
marginal attribute. Use `--protected-attribute` to restrict stratification to
`gender`, `race`, `ethnicity`, `language`, `maritalstatus`, or `age_group`.
All available marginal attributes are still reported.

Use `--disease dr|amd|glaucoma` to select the disease (they must be downloaded
before running the experiment). All available CLI flags and arguments can be
found with `--help`. Additional training settings, such as image size and
learning rate, are configured through `modeling.train.TrainingConfig`.

Best checkpoints are saved under `models/experiment/`, separated by
`model_selection`, `AT_dev`, or `AT_final`, then disease and model name. A fresh
training run replaces the checkpoint at its corresponding path. To load an
existing checkpoint and skip classifier training:

```bash
just experiment --reuse-trained-model --seed 42
just model-selection --reuse-trained-model
```

`--checkpoint-dir` selects another checkpoint root. Reuse requires a matching
checkpoint. AT still fits its auxiliary forests and runs evaluation and
benchmarking. Checkpoints retain their original training configuration, seed,
and history.

Final AT trains on FairVision training, selects the best checkpoint on
FairVision validation, and evaluates on test:

```bash
just experiment --FINAL --seed 42
```

This requires typing `FINAL` at the interactive prompt before any
final-experiment data is loaded. The prompt also applies when reusing a final
checkpoint. `--FINAL` is unavailable for model selection.

AT runs save uniquely named curves CSVs and matching metadata JSON files under
`reports/active-testing/AT_benchmarking/`. Curves contain methods, objectives,
repetitions, seeds, label counts, and overall/subgroup risk estimates. Metadata
records configuration, checkpoint details, training history, auxiliary seeds,
dataset sizes, and full-pool reference metrics. Model selection writes
`comparison.csv` and `history.json` under
`reports/active-testing/model-comparison/`. These are replaced on subsequent
runs. Curves include a `protected_attribute` column: `shared` identifies a
non-stratified run reporting multiple attributes, while an attribute name
identifies its stratification attribute. Shared runs report all attributes even
when the CLI override selects one stratification attribute. Subgroup estimates
and label counts remain wide columns (for example, `risk gender=female` and
`n_labels_gender=female`). For a stratified run, the column names the sampling
attribute. Every run contains estimates and counts for all requested marginal
attributes. A reported subgroup may cross the sampling strata, so its estimate
uses the stratum weights from that run. At smaller budgets, methods that
require observed labels may leave a subgroup's risk blank when its label count
is zero. Prediction-based methods may still estimate it. At the full-pool
budget, every requested subgroup has labels and an estimate. Overall risk is
stored once per sampling trajectory.

Python callers use `run_dev_experiment()`, `run_final_experiment()`, or
`run_model_selection_experiment()` from `experiment`. The AT functions return
reference metrics. Model selection returns a comparison DataFrame. They accept
`training_config`, `data_root`, `output_dir`, `checkpoint_dir`, and
`reuse_trained_model`. AT functions also accept `repetitions`. `modeling.data`
owns the dataset, `modeling.train` owns training and auxiliary fitting, and
`benchmark` owns repeated active evaluation.

## MLflow tracking

Tracking is enabled by default. Set `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` in `.env`, using
`.env.example` as a template. The existing `just experiment` and `just
model-selection` commands then send results to that server.

| Experiment | MLflow experiment name | Runs |
|---|---|---|
| Model selection | `FairVision/model_selection` | One run per model |
| Development AT | `FairVision/AT_dev` | One run per invocation |
| Final AT | `FairVision/AT_final` | One run per confirmed invocation |

Each epoch logs loss, accuracy, F1, and ROC AUC under separate MLflow paths:
`training/train/{metric}` and `training/validation/{metric}`, where the metric
names are `loss`, `accuracy`, `f1`, and `roc_auc`. Training metrics aggregate
predictions from the optimization pass, while validation uses the updated model
in evaluation mode after the epoch. No extra training-set inference pass is
needed. Checkpoint selection continues to maximize validation AUC.

During AT, the `stage`,
`AT.method`, and `AT.repetition` tags show progress. Risk and label-count metrics
are logged after each method/repetition. Overall and subgroup reference metrics
are logged after evaluation. Full AT curves remain available in the CSV artifact.

Each run records its configuration and seeds, best checkpoint and training
history, reference metrics, and AT CSV/JSON outputs. Reused checkpoints retain
their original training configuration and, when trained with tracking, source
run ID. Older checkpoints remain reusable. Local outputs are still
saved, and the metadata/comparison CSV includes the corresponding MLflow run ID.

For reproducibility, runs also save the current Python source as `source.zip`,
the Git commit and dirty status, `pyproject.toml`, `uv.lock`, installed package
versions, and hashes of the metadata CSV and each partition's ordered sample IDs
and labels. Dataset provenance uses metadata only. Images are not uploaded.

Use `--no-mlflow` for local-only runs, or `track=False` when calling the experiment
functions from Python:

```bash
just experiment --no-mlflow --seed 42
```

The final-experiment confirmation is required with or without tracking. Failed
runs are marked as failed in MLflow. Tracking errors propagate rather than
silently producing an untracked run.

## Acknowledgment

The weekly meeting notes are inspired by the
[WhitakerLabProjectManagement](https://github.com/WhitakerLab/WhitakerLabProjectManagement)
repository. Thank you to its makers for sharing their approach.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── justfile           <- Recipes for data, experiments, tests, linting, and formatting
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project. See www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Package dependencies and Ruff configuration
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── uv.lock            <- Locked dependencies, installed with `just requirements`
│
└── active_testing_benchmark   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes active_testing_benchmark a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── experiment.py       <- Current model-selection and AT experiment entry point
    ├── benchmark.py            <- AT methods, objectives, and repetitions
    ├── active_testing         <- Sampling methods, estimators, runner, and surrogates
    │
    └── modeling
        ├── __init__.py
        ├── data.py              <- Current FairVision dataset and data preparation
        ├── train.py        <- Current classifier and auxiliary-model training
        └── models.py           <- Model factories and image transforms
```

--------
