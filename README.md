# active-testing-benchmark

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Benchmarking of active-testing approaches on medical imaging

## Project commands

Install [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), then use
`just` to list the available commands. Commands run Python tools through uv.

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

Files are written below `data/raw/fairvision/` by default. The image command uses
HTTP byte ranges to retrieve selected JPEG members from the upstream ZIP rather
than transferring the complete 40--63 GB disease archive. See the
[upstream FairVision dataset](https://huggingface.co/datasets/harvardairobotics/FairVision)
for its CC BY-NC-ND 4.0 license. It is for non-commercial research and not for
clinical decisions or patient care.

## Image-classification experiments

Experiments run through `active_testing_benchmark.new_experiment`. The new
dataset reads FairVision's metadata and SLO images from
`data/raw/fairvision/<disease>/`, preserving its `training`, `validation`, and
`test` splits. Raw images are never copied or changed.

Run the development benchmark (the recipe first syncs dependencies):

```bash
just experiment --seed 42
```

Development AT experiment trains ResNet-18 on a stratified 80% of FairVision's training
split. The remaining 20% selects the checkpoint with the highest ROC AUC. Ties
keep the earliest checkpoint. The restored model is evaluated and benchmarked
on FairVision validation. Training defaults to 10 epochs, batch size 32,
pretrained ImageNet weights, unweighted cross-entropy, AdamW, and cosine learning
rate scheduling. A random seed is generated and logged unless `--seed` is given.

A seperate model selection experiment comparing ResNet-18 and EfficientNet-B0 can be run:

```bash
just model-selection --seed 42

# Equivalent to:
just experiment --experiment model_selection --seed 42
```

Both models use the same 80/20 training split. Their best checkpoints are compared
on FairVision validation. This command does not run AT methods or use test data.

AT benchmarking runs uniform, balanced/proportional stratified, Kossen,
active-surrogate, and prediction-powered methods. It estimates unweighted
cross-entropy risk overall and by subgroup, including `age_group`. Each method
runs for `--repetitions` (default 25), with a max label budget equal to the full
evaluation pool in 10 label increments.

The recipes forward CLI flags, for example:

```bash
just experiment --repetitions 5 --seed 42 --epochs 20
just model-selection --epochs 10 --batch-size 16 --device cpu
just experiment --help
```

Use `--disease dr|amd|glaucoma` to select the disease (they must be downloaded
before running the experiment). All available CLI flags and arguments can be
found with `--help`. Additional training
settings, such as image size and learning rate, are configured through
`modeling.new_train.TrainingConfig`.

Best checkpoints are saved under `models/new-experiment/`, separated by
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

Final AT trains on FairVision training, selects the best checkpoint on FairVision
validation, and evaluates on test:

```bash
just experiment --FINAL --seed 42
```

This requires typing `FINAL` at the interactive prompt before any final-experiment
data is loaded. The prompt also applies when reusing a final checkpoint.
`--FINAL` is unavailable for model selection.

AT runs save uniquely named curves CSVs and matching metadata JSON files under
`reports/active-testing/AT_benchmarking/`. Curves contain methods, objectives,
repetitions, seeds, label counts, and overall/subgroup risk estimates. Metadata
records configuration, checkpoint details, training history, auxiliary seeds,
dataset sizes, and full-pool reference metrics. Model selection writes
`comparison.csv` and `history.json` under
`reports/active-testing/model-comparison/`. These are replaced on subsequent runs.

Python callers use `run_dev_experiment()`, `run_final_experiment()`, or
`run_model_selection_experiment()` from `new_experiment`. The AT functions return
reference metrics. Model selection returns a comparison DataFrame. They accept
`training_config`, `data_root`, `output_dir`, `checkpoint_dir`, and
`reuse_trained_model`. AT functions also accept `repetitions`.
`modeling.new` owns the dataset, `modeling.new_train` owns training and auxiliary
fitting, and `benchmark` owns repeated active evaluation.

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
    ├── new_experiment.py       <- Current model-selection and AT experiment entry point
    ├── benchmark.py            <- AT methods, objectives, and repetitions
    ├── active_testing         <- Sampling methods, estimators, runner, and surrogates
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── new.py              <- Current FairVision dataset and data preparation
    │   ├── new_train.py        <- Current classifier and auxiliary-model training
    │   ├── models.py           <- Model factories and image transforms
    │   └── ...                 <- Older pipeline and optional MLflow tooling
    │
    ├── experiment.py           <- Older experiment entry point, retained during migration
    └── plots.py                <- Plotting utility for an older results format
```

--------
