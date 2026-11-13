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
just test tests/active_testing -q # Run tests for one area
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

Files are written in `data/raw/fairvision/` by default. The image command
uses HTTP byte ranges to retrieve selected JPEG members from the upstream ZIP
rather than transferring the complete 40--63 GB disease archive. See the
[upstream FairVision dataset](https://huggingface.co/datasets/harvardairobotics/FairVision)
for its CC BY-NC-ND 4.0 license. It is for non-commercial research and not for
clinical decisions or patient care.

## Experiments

Experiments live in `active_testing_benchmark.experiments`, with separate
modules for `feature_experiment`, `model_selection`, and `at_benchmarking`.
Shared evaluation and checkpoint helpers live in `experiments.utils`. The
dataset reads FairVision's metadata and SLO images from
`data/raw/fairvision/<disease>/`, preserving its `training`, `validation`, and
`test` splits. Raw images are never copied or changed.

Use `--disease dr|amd|glaucoma` to select a disease. Download its data before
running an experiment. Omit `--seed` to generate and log a fresh random seed,
or pass `--seed N` to reproduce a run. The recipes forward CLI flags, and
`--help` lists the options for each experiment.

### Comparison of feature extraction used in surrogates

This experiment compares image representations for random forest classifiers.
Compare the original 16×16 pooled pixels, spatial texture features, and a frozen
ImageNet ResNet-18 backbone using:

```bash
just feature-experiment

# See CLI options with:
just feature-experiment --help
```

The ImageNet backbone returns 512 features before the classification layer. Its
weights and batch-normalization statistics stay frozen. It uses the same image
resize and ImageNet normalization as the other extractors, retaining the full
image instead of adding a center crop. The first use downloads approximately 45
MB of `ResNet18_Weights.IMAGENET1K_V1` weights to torchvision's cache. MLflow
records the weight version and initialization time separately from fit time. To
select only certain extractors, pass, for example, `--extractors pooled_pixels
imagenet_resnet18`.

All forests within an invocation use the same seed, settings, and stratified
80/20 split of the FairVision training partition. The holdout is used to compare
extractors, so report final performance on a separate partition. The official
validation and test partitions are not used for feature selection. MLflow
records classification loss, accuracy, F1, AUC when available, feature counts,
fit and evaluation time, and split provenance. Use `--disease`,
`--n-estimators`, `--batch-size`, `--image-size`, or `--device` to change the
defaults, and `--no-mlflow` to run without tracking. The command prints a
comparison table when complete.

### Model selection

Compare ResNet-18 and EfficientNet-B0 using:

```bash
just model-selection

# See CLI options with:
just model-selection --help
```

Both models use the same stratified 80/20 split of FairVision's training
partition, with the 20% holdout used to select checkpoints. Their best
checkpoints are compared on FairVision validation. This command does not run AT
methods or use test data.

Model selection writes `comparison.csv` and `history.json` under
`reports/active-testing/model-comparison/`. These are replaced on subsequent
runs.

### Active-testing (AT) benchmarking

This experiment evaluates risk estimation methods using a trained image
classifier. Development and final runs use different FairVision partitions.

#### Development run

Run the development benchmark (the recipe first syncs dependencies):

```bash
just experiment

# See CLI options with:
just experiment --help
```

The development AT experiment trains ResNet-18 on a label-stratified 80% of
FairVision's training split. The remaining 20% selects the checkpoint with the
highest ROC AUC. Ties keep the earliest checkpoint. The restored model is
evaluated and benchmarked on FairVision validation. Training defaults to 10
epochs, batch size 32, pretrained ImageNet weights, unweighted cross-entropy,
AdamW, and cosine learning rate scheduling.

#### Final run

The final AT experiment's model trains on FairVision training, selects the
checkpoint performing the best on FairVision validation, and evaluates on test:

```bash
just experiment --FINAL
```

This requires typing `FINAL` at the interactive prompt before any
final-experiment data is loaded. The prompt also applies when reusing a final
checkpoint. `--FINAL` flag is only available for the AT benchmarking experiment.

#### Methods and subgroup reporting

AT benchmarking runs uniform, balanced/proportional stratified, Kossen,
active-surrogate, and prediction-powered methods. It estimates unweighted
cross-entropy risk overall and by subgroup, including `age_group`. Each method
runs for `--repetitions` (default 50), with a max label budget equal to the
full evaluation pool.

Kossen and PPAT use the same average predictive probabilities from both
surrogate forests. ASE starts with that ensemble and uses its individual
members for XWED.

Both surrogates and the proxy use frozen ImageNet ResNet-18 features for their
random forests by default. The classification layer is removed, producing 512
features per image.  Backbone weights and batch-normalization statistics remain
frozen during fitting, prediction, and ASE updates. It accepts the same resized,
ImageNet-normalized images as the target model. The first use downloads
approximately 45 MB of `ResNet18_Weights.IMAGENET1K_V1` weights to torchvision's
cache.  Custom extractors remain supported through
`RandomForestModule(feature_extractor=...)`, including
`SpatialTextureFeatures()` for the previous texture baseline.

ASE refits its surrogate forests on the original development data plus all
acquired labels every 100 labels. Set `--ase-update-every N` to change that
frequency (positive integers only). Acquisition batches stop at update
thresholds so refitting happens at the requested label counts. Each objective
and repetition starts with fresh copies of the development-trained ASE
surrogates.

By default, AT benchmarking reports all marginal subgroup attributes except raw
`age`. Uniform sampling, Kossen, ASE, and PPAT each share one sampling run per
repetition and estimates risk across all attributes. The two stratified
baselines run separately for each protected attribute, and each run also reports
every requested marginal attribute. Use `--protected-attribute` to restrict
stratification to `gender`, `race`, `ethnicity`, `language`, `maritalstatus`, or
`age_group`.  All available marginal attributes are still reported.

For example:

```bash
just experiment --repetitions 5 --seed 42 --epochs 20
just experiment --ase-update-every 100 --seed 42
just experiment --protected-attribute gender --seed 42
```

#### Surrogate diagnostics

Before benchmarking, AT runs measure how well the surrogates and proxy track
the classifier's cross-entropy losses. These measurements use the
checkpoint-selection holdout, never the evaluation pool. Active testing can
only beat uniform sampling when its acquisition scores follow the true losses.
The diagnostics are:

| Name | Meaning |
|---|---|
| `kossen_score_spearman` | Rank correlation of the Kossen acquisition score with the true loss |
| `kossen_variance_gain` | Uniform variance divided by Kossen proposal variance (above 1 is better than uniform) |
| `oracle_variance_gain` | The same ratio for a proposal proportional to the true loss (the upper limit) |
| `ase_relative_bias` | Relative error of the ASE estimate before acquiring labels |
| `ppat_proxy_correlation` | Correlation of the PPAT proxy loss with the true loss (λ = 0.5 helps above 0.25) |

They are logged to MLflow under `diagnostics/` and saved in the metadata JSON
as `surrogate_diagnostics`.

#### Outputs and tracking

AT runs save uniquely named curves CSVs and matching metadata JSON files under
`reports/active-testing/AT_benchmarking/`. Curves contain methods, objectives,
repetitions, seeds, label counts, and overall/subgroup risk estimates. Metadata
records configuration, checkpoint details, training history, auxiliary seeds,
dataset sizes, and full-pool reference metrics.

Curves include a `protected_attribute` column: `shared` identifies a
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

During AT, the MLflow tags `stage`, `AT.method`, and `AT.repetition` show
progress. Risk and label-count metrics are logged after each method/repetition. Overall and subgroup reference metrics
are logged after evaluation. Full AT curves remain available in the CSV artifact.

### Classifier training and checkpointing

Model selection and AT use the classifier training pipeline. Additional
training settings, such as image size and learning rate, are configured
through `modeling.train.TrainingConfig`. For example:

```bash
just model-selection --epochs 10 --batch-size 16 --device cpu
```

Each epoch logs loss, accuracy, F1, and ROC AUC under separate MLflow paths:
`training/train/{metric}` and `training/validation/{metric}`, where the metric
names are `loss`, `accuracy`, `f1`, and `roc_auc`. Training metrics aggregate
predictions from the optimization pass, while validation uses the updated model
in evaluation mode after the epoch. No extra training-set inference pass is
needed. Checkpoint selection continues to maximize validation AUC.

Best checkpoints are saved under `models/experiment/`, separated by
`model_selection`, `AT_dev`, or `AT_final`, then disease and model name. A fresh
training run replaces the checkpoint at its corresponding path. To load an
existing checkpoint and skip classifier training:

```bash
just experiment --reuse-trained-model
just model-selection --reuse-trained-model
```

`--checkpoint-dir` selects another checkpoint root. Reuse requires a matching
checkpoint. AT reuse runs with the checkpoint's training seed, so the
development holdout, auxiliary models and repetitions match the original run.
Omit `--seed` to adopt it. A mismatching `--seed` stops the run before any
prompt or data loading. Python callers get a `ValueError` if
`training_config.random_state` differs from the checkpoint's seed. AT still
fits its auxiliary forests and runs evaluation and benchmarking. Checkpoints
retain their original training configuration, seed, and history.

### Python entry points

Python callers import `run_feature_experiment()` from
`active_testing_benchmark.experiments.feature_experiment` to return a
comparison DataFrame.

Python callers import `run_dev_experiment()` and `run_final_experiment()` from
`active_testing_benchmark.experiments.at_benchmarking`, and
`run_model_selection_experiment()` from
`active_testing_benchmark.experiments.model_selection`. The AT functions return
reference metrics. Model selection returns a comparison DataFrame. They accept
`training_config`, `data_root`, `output_dir`, `checkpoint_dir`, and
`reuse_trained_model`. AT functions also accept `repetitions`. `modeling.data`
owns the dataset, `modeling.train` owns training and auxiliary fitting, and
`benchmark` owns repeated active evaluation.

## MLflow tracking

Tracking is enabled by default. Set `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` in `.env`, using
`.env.example` as a template. The `just feature-experiment`,
`just model-selection`, and `just experiment` commands then send results to that
server.

| Experiment | MLflow experiment name | Runs |
|---|---|---|
| Model selection | `FairVision/model_selection` | One parent per invocation, one child per model |
| Random forest features | `FairVision/feature_extractors` | One parent per invocation, one child per extractor |
| Development AT | `FairVision/AT_dev` | One run per invocation |
| Final AT | `FairVision/AT_final` | One run per confirmed invocation |

Parents record the shared configuration, seed, selected methods, dataset
provenance, and a `comparison.csv` artifact. Child runs retain method-specific
parameters, metrics, and artifacts. Returned comparison tables include both
`mlflow_run_id` and `mlflow_parent_run_id`. A method failure marks both that child
and its parent failed. Tracking can still be disabled with `--no-mlflow`.

Classifier and AT runs record their configuration and seeds, best checkpoint
and training history. AT runs also record reference metrics and CSV/JSON outputs.
Reused checkpoints retain
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
└── active_testing_benchmark    <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes active_testing_benchmark a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── experiments             <- Feature, model-selection, and AT experiment entry points
    │   ├── __init__.py
    │   ├── at_benchmarking.py
    │   ├── train.py
    │   └── models.py
    │
    ├── benchmark.py            <- AT methods, objectives, and repetitions
    ├── active_testing          <- Sampling methods, estimators, runner, and surrogates
    │
    └── modeling
        ├── __init__.py
        ├── data.py             <- Current FairVision dataset and data preparation
        ├── train.py            <- Current classifier and auxiliary-model training
        └── models.py           <- Model factories and image transforms
```

--------
