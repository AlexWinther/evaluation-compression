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
just test tests/test_experiment.py -q
just lint                      # Check formatting and lint rules
just format                    # Apply formatting and lint fixes
```

## FairVision raw data

The FairVision downloader retains the standalone, original-resolution SLO fundus
JPEGs and the disease metadata CSVs only; it does not download or retain the 3D
OCT-containing NPZ files. For example:

```bash
just fairvision-metadata glaucoma
just data glaucoma
```

The disease defaults to `dr`; pass `amd`, `glaucoma`, or `all` as the first
argument. Additional downloader flags follow it, for example
`just data glaucoma --output-dir /path/to/fairvision`.

Files are written below `data/raw/fairvision/` by default. The image command uses
HTTP byte ranges to retrieve selected JPEG members from the upstream ZIP rather
than transferring the complete 40--63 GB disease archive. See the
[upstream FairVision dataset](https://huggingface.co/datasets/harvardairobotics/FairVision)
for its CC BY-NC-ND 4.0 license; it is for non-commercial research and not for
clinical decisions or patient care.

## Image-classification experiments

The downloaded DR SLO images already use FairVision's `training`, `validation`,
and `test` splits. The default command reads
`data/raw/fairvision/dr/metadata.csv`, maps its `filename` entries such as
`data_09382.npz` to `test/slo_fundus_09382.jpg`, and uses `dr` as the binary
target. Raw images are never copied or changed.

Run the development benchmark (the recipe first syncs dependencies):

```bash
just experiment --seed 42
```

The experiment uses `train` for development and `val` for active evaluation.
A seeded, stratified 20% of development data selects the pilot duration; a fresh
ResNet-18 is then refitted on all development data. Training defaults to ten pilot
epochs, inverse-frequency weighted loss, and earliest-minimum class-balanced
validation cross-entropy. The evaluation pool never determines model selection.
Final experiments involving the test split remain blocked.

For pipeline development, replace classifier training with a bounded real loop:

```bash
just experiment-fast --seed 42
```

Fast mode runs one pilot epoch and one fresh refit epoch, each capped at ten
batches. Counts and weights still use the complete relevant training partition.
Ten batches do not guarantee observing every class. Validation, surrogate/proxy
forest fitting, reference inference, and active testing still run in full.
This mode checks the pipeline; its results should not be used as trained-model
benchmark evidence.

Use `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--image-size`,
and `--num-workers` to configure training. `--no-weighted-loss` and
`--selection-objective unweighted_cross_entropy` independently control fitting
and epoch selection. `--weighted-sampling` enables inverse-frequency sampling.
ResNet-18 uses pretrained ImageNet weights by default; `--no-pretrained` avoids
the weight download. `--device cpu` bypasses accelerator selection.

The benchmark retains uniform, balanced/proportional stratified, Kossen,
active-surrogate, and prediction-powered methods. Each runs for `--repetitions`
(default 1) with `--budget` labels (default 1000, no greater than the evaluation
pool size). `--no-class-balanced-at` disables the additional balanced-risk
objective without changing training or selection.

Both experiment recipes forward CLI flags, for example:

```bash
just experiment --repetitions 5 --seed 42 --epochs 20
just experiment-fast --seed 42 --device cpu --budget 100 --no-pretrained
just experiment --help
```

Each run writes a uniquely named CSV and JSON sidecar beneath
`results/active-testing/`, configurable with `--output-dir`. CSV rows contain
method, objective, repetition, seed, label counts, and risk estimates. JSON stores
resolved configuration and seeds, training history and mode, class counts and
weights, and full-pool reference metrics. No MLflow credentials are required.
`--data-root` selects an alternative FairVision root and `--disease` selects
`dr`, `glaucoma`, or `amd`.

Python callers use `run_experiment(ExperimentConfig(...))`, which returns an
`ExperimentResult` with curves, reference metrics, training metadata, resolved
configuration, and artifact paths. `modeling.preparation` prepares partitions;
`modeling.train` owns classifier and auxiliary fitting; `modeling.evaluation`
computes reference metrics; `benchmark` constructs methods and runs repetitions.
The full and fast classifier trainers accept the same prepared data,
`TrainingConfig`, and device and return the same `TrainingResult` type.

## Acknowledgment

The weekly meeting notes are inspired by the
[WhitakerLabProjectManagement](https://github.com/WhitakerLab/WhitakerLabProjectManagement)
repository. We thank its makers for sharing their approach.

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
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         active_testing_benchmark and configuration for tools like black
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
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
