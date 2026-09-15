# active-testing-benchmark

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Benchmarking of active-testing approaches on medical imaging

## FairVision raw data

The FairVision downloader retains the standalone, original-resolution SLO fundus
JPEGs and the disease metadata CSVs only; it does not download or retain the 3D
OCT-containing NPZ files. For example:

```bash
uv run active_testing_benchmark/dataset.py --disease glaucoma --metadata-only
uv run active_testing_benchmark/dataset.py --disease glaucoma
```
or 
```bash
make fairvision-metadata DISEASE=glaucoma
make data DISEASE=glaucoma
```

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

Install dependencies once with `make requirements`, then run one model per
experiment (the first pretrained run downloads published weights):

```bash
uv run python -m active_testing_benchmark.modeling.train --model cnn --epochs 5 --batch-size 32 --learning-rate 1e-4
uv run python -m active_testing_benchmark.modeling.train --model resnet18 --epochs 5 --batch-size 32 --learning-rate 1e-4
uv run python -m active_testing_benchmark.modeling.train --model densenet121 --epochs 5 --batch-size 32 --learning-rate 1e-4
uv run python -m active_testing_benchmark.modeling.train --model vit --epochs 5 --batch-size 32 --learning-rate 1e-4
uv run python -m active_testing_benchmark.modeling.train --model clip --epochs 5 --batch-size 32 --learning-rate 1e-4
```

For imbalanced training labels, choose exactly one training-only strategy with
`--imbalance-strategy weighted-sampling` (balanced draws with replacement) or
`--imbalance-strategy class-weighted-loss` (inverse-frequency cross-entropy).
The default, `--imbalance-strategy none`, preserves the unweighted shuffled
training baseline. With `class-weighted-loss`, validation uses the same weighted
cross-entropy for checkpoint selection; test loss remains unweighted. Each epoch
reports macro-F1 and, for binary splits containing both classes, ROC-AUC.

`cnn` starts from scratch. Torchvision models use ImageNet weights; CLIP uses a
frozen OpenCLIP ViT-B/32 encoder with a trainable classification head. Pass
`--no-pretrained` to avoid weight downloads. Each run records metrics,
configuration, classes, test confusion matrix, and the best-validation model in
the remote MLflow server. Training does not persist a local checkpoint.

For normal laptop development, copy `.env.example` to the ignored `.env`, replace
the credential placeholders, and restrict its permissions:

```bash
cp .env.example .env
chmod 600 .env
```

The standard tracking server is `https://mlflow.alwn.dev`. The required client
variables are `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and
`MLFLOW_TRACKING_PASSWORD`. `MLFLOW_REGISTRY_URI` is optional; when omitted, the
registry uses the tracking URI.

On HPC jobs, inject the same variables through the scheduler's protected
secret/environment mechanism. Do not put credentials in committed job scripts or
scheduler command history. Confirm compute nodes can resolve the MLflow hostname
and make outbound HTTPS connections before starting long jobs.

The remote UI is available at `https://mlflow.alwn.dev` using the same Basic Auth
credentials.

Each run records sanitized dataset lineage for the training, validation, and test
splits: FairVision source, metadata checksum, selected-record manifest checksum,
schema, and aggregate split counts. It never uploads raw images or metadata rows.
The native MLflow model artifact includes a tensor signature, representative input,
configuration, and class mapping. A completed training run is automatically
registered as `fairvision-{disease-type}-{model-type}`. MLflow reserves the alias
name `latest`, so its native latest-version selector is used. For example, the
default DR CNN is available at `models:/fairvision-dr-cnn/latest`. Pass
`--disease-type` when the target column is not the desired disease identifier.
Project commands also accept `fairvision-dr-cnn@latest` as shorthand and convert
it to MLflow's native selector.

After reviewing a run, promote its latest version to the curated
`baseline` alias with:

```bash
uv run python -m active_testing_benchmark.modeling.registry \
  --registry-name fairvision-dr-cnn
```

This moves `@baseline` to the latest registered version without creating a
duplicate. To promote an older training run instead, add `--run-id <run-id>` (or
`RUN_ID=<run-id>` when using `make mlflow-model-promote`). Load the curated model
through `models:/fairvision-dr-cnn@baseline`.

For another layout, pass `--metadata-path`, `--image-root`, `--image-column`,
`--target-column`, and `--split-column`. Split values must be `training`,
`validation`, and `test` (case-insensitive).

radT is intentionally not installed or enabled yet. Its current published
materials describe an MLflow extension but do not document a stable minimal
in-process API for this loop. It can later be evaluated separately (`uv add
radt`) without affecting normal MLflow training.

## Offline active-testing experiment

Compare random test subsets against full-test metrics for one or more saved
models. Bare registry names default to MLflow's `/latest` selector; use `name@alias` for
another alias, or `run:<run-id>` / `runs:/<run-id>/model` for a training run:

```bash
uv run python -m active_testing_benchmark.active_testing \
  --model fairvision-dr-cnn@baseline \
  --model run:<run-id> \
  --test-size 25 --test-size 0.1 --test-size 1.0 \
  --runs 5 --seed 42

make active-testing ACTIVE_TESTING_FLAGS="--model fairvision-dr-cnn --test-size 0.1"
```

Integer sizes are counts (`1` means one row); decimal or exponent sizes are
fractions in `(0, 1]`, rounded up (`1.0` means the whole test split). Oversized or
empty budgets are rejected. Each size/repetition gets an independent seeded
sample without replacement, shared across all models. Samples may overlap or
coincide by chance; they are not nested or forced to be disjoint.

This POC loads the project's native MLflow PyTorch classifiers with their saved
configuration and class mapping. It performs one full-test inference pass per
model on CPU and retains row IDs, labels, predictions, and probabilities in
memory. Registry aliases are resolved to recorded version IDs. Preprocessing and
image size follow each saved model, with an optional `--image-size` override.
The dataset options are `--metadata-path`, `--image-root`, `--image-column`,
`--target-column`, `--split-column`, and `--dataset-source`; batching uses `--batch-size` and
`--num-workers`.

Each invocation writes `reports/active-testing/<mlflow-run-id>/` (or beneath
`--output-dir`) containing `config.json`, `full_metrics.csv`, `results.csv`, and
`subsets/*.csv`. Results include signed error (estimate minus full-test value)
and absolute error. Metrics are accuracy, macro-F1, macro precision/recall,
binary ROC-AUC, and top-label expected calibration error with 10 equal-width
confidence bins. Macro metrics always include every class from the saved mapping
and use zero for undefined precision/recall. ROC-AUC is empty when the model is
not binary or the evaluated sample lacks both classes. Probability column 1 is
the positive class. Calibration error is the bin-size-weighted absolute gap
between mean confidence and accuracy.

Manifests contain zero-based source CSV row IDs (excluding the header), budget,
repetition, seed, subset ID, and zero-based selection order. Keep the source CSV
unchanged to reuse these IDs; its SHA-256 is recorded. No images are copied.
Configuration, full metrics, results, and manifests are logged as artifacts in
one MLflow run under `--experiment-name active-testing`, using the same tracking
settings as training.

Visualize the estimates and their absolute error across subset sizes with:

```bash
uv run python -m active_testing_benchmark.plots reports/active-testing/<mlflow-run-id>

make active-testing-plot EXPERIMENT_DIR=reports/active-testing/<mlflow-run-id>
```

The figure is written to `active-testing-results.png` inside the experiment
directory. It contains one row per metric, shows every repetition as a faint dot,
connects the mean estimates, and marks each model's full-test value with a dashed
line. Use repeated `--metric` options to select metrics, `--x-axis count` for raw
sample counts, or `--output path/to/figure.pdf` to choose another path or format.

`Reducer` and `Metric` are lightweight callable protocols in `active_testing.py`.
Reducers can accept aligned model predictions and a mapping of previously
revealed labels; metrics accept labels, predictions, probabilities, and optional
metadata. The current shared-subset loop passes neither predictions nor labels
to the random reducer. A future iterative caller must manage earlier-round label
reveals and model-dependent subset provenance. No active method, plugin system,
or YAML configuration is implemented in this POC.

## Acknowledgment

The weekly meeting notes are inspired by the
[WhitakerLabProjectManagement](https://github.com/WhitakerLab/WhitakerLabProjectManagement)
repository. We thank its makers for sharing their approach.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
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
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
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
