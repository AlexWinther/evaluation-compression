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
training baseline. Validation and test metrics always use ordinary cross-entropy.

`cnn` starts from scratch. Torchvision models use ImageNet weights; CLIP uses a
frozen OpenCLIP ViT-B/32 encoder with a trainable classification head. Pass
`--no-pretrained` to avoid weight downloads. Each run records metrics,
checkpoint, configuration, classes, and test confusion matrix in the local
SQLite-backed MLflow store. View runs from the project root with:

```bash
make mlflow-ui
```

For another layout, pass `--metadata-path`, `--image-root`, `--image-column`,
`--target-column`, and `--split-column`. Split values must be `training`,
`validation`, and `test` (case-insensitive).

radT is intentionally not installed or enabled yet. Its current published
materials describe an MLflow extension but do not document a stable minimal
in-process API for this loop. It can later be evaluated separately (`uv add
radt`) without affecting normal MLflow training.

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
