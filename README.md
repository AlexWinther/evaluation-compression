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

A higher AUC does not by itself make a better active-testing surrogate. To also
check how well each forest tracks the losses of the model under test, pass an
`AT_dev` checkpoint:

```bash
just feature-experiment --target-checkpoint models/experiment/AT_dev/dr/resnet.pt
```

Each extractor then also reports the
[surrogate diagnostics](#surrogate-diagnostics), with the forest acting as both
surrogate and proxy. The run uses the checkpoint's training seed, so the holdout
is the checkpoint's own selection holdout and contains no images the classifier
was trained on. Omit `--seed` to adopt it. A different `--seed` stops the run.
Add `--output <csv>` to save the comparison for the
[report tables](#report-tables-and-figures).

### Comparison of PPAT proxy candidates

PPAT only reduces variance when its proxy loss correlates with the true loss.
This experiment compares proxies that need no fine-tuning on the holdout of an
`AT_dev` checkpoint, before any benchmark:

```bash
just proxy-experiment --target-checkpoint models/experiment/AT_dev/dr/resnet.pt
just proxy-experiment --target-checkpoint models/experiment/AT_dev/dr/resnet.pt --candidates imagenet_probe flair_zero_shot --no-mlflow --output reports/active-testing/diagnostics/dr-proxy-candidates.csv
```

| Candidate | Proxy |
|---|---|
| `imagenet_probe` | Logistic regression on frozen ImageNet ResNet-18 features |
| `retfound_probe` | Logistic regression on frozen RETFound (MAE, colour fundus) ViT-L/16 features |
| `flair_probe` | Logistic regression on frozen FLAIR image embeddings |
| `flair_zero_shot` | FLAIR zero-shot with the prompts in `modeling/foundation.py` |

Probes train on the checkpoint's 80 % development split, and every candidate is
scored on its 20 % holdout with the proxy diagnostics of the
[surrogate diagnostics](#surrogate-diagnostics) table. Frozen features are
cached in `data/interim/proxy_features/<disease>/`, so each encoder runs once.

FLAIR and RETFound are not project dependencies:

- FLAIR comes from its repository, not the unrelated `flair` package on PyPI. The
  `just` recipes run `uv sync --locked`, which removes packages outside the lock
  file, so add FLAIR as a temporary layer instead of installing it. Its weights
  download from Hugging Face (`jusiro2/FLAIR`) on first use:

  ```bash
  uv run --with "flair @ git+https://github.com/jusiro/FLAIR.git" python -m active_testing_benchmark.experiments.proxy_experiment --target-checkpoint models/experiment/AT_dev/dr/resnet.pt --candidates flair_zero_shot flair_probe
  ```
- RETFound's weights (`YukunZhou/RETFound_mae_natureCFP`, about 4 GB) are gated.
  Request access on the Hugging Face page and log in with `hf auth login`.

Both were pretrained on colour fundus photographs, while FairVision provides SLO
images, so their transfer must be measured here before use.

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
highest ROC AUC by default. Ties keep the earliest checkpoint. The restored model is
evaluated and benchmarked on FairVision validation. Training follows the CNN
recipe of the FairVision paper (Luo et al., 2024, Sec. 5): 10 epochs, AdamW
with learning rate 1e-4 held constant, and the checkpoint with the highest
validation AUC. It deviates in three ways: batch size 32 instead of 6, weight
decay 1e-3 (the paper gives none), and selection on the 20% development holdout
instead of FairVision validation, which is the evaluation pool here. Training uses
pretrained ImageNet weights and unweighted cross-entropy. Pass
`--lr-schedule cosine` for the cosine schedule that earlier checkpoints were
trained with. The deep-ensemble surrogate always keeps its own cosine recipe.

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
runs for `--repetitions` (default 50).

All active methods share one surrogate ensemble of five members, chosen with
`--surrogate` (see `specs/surrogate-pipeline-review.md` for the reasoning):

| `--surrogate` | Members | Updated during acquisition |
|---|---|---|
| `forest` (default, cheap) | Five random forests on frozen ImageNet ResNet-18 features, each on its own bootstrap of the development split, without class weights, temperature-calibrated on the holdout | ASE refits them on the retraining schedule of Kossen et al. (2021): after 5, 10, 20, 30, 40, 100, 250, 400, ... labels |
| `resnet_ensemble` (expensive) | Five fine-tuned ResNet-18s with the classifier's recipe and new seeds, early-stopped on validation loss and temperature-scaled on the holdout | Never, as in the large-scale experiments of Kossen et al. (2021, 2022). ASE then reports a constant estimate |

Kossen averages the member probabilities. ASE uses the members for XWED and PPAT
for its RMS-residual proposal. Kossen and PPAT mix their proposals with 20 %
uniform sampling (the minimum proposal probability of Kossen et al., 2021), and
PPAT uses the fixed coefficient λ = 0.5.

The PPAT proxy is chosen with `--proxy` and is the same for both surrogate
conditions. `resnet_ensemble` (default) uses the argmax of the deep ensemble,
`imagenet_probe` a logistic-regression probe on frozen ImageNet features. The
deep ensemble is trained once per checkpoint seed and cached in
`models/experiment/<experiment>/<disease>/surrogate_ensemble/seed-<seed>/`. Its
members log their training curves under `surrogate_ensemble/member_<k>/` in the
run, so the classifier's own `training/` history stays separate.

Both `--surrogate` and `--proxy` also accept three control conditions that read
the true labels of the pool (`LabelOracle`). They show what the estimators gain
from the best possible auxiliary model and how fast that gain is lost. They are
never usable methods, since they need the labels that active testing saves.
`--oracle-noise` sets the noise level of the two noisy kinds:

| Kind | Output | `--oracle-noise` |
|---|---|---|
| `oracle` | The true label with certainty. The Kossen score equals the true loss and the proxy loss equals the true loss | Ignored |
| `noisy_oracle` | A simulated second grader whose mistakes are independent of the classifier. Surrogates receive the calibrated posterior given the noisy label, proxies the noisy label itself | Probability that a label is replaced by another class |
| `model_noisy_oracle` | The true label, or the classifier's own predicted distribution. Its mistakes coincide with the classifier's confident errors, as those of real surrogates tend to | Share of images that fall back to the classifier's prediction. At 1 the classifier is its own surrogate |

An oracle surrogate is a single fixed member, so ASE acquires by expected loss
instead of XWED. The surrogate and the proxy draw their noise from different
seeds. The holdout diagnostics are computed for oracles as well, so a sweep over
`--oracle-noise` relates surrogate quality to the estimation error.

The frozen ImageNet ResNet-18 features drop the classification layer and give
512 features per image. Backbone weights and batch-normalization statistics
remain frozen during fitting, prediction, and ASE refits. The first use
downloads approximately 45 MB of `ResNet18_Weights.IMAGENET1K_V1` weights to
torchvision's cache. Custom extractors remain supported through
`RandomForestModule(feature_extractor=...)`, including
`SpatialTextureFeatures()` for the previous texture baseline.

Set `--ase-update-every N` to refit the forests every N labels instead of on the
schedule. Acquisition batches stop at refit thresholds, so refits happen at the
requested label counts. Each objective and repetition starts with fresh copies
of refitted surrogates. A fixed surrogate is shared without copying.

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
just experiment --surrogate resnet_ensemble --seed 42
just experiment --surrogate forest --proxy imagenet_probe --seed 42
just experiment --surrogate oracle --proxy oracle --reuse-trained-model
just experiment --surrogate model_noisy_oracle --proxy noisy_oracle --oracle-noise 0.2 --reuse-trained-model
just experiment --protected-attribute gender --seed 42
just experiment --pool-size 500 --budget 150 --repetitions 100 --seed 42
just experiment --acquisition-batch-size 10 --budget 200 --seed 42
```

`--acquisition-batch-size` (default 25) sets how many labels every method
acquires before its proposal is recomputed. 10 labels is a 1% step on the
1,000-image development pool.

#### Evaluation pools and repetitions

By default the evaluation pool is frozen: every repetition evaluates the whole
evaluation split (FairVision validation for dev runs, test for final runs) and
only the acquisition randomness changes, as in PPAT (Ashouritaklimi et al.,
2026, app. B.1). With `--pool-size N`, every repetition instead draws a new pool
of N images from the split, without replacement and stratified by class, as in
Kossen et al. (2021, app. C.5). All methods of a repetition share its pool and
seed, and the pool depends only on `--seed` and the repetition number.
`--budget M` caps the labels per repetition and defaults to the whole pool.

Errors are measured against the exact risk of the pool that each repetition
evaluated. A frozen pool is the right reference for the unbiased estimators,
but a deterministic method such as ASE then gives the same answer in every
repetition, so its error is pure bias. Resampled pools also average that bias
over pools.

The model, surrogates and proxy stay fixed across repetitions, so their outputs
on the split are computed once and looked up by image index
(`active_testing/prediction_cache.py`). For forests the cache holds the frozen
image features, so ASE refits still predict from the same features. This makes
a repetition cost mostly the forest refits of ASE.

The sizes proposed for the thesis are a pool of 500 of the 1,000 validation
images with a budget of 150 for dev runs, and 1,000 of the 3,000 test images
with a budget of 300 for final runs.

#### Surrogate diagnostics

Before benchmarking, AT runs measure how well the surrogates and proxy track
the classifier's cross-entropy losses. These measurements use the
checkpoint-selection holdout, never the evaluation pool. Active testing can
only beat uniform sampling when its acquisition scores follow the true losses.
The diagnostics are:

| Name | Meaning |
|---|---|
| `kossen_score_spearman` | Rank correlation of the Kossen acquisition score with the true loss |
| `kossen_error_auc` | AUC of the Kossen score for separating the images the classifier misclassifies from the rest (0.5 is chance) |
| `kossen_variance_gain` | Uniform variance divided by Kossen proposal variance (above 1 is better than uniform) |
| `oracle_variance_gain` | The same ratio for a proposal proportional to the true loss (the upper limit) |
| `target_variance_gain` | The same ratio when the classifier is its own surrogate (Kossen without a surrogate, "Ori" in Lee et al., 2024) |
| `confidence_oracle_variance_gain` | The same ratio for the best proposal using only the classifier's confidence, proportional to the RMS loss per confidence bin (an optimistic in-sample estimate, not a strict limit, of what any surrogate-free or recalibrated proposal can reach, never below about 1) |
| `ase_relative_bias` | Relative error of the ASE estimate before acquiring labels |
| `surrogate_nll` | Negative log-likelihood of the labels under the ensemble mean (calibration and sharpness) |
| `surrogate_class_frequency_error` | Largest gap between the ensemble's mean class probability and the class frequency (calibration in the large, the direct cause of ASE bias) |
| `member_disagreement` | Mean fraction of images on which two members predict different classes |
| `xwed_spearman` | Rank correlation of the XWED score with the true loss. XWED targets where a label would most improve the surrogate, not where the loss is, so low or negative values are expected |
| `ppat_proxy_correlation` | Correlation of the PPAT proxy loss L(f(x), argmax g(x)) with the true loss (λ = 0.5 helps above 0.25) |
| `ppat_soft_proxy_correlation` | The same for the soft proxy loss, an ablation |
| `ppat_lambda_dagger` | The variance-minimising λ of PPAT Prop. 3.4 on the holdout. λ = 0.5 reduces variance when this exceeds 0.25. The largest possible gain is about the squared proxy correlation |
| `proxy_auc`, `proxy_accuracy`, `proxy_class_frequency_error` | Label quality of the proxy, including whether its hard labels ignore the rare class |

They are logged to MLflow under `diagnostics/` and saved in the metadata JSON
as `surrogate_diagnostics`. The holdout also calibrated the surrogates, so the
values are slightly optimistic.

#### Outputs and tracking

AT runs save uniquely named curves CSVs and matching metadata JSON files under
`reports/active-testing/AT_benchmarking/`. Curves contain methods, objectives,
repetitions, seeds, label counts, and overall/subgroup risk estimates. Each row
also stores its pool size, the pool size of every group (`pool_n_<group>`), and
the exact risk of its pool (`reference_risk` and `reference_risk <group>`).
Metadata records configuration (including `pool_size`, `pool_sampling` and
`budget`), checkpoint details, training history, auxiliary seeds, dataset sizes,
and reference metrics for the whole evaluation split.

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
budget, every requested subgroup has labels and an estimate. A small group can
be missing from a resampled pool, and then its columns are blank for that
repetition. Overall risk is
stored once per sampling trajectory.

During AT, the MLflow tags `stage`, `AT.method`, and `AT.repetition` show
progress. Risk and label-count metrics are logged after each method/repetition. Overall and subgroup reference metrics
are logged after evaluation. Full AT curves remain available in the CSV artifact.

#### Report tables and figures

Two steps turn a finished AT run into the tables and figures of the report. Pass
the run as its CSV, its JSON, or their shared stem:

```bash
just pool-diagnostics reports/active-testing/AT_benchmarking/<run>
just report-results reports/active-testing/AT_benchmarking/<run>
```

`pool-diagnostics` refits the run's surrogates from its seed and saves
`<run>-diagnostics.json` next to the run. It holds the surrogate diagnostics on
the classifier's own holdout and on the evaluation pool, and how concentrated
the classifier's loss is on the pool. The pool diagnostics use the pool labels,
so they explain a finished run and must not inform choices made before it.

`report-results` (`active_testing_benchmark/reporting.py`) writes the outputs
below. The prefix defaults to the disease. The report expects one prefix per
dataset and surrogate condition, `dr_forest`, `dr_resnet`, `glaucoma_forest`, and
`glaucoma_resnet`, so pass it with `--prefix`. The report includes each table and
figure once its file exists and shows a placeholder box until then.

| Output | Content |
|---|---|
| `report/figures/results/<prefix>_rmse_curves.pdf` | RMSE against label budget, overall and per group |
| `report/figures/tables/<prefix>_reference.tex` | Full-pool size, risk, and AUC per group |
| `report/figures/tables/<prefix>_rmse.tex` | RMSE at selected budgets, lowest per column in bold |
| `report/figures/tables/<prefix>_mse_ratio.tex` | MSE(uniform) / MSE(method) with paired bootstrap intervals |
| `report/figures/tables/<prefix>_allocation.tex` | Mean labels per group at one budget |
| `report/figures/tables/<prefix>_diagnostics.tex` | Surrogate diagnostics, when diagnostics are available |
| `reports/active-testing/analysis/<prefix>_*.csv` | The same numbers at full precision |

The `.tex` files hold only the `tabular`, so captions and labels stay in the
report. Options choose the attributes (`--attributes gender race`), the RMSE
table groups (`--table-targets overall gender=female race=black`), and the
budgets. `--feature-experiment <csv>` adds one holdout column per feature
extractor to the diagnostics table. The development DR tables that currently stand
in for the final `dr_forest` results were made with:

```bash
just report-results reports/active-testing/AT_benchmarking/20261112-221743-817504-9561f439132f44b58b1ee66c27cca9b5 --table-targets overall gender=female race=black --feature-experiment reports/active-testing/diagnostics/dr-feature-experiment-at-dev.csv --prefix dr_forest
```

#### Overview diagram

The overview figure of the report is drawn from `diagrams/overview.mmd` with
mermaid-cli (`brew install mermaid-cli`):

```bash
just overview-diagram
```

It writes `report/figures/overview.pdf`. mermaid-cli renders through Puppeteer,
which looks for its own `chrome-headless-shell` by default.
`diagrams/puppeteer-config.json` points it to the installed Google Chrome
instead, so change `executablePath` there if Chrome lives elsewhere.

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
needed. Checkpoint selection maximizes validation AUC by default.

Active testing estimates the cross-entropy, so an overconfident checkpoint
changes the risk being estimated, not only the ranking quality. To select the
checkpoint with the lowest validation cross-entropy instead, and to stop after
a number of epochs without improvement of that metric, pass:

```bash
just experiment --checkpoint-metric validation_loss --early-stopping-patience 3
```

Both values are stored in the checkpoint configuration, and the reported best
epoch follows the checkpoint's own selection metric.

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
It also records the checkpoint's stored training recipe instead of the
command-line training flags, which do not apply to a reused model.
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

### Querying MLflow from Claude Code

`.mcp.json` registers MLflow's own MCP server as `mlflow`, started by
`deploy/mlflow/mcp-server.sh`. It talks to the remote tracking server, never to
a local `mlflow.db`. The script uses `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` from the environment,
or from `.env` when they are not set. It exposes the experiment, run, and model
tools (`MLFLOW_MCP_TOOLS=ml`). `.claude/settings.json` enables the server,
allows the read-only tools, and denies the delete, rename, and update tools.

Cloud sessions do not have `.env`, so the cloud environment needs two settings:

1. Add the three `MLFLOW_TRACKING_*` variables as environment variables.
2. Allow network access to the MLflow host, either with a custom network policy
   that adds the host to the allowed domains or with full network access.

With these variables set, the experiment code in cloud sessions also logs to
the remote server.

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
