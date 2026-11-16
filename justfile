# Forward recipe arguments to commands without shell interpolation.
set positional-arguments

_list:
    just --list

# sync environment to the uv lock file
requirements:
    uv sync --locked

# Remove cache and compiled files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Ruff linting checks
lint:
    uv run ruff format --check
    uv run ruff check

# Ruff linting and formatting
format:
    uv run ruff check --fix
    uv run ruff format

# Run all tests
test *args:
    uv run pytest "$@"

# Download FairVision SLO images and metadata for one disease (default: dr)
data disease="dr" *args: requirements
    dataset_disease="$1"; shift; uv run python -m active_testing_benchmark.dataset --disease "$dataset_disease" "$@"

# Download FairVision metadata without images (default: dr)
fairvision-metadata disease="dr" *args: requirements
    dataset_disease="$1"; shift; uv run python -m active_testing_benchmark.dataset --disease "$dataset_disease" --metadata-only "$@"

# Run the new AT development benchmark; accepts experiment CLI flags
experiment *args: requirements
    uv run python -m active_testing_benchmark.experiments.at_benchmarking "$@"

# Compare ResNet-18 and EfficientNet-B0 using the new training pipeline
model-selection *args: requirements
    uv run python -m active_testing_benchmark.experiments.model_selection "$@"

# Compare random forest image features on held-out training data
feature-experiment *args: requirements
    uv run python -m active_testing_benchmark.experiments.feature_experiment "$@"

# Compare candidate PPAT proxies (frozen-feature probes, FLAIR zero-shot) on a checkpoint's holdout
proxy-experiment *args: requirements
    uv run python -m active_testing_benchmark.experiments.proxy_experiment "$@"

# Fit surrogate conditions for a checkpoint and run the holdout diagnostics (pre-benchmark gate)
surrogate-experiment *args: requirements
    uv run python -m active_testing_benchmark.experiments.surrogate_experiment "$@"

# Diagnose a finished AT run's surrogates on its evaluation pool (post hoc, uses pool labels)
pool-diagnostics run *args: requirements
    uv run python -m active_testing_benchmark.experiments.pool_diagnostics "$@"

# Write the report tables and figures for one AT run (see active_testing_benchmark/reporting.py)
report-results run *args: requirements
    uv run python -m active_testing_benchmark.reporting "$@"

