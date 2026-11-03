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

# Run the development experiment with full classifier training; accepts experiment CLI flags
experiment *args: requirements
    uv run python -m active_testing_benchmark.experiment "$@"

# Run the development experiment with classifier training capped at ten batches per phase
experiment-fast *args: requirements
    uv run python -m active_testing_benchmark.experiment --training-mode fast "$@"
