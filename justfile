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
    ruff format --check
    ruff check

# Ruff linting and formatting
format:
    ruff check --fix
    ruff format

# Run all tests
test:
    uv run pytest tests

# Run experiment (currently only dev)
[arg("repititions", long="repititions")]
experiment repititions="1": requirements
    uv run active_testing_benchmark/experiment.py --repetitions {{repititions}}
