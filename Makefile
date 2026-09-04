#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = active-testing-benchmark
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download FairVision data (set DISEASE=amd, dr, glaucoma, or all)
.PHONY: data
data: requirements
	@test -n "$(DISEASE)" || (echo "Set DISEASE, e.g. make data DISEASE=glaucoma" && exit 2)
	uv run active_testing_benchmark/dataset.py --disease $(DISEASE) $(FAIRVISION_FLAGS)

## Download FairVision metadata for one disease (set DISEASE=amd, dr, or glaucoma)
.PHONY: fairvision-metadata
fairvision-metadata: requirements
	uv run active_testing_benchmark/dataset.py --disease $(or $(DISEASE),glaucoma) --metadata-only


## Train an image classifier (set MODEL=cnn, resnet18, densenet121, vit, or clip)
.PHONY: train
train: requirements
	uv run python -m active_testing_benchmark.modeling.train \
		--model $(or $(MODEL),resnet18) \
		--epochs $(or $(EPOCHS),5) \
		--batch-size $(or $(BATCH_SIZE),32) \
		--learning-rate $(or $(LEARNING_RATE),1e-4) \
		--imbalance-strategy $(or $(IMBALANCE_STRATEGY),none) \
		$(TRAIN_FLAGS)


## Start the local MLflow UI using the project SQLite tracking database
.PHONY: mlflow-ui
mlflow-ui: requirements
	uv run mlflow db upgrade sqlite:///mlflow.db
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
