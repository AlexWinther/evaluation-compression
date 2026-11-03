"""Explicit orchestration of training and finite-pool active-testing experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import datetime
import json
from pathlib import Path
import secrets
from typing import Literal
import uuid

from loguru import logger
import pandas as pd
from torch.utils.data import DataLoader

from active_testing_benchmark.benchmark import run_benchmark
from active_testing_benchmark.config import RAW_DATA_DIR, RESULTS_DIR
from active_testing_benchmark.modeling.evaluation import evaluate
from active_testing_benchmark.modeling.preparation import prepare_data
from active_testing_benchmark.modeling.train import (
    TrainingConfig,
    fit_auxiliary_models,
    select_device,
    train_classifier,
    train_classifier_fast,
)


@dataclass(frozen=True)
class ExperimentConfig:
    disease: str = "dr"
    dev_splits: tuple[str, ...] = ("train",)
    eval_splits: tuple[str, ...] = ("val",)
    training_mode: Literal["full", "fast"] = "full"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int | None = None
    repetitions: int = 1
    budget: int = 1000
    class_balanced_at: bool = True
    include_accuracy: bool = False
    device: str = "auto"
    data_root: Path = RAW_DATA_DIR / "fairvision"
    output_dir: Path = RESULTS_DIR / "active-testing"

    def validate(self):
        if self.training_mode not in {"full", "fast"}:
            raise ValueError("training_mode must be full or fast")
        if self.seed is not None and not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 4294967295")
        if self.repetitions < 1 or self.budget < 1:
            raise ValueError("repetitions and budget must be positive")
        if not self.dev_splits or not self.eval_splits:
            raise ValueError("development and evaluation splits must be nonempty")
        if "test" in self.dev_splits or "test" in self.eval_splits:
            raise RuntimeError(
                "Final test-set experiments remain blocked pending explicit authorization"
            )


@dataclass
class ExperimentResult:
    curves: pd.DataFrame
    reference_metrics: dict
    training_metadata: dict
    configuration: ExperimentConfig
    curves_path: Path
    metadata_path: Path


def save_results(curves, metadata, output_dir):
    """Save a uniquely identified CSV and JSON sidecar without overwriting prior runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.datetime.now().astimezone():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex}"
    curves_path = output_dir / f"{run_id}.csv"
    metadata_path = output_dir / f"{run_id}.json"
    # Serialize before opening either destination so invalid metadata cannot leave partial output.
    payload = json.dumps(metadata, indent=2, allow_nan=False)
    with metadata_path.open("x") as stream:
        stream.write(payload + "\n")
    curves.to_csv(curves_path, index=False, mode="x")
    logger.info(f"Saved experiment curves to {curves_path}")
    return curves_path, metadata_path


def run_experiment(
    config: ExperimentConfig | None = None, *, model_factory=None
) -> ExperimentResult:
    """Prepare data, fit models, evaluate reference risks, run AT, and persist artifacts."""
    config = config or ExperimentConfig()
    config.validate()
    seed = config.seed if config.seed is not None else secrets.randbits(32)
    config = replace(config, seed=seed, training=replace(config.training, seed=seed))
    device = select_device(config.device)
    logger.info(f"Experiment master seed: {seed}; training mode: {config.training_mode}")

    data = prepare_data(
        disease=config.disease,
        dev_splits=config.dev_splits,
        eval_splits=config.eval_splits,
        image_size=config.training.image_size,
        seed=seed,
        fairvision_data_root=config.data_root,
    )
    if config.budget > len(data.evaluation):
        raise ValueError("budget must not exceed evaluation pool size")

    trainer = train_classifier_fast if config.training_mode == "fast" else train_classifier
    trained = trainer(data, config.training, device, model_factory=model_factory)
    auxiliary = fit_auxiliary_models(data, config.training, device)

    reference = evaluate(
        trained.model,
        DataLoader(
            data.evaluation,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
        ),
        device,
        protected_column="gender",
        class_weights=trained.risk_weights if config.class_balanced_at else None,
    )
    reference_metrics = {key: value for key, value in reference.items() if key != "predictions"}
    curves = run_benchmark(
        model=trained.model,
        dataset=data.evaluation,
        auxiliary=auxiliary,
        num_classes=len(data.classes),
        device=device,
        seed=seed,
        repetitions=config.repetitions,
        budget=config.budget,
        class_weights=trained.risk_weights if config.class_balanced_at else None,
        include_accuracy=config.include_accuracy,
    )

    metadata = {
        "configuration": {
            **asdict(config),
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
            "resolved_device": str(device),
        },
        "training": trained.metadata,
        "selected_epochs": trained.selected_epochs,
        "classes": data.classes,
        "class_counts": trained.class_counts.cpu().tolist(),
        "class_prevalences": (trained.class_counts / trained.class_counts.sum()).cpu().tolist(),
        "risk_weights": trained.risk_weights.cpu().tolist(),
        "auxiliary_seeds": list(auxiliary.seeds),
        "evaluation_size": len(data.evaluation),
        "reference_metrics": reference_metrics,
    }
    curves_path, metadata_path = save_results(curves, metadata, config.output_dir)
    return ExperimentResult(
        curves, reference_metrics, trained.metadata, config, curves_path, metadata_path
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=["dr", "glaucoma", "amd"], default="dr")
    parser.add_argument("--training-mode", choices=["full", "fast"], default="full")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--weighted-sampling", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--selection-objective",
        choices=["class_balanced_cross_entropy", "unweighted_cross_entropy"],
        default="class_balanced_cross_entropy",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--class-balanced-at", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-accuracy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR / "fairvision")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "active-testing")
    args = parser.parse_args(argv)
    try:
        training = TrainingConfig(
            **{
                name: getattr(args, name)
                for name in (
                    "epochs",
                    "batch_size",
                    "image_size",
                    "learning_rate",
                    "weight_decay",
                    "num_workers",
                    "weighted_loss",
                    "weighted_sampling",
                    "pretrained",
                    "selection_objective",
                )
            }
        )
        config = ExperimentConfig(
            training=training,
            **{
                name: getattr(args, name)
                for name in (
                    "disease",
                    "training_mode",
                    "repetitions",
                    "budget",
                    "seed",
                    "device",
                    "class_balanced_at",
                    "include_accuracy",
                    "data_root",
                    "output_dir",
                )
            },
        )
        config.validate()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return run_experiment(config)


if __name__ == "__main__":
    main()
