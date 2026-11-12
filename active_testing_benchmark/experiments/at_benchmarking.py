"""Run development or final active-testing benchmarking."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from pprint import pprint
import secrets
import uuid

from loguru import logger
import mlflow
import pandas as pd
import torch
from torch.utils.data import Subset

from active_testing_benchmark.benchmark import run_benchmark
from active_testing_benchmark.config import CLASS_TO_IDX, MODELS_DIR, PROJ_ROOT, RAW_DATA_DIR
from active_testing_benchmark.experiments.utils import (
    _log_evaluation,
    _train_or_load,
    evaluate_model,
)
from active_testing_benchmark.modeling.data import FairVisionData
from active_testing_benchmark.modeling.models import create_image_transform
from active_testing_benchmark.modeling.provenance import log_dataset_provenance
from active_testing_benchmark.modeling.registry import experiment_run
from active_testing_benchmark.modeling.train import (
    TrainingConfig,
    fit_auxiliary_models,
    split_dataset,
)


def save_results(curves, metadata, output_dir):
    """Save a uniquely identified CSV and JSON sidecar without overwriting prior runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex}"
    curves_path = output_dir / f"{run_id}.csv"
    metadata_path = output_dir / f"{run_id}.json"
    # Serialize before opening either destination so invalid metadata cannot leave partial output.
    payload = json.dumps(metadata, indent=2, allow_nan=False)
    with metadata_path.open("x") as stream:
        stream.write(payload + "\n")
    curves.to_csv(curves_path, index=False, mode="x")
    logger.info(f"Saved experiment curves to {curves_path}")
    return curves_path, metadata_path


def _run_experiment(
    train: FairVisionData | Subset[FairVisionData],
    validation: FairVisionData | Subset[FairVisionData],
    test: FairVisionData,
    config: TrainingConfig,
    device: torch.device | str | None,
    checkpoint_path: Path,
    reuse_trained_model: bool = False,
    *,
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "AT_benchmarking",
    experiment: str = "AT_dev",
    repetitions: int = 50,
    track: bool = True,
    ase_update_every: int = 50,
    protected_attribute: str | None = None,
):
    if ase_update_every <= 1:
        raise ValueError("ase_update_every must be positive")
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(
        f"Calling _run_experiment with: "
        f"train ({len(train)}), "
        f"val ({len(validation)}), "
        f"test ({len(test)})"
    )
    with experiment_run(experiment, config, test.disease, device, enabled=track) as run:
        if run is not None:
            log_dataset_provenance(train=train, validation=validation, evaluation=test)
            mlflow.log_params(
                {
                    "repetitions": repetitions,
                    "ase_update_every": ase_update_every,
                    "budget": len(test),
                    "train_size": len(train),
                    "validation_size": len(validation),
                    "evaluation_size": len(test),
                    "checkpoint_selection": "max_validation_auc",
                }
            )
        model, checkpoint = _train_or_load(
            config, test.classes, train, validation, device, checkpoint_path, reuse_trained_model
        )

        if run is not None:
            mlflow.set_tag("stage", "evaluation")
        reference_metrics = evaluate_model(model, test, config.batch_size, device)
        _log_evaluation(reference_metrics)
        available_protected_columns = tuple(
            column for column in reference_metrics["subgroup_metrics"] if column != "age"
        )
        if (
            protected_attribute is not None
            and protected_attribute not in available_protected_columns
        ):
            raise ValueError(
                f"Protected attribute {protected_attribute!r} is not available in the evaluation data"
            )
        protected_columns = available_protected_columns
        if run is not None:
            mlflow.set_tag("stage", "auxiliary_fit")
            mlflow.log_param("protected_columns", json.dumps(protected_columns))
            mlflow.log_param("stratification_attribute", protected_attribute or "all")

        auxiliary = fit_auxiliary_models(train, test.classes, config, device)

        curves = run_benchmark(
            model=model,
            dataset=test,
            auxiliary=auxiliary,
            num_classes=len(test.classes),
            device=device,
            seed=config.random_state,
            protected_columns=protected_columns,
            stratification_attribute=protected_attribute,
            repetitions=repetitions,
            ase_update_every=ase_update_every,
            budget=len(test),
        )

        metadata = {
            "mlflow_run_id": run.info.run_id if run is not None else None,
            "configuration": {
                "experiment": experiment,
                "disease": test.disease,
                "training": asdict(config),
                "seed": config.random_state,
                "data_root": str(test.root_dir.parent),
                "output_dir": str(output_dir),
                "resolved_device": str(device),
                "reuse_trained_model": reuse_trained_model,
                "protected_columns": list(protected_columns),
                "stratification_attribute": protected_attribute,
                "repetitions": repetitions,
                "ase_update_every": ase_update_every,
                "budget": len(test),
            },
            "training": {
                "configuration": checkpoint["config"],
                "history": checkpoint["history"],
                "training_seconds": checkpoint["training_seconds"],
                "checkpoint_path": str(checkpoint_path),
                "source_run_id": checkpoint.get("training_run_id"),
            },
            "selected_epochs": max(
                (row for row in checkpoint["history"] if row["roc_auc"] is not None),
                key=lambda row: row["roc_auc"],
            )["epoch"],
            "classes": test.classes,
            "auxiliary_seeds": list(auxiliary.seeds),
            "train_size": len(train),
            "validation_size": len(validation),
            "evaluation_size": len(test),
            "reference_metrics": reference_metrics,
        }
        curves_path, metadata_path = save_results(curves, metadata, output_dir)
        if run is not None:
            mlflow.log_artifact(str(curves_path), "results")
            mlflow.log_artifact(str(metadata_path), "results")
            mlflow.log_params(
                {f"auxiliary_seed_{i}": seed for i, seed in enumerate(auxiliary.seeds)}
            )

        return reference_metrics


def run_dev_experiment(
    disease: str = "dr",
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "AT_benchmarking",
    reuse_trained_model: bool = False,
    checkpoint_dir: Path = MODELS_DIR / "experiment",
    repetitions: int = 50,
    track: bool = True,
    ase_update_every: int = 50,
    protected_attribute: str | None = None,
):
    # Set defaults if no training config is given
    config = training_config or TrainingConfig(random_state=secrets.randbits(32))

    train = FairVisionData(
        disease=disease,
        split="train",
        transform=create_image_transform(config.image_size, training=False),
        class_to_idx=CLASS_TO_IDX[disease.lower()],
        fairvision_data_root=data_root,
    )

    validation = FairVisionData(
        disease=disease,
        split="validation",
        transform=create_image_transform(config.image_size, training=False),
        class_to_idx=CLASS_TO_IDX[disease.lower()],
        fairvision_data_root=data_root,
    )

    # Create train and validation set for training
    fit, holdout = next(
        split_dataset(
            train,
            train.labels,
            validation_fraction=config.validation_fraction,
            random_state=config.random_state,
        )
    )

    return _run_experiment(
        fit,
        holdout,
        validation,
        config,
        device,
        Path(checkpoint_dir) / "AT_dev" / disease.lower() / f"{config.model_type}.pt",
        reuse_trained_model,
        output_dir=output_dir,
        experiment="AT_dev",
        repetitions=repetitions,
        track=track,
        ase_update_every=ase_update_every,
        protected_attribute=protected_attribute,
    )


def run_final_experiment(
    disease: str = "dr",
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "AT_benchmarking",
    reuse_trained_model: bool = False,
    checkpoint_dir: Path = MODELS_DIR / "experiment",
    repetitions: int = 25,
    track: bool = True,
    ase_update_every: int = 50,
    protected_attribute: str | None = None,
):
    try:
        confirmation = input(
            "This will run the FINAL experiment using the held-out FairVision test set. "
            "Type FINAL to confirm: "
        )
    except (EOFError, KeyboardInterrupt):
        logger.info("Final experiment cancelled")
        return {"CANCELED": "Final experiment cancelled"}
    if confirmation.strip() != "FINAL":
        logger.info("Final experiment cancelled")
        return {"CANCELED": "Final experiment cancelled"}

    # Set defaults if no training config is given
    config = training_config or TrainingConfig(random_state=secrets.randbits(32))

    train = FairVisionData(
        disease=disease,
        split="train",
        transform=create_image_transform(config.image_size, training=False),
        class_to_idx=CLASS_TO_IDX[disease.lower()],
        fairvision_data_root=data_root,
    )

    validation = FairVisionData(
        disease=disease,
        split="validation",
        transform=create_image_transform(config.image_size, training=False),
        class_to_idx=CLASS_TO_IDX[disease.lower()],
        fairvision_data_root=data_root,
    )

    test = FairVisionData(
        disease=disease,
        split="test",
        transform=create_image_transform(config.image_size, training=False),
        class_to_idx=CLASS_TO_IDX[disease.lower()],
        fairvision_data_root=data_root,
    )

    return _run_experiment(
        train,
        validation,
        test,
        config,
        device,
        Path(checkpoint_dir) / "AT_final" / disease.lower() / f"{config.model_type}.pt",
        reuse_trained_model,
        output_dir=output_dir,
        experiment="AT_final",
        repetitions=repetitions,
        track=track,
        ase_update_every=ase_update_every,
        protected_attribute=protected_attribute,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--FINAL", action="store_true", help="Run final AT benchmarking on test, with confirmation"
    )
    parser.add_argument("--disease", choices=("dr", "amd", "glaucoma"), default="dr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--repetitions", type=int, default=50, help="Number of AT benchmark repetitions"
    )
    parser.add_argument(
        "--ase-update-every",
        type=int,
        default=100,
        help="Refit ASE surrogates after every N acquired labels (default: 50)",
    )
    parser.add_argument(
        "--protected-attribute",
        choices=("gender", "race", "ethnicity", "language", "maritalstatus", "age_group"),
        default=None,
        help="Stratify by one attribute (default: run each available attribute)",
    )
    parser.add_argument("--seed", type=int, help="Random seed (generated when omitted)")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--reuse-trained-model",
        action="store_true",
        help="Load the saved best checkpoint instead of training",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=MODELS_DIR / "experiment")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Run locally without MLflow tracking"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.ase_update_every < 1:
        parser.error("--ase-update-every must be positive")
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    logger.info(f"Experiment seed: {seed}")

    runner = run_final_experiment if args.FINAL else run_dev_experiment
    results = runner(
        disease=args.disease,
        training_config=TrainingConfig(
            max_epochs=args.epochs, batch_size=args.batch_size, random_state=seed
        ),
        device=args.device,
        reuse_trained_model=args.reuse_trained_model,
        checkpoint_dir=args.checkpoint_dir,
        track=not args.no_mlflow,
        repetitions=args.repetitions,
        ase_update_every=args.ase_update_every,
        protected_attribute=args.protected_attribute,
    )
    if not isinstance(results, pd.DataFrame):
        pprint(results)
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
