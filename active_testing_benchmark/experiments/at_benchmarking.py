"""Run development or final active-testing benchmarking."""

import argparse
from dataclasses import asdict, replace
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

from active_testing_benchmark.active_testing.diagnostics import surrogate_diagnostics
from active_testing_benchmark.benchmark import DEFAULT_ACQUISITION_BATCH_SIZE, run_benchmark
from active_testing_benchmark.config import CLASS_TO_IDX, MODELS_DIR, PROJ_ROOT, RAW_DATA_DIR
from active_testing_benchmark.experiments.utils import (
    _log_evaluation,
    _train_or_load,
    ensemble_cache_dir,
    evaluate_model,
    load_checkpoint,
    reused_checkpoint_seed,
    selected_history_row,
)
from active_testing_benchmark.modeling.data import FairVisionData
from active_testing_benchmark.modeling.models import create_image_transform
from active_testing_benchmark.modeling.provenance import log_dataset_provenance
from active_testing_benchmark.modeling.registry import experiment_run, log_metrics
from active_testing_benchmark.modeling.train import (
    CHECKPOINT_METRICS,
    LR_SCHEDULES,
    ORACLE_KINDS,
    PROXY_KINDS,
    SURROGATE_KINDS,
    TrainingConfig,
    fit_auxiliary_models,
    split_dataset,
    stored_training_config,
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


def _checkpoint_path(checkpoint_dir: Path, experiment: str, disease: str, model_type: str) -> Path:
    """Return where an AT experiment saves, and reuses, its best checkpoint."""
    return Path(checkpoint_dir) / experiment / disease.lower() / f"{model_type}.pt"


def _resolve_config(
    training_config: TrainingConfig | None,
    checkpoint_path: Path,
    reuse_trained_model: bool,
) -> TrainingConfig:
    """Use a fresh random seed for training, or the checkpoint's own config when reusing it.

    A reused checkpoint keeps the recipe it was trained with (epochs, batch size,
    optimiser settings, schedule and checkpoint selection), so the run records what
    actually produced the model, whatever the command line asked for. Only the
    `num_workers` of the request is kept, since it does not change the model.
    """
    if not reuse_trained_model:
        return training_config or TrainingConfig(random_state=secrets.randbits(32))
    config = training_config or TrainingConfig()
    requested_seed = training_config.random_state if training_config is not None else None
    seed = reused_checkpoint_seed(checkpoint_path, requested_seed)
    stored = stored_training_config(load_checkpoint(checkpoint_path)["config"])
    return replace(stored, random_state=seed, num_workers=config.num_workers)


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
    ase_update_every: int | None = None,
    protected_attribute: str | None = None,
    surrogate_kind: str = "forest",
    proxy_kind: str = "resnet_ensemble",
    pool_size: int | None = None,
    budget: int | None = None,
    oracle_noise: float = 0.0,
    acquisition_batch_size: int = DEFAULT_ACQUISITION_BATCH_SIZE,
):
    if ase_update_every is not None and ase_update_every < 1:
        raise ValueError("ase_update_every must be positive")
    # Without a pool size every repetition evaluates the whole evaluation split (a frozen
    # pool). With one, every repetition draws a new class-stratified pool from it.
    evaluation_pool_size = len(test) if pool_size is None else int(pool_size)
    if not 1 <= evaluation_pool_size <= len(test):
        raise ValueError(f"pool_size must be between 1 and the {len(test)} evaluation images")
    label_budget = evaluation_pool_size if budget is None else int(budget)
    if not 1 <= label_budget <= evaluation_pool_size:
        raise ValueError(f"budget must be between 1 and the pool size {evaluation_pool_size}")
    pool_sampling = "fixed" if pool_size is None else "class_stratified"
    ase_refits = (
        "none (fixed deep ensemble)"
        if surrogate_kind == "resnet_ensemble"
        else "none (fixed oracle)"
        if surrogate_kind in ORACLE_KINDS
        else "kossen_schedule"
        if ase_update_every is None
        else f"every {ase_update_every}"
    )
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
                    "ase_refits": ase_refits,
                    "surrogate_kind": surrogate_kind,
                    "proxy_kind": proxy_kind,
                    "oracle_noise": oracle_noise,
                    "acquisition_batch_size": acquisition_batch_size,
                    "budget": label_budget,
                    "pool_size": evaluation_pool_size,
                    "pool_sampling": pool_sampling,
                    "train_size": len(train),
                    "validation_size": len(validation),
                    "evaluation_size": len(test),
                    "checkpoint_selection": config.checkpoint_metric,
                    "early_stopping_patience": config.early_stopping_patience,
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

        auxiliary = fit_auxiliary_models(
            train,
            validation,
            test.classes,
            config,
            device,
            surrogate_kind=surrogate_kind,
            proxy_kind=proxy_kind,
            ensemble_dir=ensemble_cache_dir(checkpoint_path, config.random_state),
            oracle_noise=oracle_noise,
        )

        # Measure surrogate and proxy quality on the checkpoint-selection holdout,
        # never on the evaluation pool, so that no evaluation labels are used. The
        # holdout also calibrated the surrogates, so these values are slightly optimistic.
        if run is not None:
            mlflow.set_tag("stage", "diagnostics")
        diagnostics = surrogate_diagnostics(
            model, auxiliary.surrogates, auxiliary.proxy, validation, device, config.batch_size
        )
        logger.info(f"Surrogate diagnostics on the holdout: {diagnostics}")
        log_metrics(diagnostics, prefix="diagnostics")

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
            budget=label_budget,
            pool_size=pool_size,
            acquisition_batch_size=acquisition_batch_size,
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
                "ase_refits": ase_refits,
                "surrogate_kind": surrogate_kind,
                "proxy_kind": proxy_kind,
                "oracle_noise": oracle_noise,
                "acquisition_batch_size": acquisition_batch_size,
                "budget": label_budget,
                "pool_size": evaluation_pool_size,
                "pool_sampling": pool_sampling,
            },
            "training": {
                "configuration": checkpoint["config"],
                "history": checkpoint["history"],
                "training_seconds": checkpoint["training_seconds"],
                "checkpoint_path": str(checkpoint_path),
                "source_run_id": checkpoint.get("training_run_id"),
            },
            "selected_epochs": selected_history_row(checkpoint)["epoch"],
            "classes": test.classes,
            "auxiliary_seeds": list(auxiliary.seeds),
            "train_size": len(train),
            "validation_size": len(validation),
            "evaluation_size": len(test),
            "reference_metrics": reference_metrics,
            "surrogate_diagnostics": diagnostics,
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
    ase_update_every: int | None = None,
    protected_attribute: str | None = None,
    surrogate_kind: str = "forest",
    proxy_kind: str = "resnet_ensemble",
    pool_size: int | None = None,
    budget: int | None = None,
    oracle_noise: float = 0.0,
    acquisition_batch_size: int = DEFAULT_ACQUISITION_BATCH_SIZE,
):
    model_type = (training_config or TrainingConfig()).model_type
    checkpoint_path = _checkpoint_path(checkpoint_dir, "AT_dev", disease, model_type)
    config = _resolve_config(training_config, checkpoint_path, reuse_trained_model)

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
        checkpoint_path,
        reuse_trained_model,
        output_dir=output_dir,
        experiment="AT_dev",
        repetitions=repetitions,
        track=track,
        ase_update_every=ase_update_every,
        protected_attribute=protected_attribute,
        surrogate_kind=surrogate_kind,
        proxy_kind=proxy_kind,
        oracle_noise=oracle_noise,
        pool_size=pool_size,
        budget=budget,
        acquisition_batch_size=acquisition_batch_size,
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
    ase_update_every: int | None = None,
    protected_attribute: str | None = None,
    surrogate_kind: str = "forest",
    proxy_kind: str = "resnet_ensemble",
    pool_size: int | None = None,
    budget: int | None = None,
    oracle_noise: float = 0.0,
    acquisition_batch_size: int = DEFAULT_ACQUISITION_BATCH_SIZE,
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

    model_type = (training_config or TrainingConfig()).model_type
    checkpoint_path = _checkpoint_path(checkpoint_dir, "AT_final", disease, model_type)
    config = _resolve_config(training_config, checkpoint_path, reuse_trained_model)

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
        checkpoint_path,
        reuse_trained_model,
        output_dir=output_dir,
        experiment="AT_final",
        repetitions=repetitions,
        track=track,
        ase_update_every=ase_update_every,
        protected_attribute=protected_attribute,
        surrogate_kind=surrogate_kind,
        proxy_kind=proxy_kind,
        oracle_noise=oracle_noise,
        pool_size=pool_size,
        budget=budget,
        acquisition_batch_size=acquisition_batch_size,
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
        default=None,
        help=(
            "Refit the forest surrogates of ASE every N acquired labels "
            "(default: the retraining schedule of Kossen et al., 2021)"
        ),
    )
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_KINDS + ORACLE_KINDS,
        default="forest",
        help=(
            "Surrogate ensemble: five bootstrapped forests or five fine-tuned ResNets. "
            "The oracle kinds are control conditions that read the true labels"
        ),
    )
    parser.add_argument(
        "--proxy",
        choices=PROXY_KINDS + ORACLE_KINDS,
        default="resnet_ensemble",
        help="PPAT proxy labeller, shared by both surrogate conditions",
    )
    parser.add_argument(
        "--oracle-noise",
        type=float,
        default=0.0,
        help=(
            "Noise level in [0, 1] of the noisy oracle kinds: the label flip probability "
            "(noisy_oracle) or the share of items that fall back to the model's own "
            "prediction (model_noisy_oracle)"
        ),
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help=(
            "Draw a new class-stratified pool of N evaluation images in every repetition "
            "(default: a frozen pool of the whole evaluation split)"
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Labels acquired per repetition (default: the whole pool)",
    )
    parser.add_argument(
        "--acquisition-batch-size",
        type=int,
        default=DEFAULT_ACQUISITION_BATCH_SIZE,
        help=(
            "Labels every method acquires per step before its proposal is recomputed "
            f"(default: {DEFAULT_ACQUISITION_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--protected-attribute",
        choices=("gender", "race", "ethnicity", "language", "maritalstatus", "age_group"),
        default=None,
        help="Stratify by one attribute (default: run each available attribute)",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=CHECKPOINT_METRICS,
        default="roc_auc",
        help="Validation metric that selects the saved checkpoint (default: roc_auc)",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="constant",
        help=(
            "Learning-rate schedule of the model under test (default: constant, as in "
            "FairVision). The deep-ensemble surrogate always uses cosine"
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Stop after N epochs without improvement of the checkpoint metric",
    )
    parser.add_argument("--seed", type=int, help="Random seed (generated when omitted)")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--reuse-trained-model",
        action="store_true",
        help="Load the saved best checkpoint instead of training. Uses its training seed",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=MODELS_DIR / "experiment")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Run locally without MLflow tracking"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.ase_update_every is not None and args.ase_update_every < 1:
        parser.error("--ase-update-every must be positive")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be positive")
    if not 0.0 <= args.oracle_noise <= 1.0:
        parser.error("--oracle-noise must be in [0, 1]")
    if args.pool_size is not None and args.pool_size < 1:
        parser.error("--pool-size must be positive")
    if args.acquisition_batch_size < 1:
        parser.error("--acquisition-batch-size must be positive")
    if args.budget is not None and args.budget < 1:
        parser.error("--budget must be positive")
    if args.pool_size is not None and args.budget is not None and args.budget > args.pool_size:
        parser.error("--budget cannot exceed --pool-size")
    if args.reuse_trained_model:
        # Fail before any prompt or data loading when the seed conflicts with the checkpoint.
        experiment_name = "AT_final" if args.FINAL else "AT_dev"
        checkpoint_path = _checkpoint_path(
            args.checkpoint_dir, experiment_name, args.disease, TrainingConfig().model_type
        )
        try:
            seed = reused_checkpoint_seed(checkpoint_path, args.seed)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        logger.info(f"Experiment seed: {seed} (from the reused checkpoint)")
    else:
        seed = args.seed if args.seed is not None else secrets.randbits(32)
        logger.info(f"Experiment seed: {seed}")

    runner = run_final_experiment if args.FINAL else run_dev_experiment
    results = runner(
        disease=args.disease,
        training_config=TrainingConfig(
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=seed,
            checkpoint_metric=args.checkpoint_metric,
            early_stopping_patience=args.early_stopping_patience,
            lr_schedule=args.lr_schedule,
        ),
        device=args.device,
        reuse_trained_model=args.reuse_trained_model,
        checkpoint_dir=args.checkpoint_dir,
        track=not args.no_mlflow,
        repetitions=args.repetitions,
        ase_update_every=args.ase_update_every,
        protected_attribute=args.protected_attribute,
        surrogate_kind=args.surrogate,
        proxy_kind=args.proxy,
        oracle_noise=args.oracle_noise,
        acquisition_batch_size=args.acquisition_batch_size,
        pool_size=args.pool_size,
        budget=args.budget,
    )
    if not isinstance(results, pd.DataFrame):
        pprint(results)
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
