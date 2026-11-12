"""Run model selection or development/final active-testing benchmarking."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from pprint import pprint
import random
import secrets
from time import perf_counter
import uuid

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

from active_testing_benchmark.benchmark import run_benchmark
from active_testing_benchmark.config import CLASS_TO_IDX, MODELS_DIR, PROJ_ROOT, RAW_DATA_DIR
from active_testing_benchmark.modeling.data import FairVisionData, prepare_data
from active_testing_benchmark.modeling.models import create_image_transform, create_model
from active_testing_benchmark.modeling.train import (
    TrainingConfig,
    TrainingSetup,
    fit_auxiliary_models,
    split_dataset,
    train_model,
)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device | str
) -> dict[str, object]:
    """Evaluate overall and per-subgroup loss, accuracy, F1, and AUC."""
    model.eval()
    labels, probabilities, losses = [], [], []
    subgroups = defaultdict(list)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        targets = batch["label"].long().to(device)
        logits = model(batch["image"].to(device))
        losses.append(nn.functional.cross_entropy(logits, targets, reduction="none").cpu().numpy())
        labels.append(targets.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        for name, values in batch.get("subgroups", {}).items():
            subgroups[name].extend(values)

    y_true = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    losses = np.concatenate(losses)

    def metrics(targets, probabilities, sample_losses):
        predictions = probabilities.argmax(axis=1)
        num_classes = probabilities.shape[1]
        auc = None
        if np.array_equal(np.unique(targets), np.arange(num_classes)):
            auc = (
                roc_auc_score(targets, probabilities[:, 1])
                if num_classes == 2
                else roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")
            )
        return {
            "loss": float(sample_losses.mean()),
            "accuracy": accuracy_score(targets, predictions),
            "f1": f1_score(
                targets,
                predictions,
                average="binary" if num_classes == 2 else "macro",
                labels=np.arange(num_classes),
                zero_division=0.0,
            ),
            "auc": float(auc) if auc is not None else None,
        }

    result = metrics(y_true, scores, losses)
    subgroup_metrics = {}
    for name, values in subgroups.items():
        values = np.asarray(values)
        subgroup_metrics[name] = {}
        for value in np.unique(values):
            mask = values == value
            subgroup_metrics[name][str(value)] = {
                "n": int(mask.sum()),
                **metrics(y_true[mask], scores[mask], losses[mask]),
            }
    result["subgroup_metrics"] = subgroup_metrics
    return result


def _train_or_load(config, classes, train, validation, device, checkpoint_path, reuse):
    if reuse:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["classes"] != classes or any(
            checkpoint["config"][name] != getattr(config, name)
            for name in ("model_type", "image_size")
        ):
            raise ValueError(
                f"Checkpoint does not match the model, classes or image size: {checkpoint_path}"
            )
        model = create_model(config.model_type, out_dim=len(classes), use_pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info(
            f"Loaded {checkpoint_path} (training seed: {checkpoint['config']['random_state']})"
        )
        return model, checkpoint

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    model = create_model(config.model_type, out_dim=len(classes), use_pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)
    setup = TrainingSetup(
        model,
        optimizer=optimizer,
        scheduler=CosineAnnealingLR(optimizer, T_max=config.max_epochs),
    )
    torch.manual_seed(config.random_state)
    start = perf_counter()
    model, history = train_model(setup, config, train, val_dataset=validation, device=device)
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "config": asdict(config),
        "classes": list(classes),
        "history": json.loads(json.dumps(history)),
        "training_seconds": perf_counter() - start,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved best checkpoint to {checkpoint_path}")

    return model, checkpoint


def run_model_selection_experiment(
    disease: str = "dr",
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "model-comparison",
    reuse_trained_model: bool = False,
    checkpoint_dir: Path = MODELS_DIR / "experiment",
) -> pd.DataFrame:
    """Select checkpoints on 20% of train, then compare models on FairVision validation."""
    config = replace(
        training_config or TrainingConfig(random_state=secrets.randbits(32)),
        validation_fraction=0.2,
    )
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    data = prepare_data(
        disease=disease,
        train_split="train",
        test_split="val",
        transform=create_image_transform(config.image_size, training=False),
        fairvision_data_root=data_root,
    )
    # Both models use exactly the same fit and checkpoint-selection samples.
    fit, holdout = next(
        split_dataset(
            data.train,
            data.train.labels,
            validation_fraction=0.2,
            random_state=config.random_state,
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}
    for model_type in ("resnet", "efficientnet"):
        model_config = replace(config, model_type=model_type)
        model, checkpoint = _train_or_load(
            model_config,
            data.classes,
            fit,
            holdout,
            device,
            Path(checkpoint_dir) / "model_selection" / disease.lower() / f"{model_type}.pt",
            reuse_trained_model,
        )
        history = checkpoint["history"]
        best = max(
            (row for row in history if row["roc_auc"] is not None),
            key=lambda row: row["roc_auc"],
        )

        # PreparedData.test contains "val" here, not the final FairVision test set.
        metrics = evaluate_model(model, data.test, config.batch_size, device)
        rows.append(
            {
                "model": model_type,
                "best_epoch": best["epoch"],
                "holdout_auc": best["roc_auc"],
                "training_seconds": checkpoint["training_seconds"],
                "training_seed": checkpoint["config"]["random_state"],
                **metrics,
            }
        )
        histories[model_type] = history
        logger.info(f"{model_type}: {rows[-1]}")
        del model, checkpoint

    results = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    results.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "history.json").write_text(
        json.dumps(
            {"disease": disease, "device": device, "config": asdict(config), "history": histories},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    logger.info(f"Comparison saved to {output_dir}")
    return results


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
    repetitions: int = 25,
):
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
    model, checkpoint = _train_or_load(
        config, test.classes, train, validation, device, checkpoint_path, reuse_trained_model
    )

    reference_metrics = evaluate_model(model, test, config.batch_size, device)

    auxiliary = fit_auxiliary_models(train, test.classes, config, device)

    curves = run_benchmark(
        model=model,
        dataset=test,
        auxiliary=auxiliary,
        num_classes=len(test.classes),
        device=device,
        seed=config.random_state,
        protected_columns=tuple(reference_metrics["subgroup_metrics"]),
        repetitions=repetitions,
        budget=len(test),
    )

    metadata = {
        "configuration": {
            "experiment": experiment,
            "disease": test.disease,
            "training": asdict(config),
            "seed": config.random_state,
            "data_root": str(test.root_dir.parent),
            "output_dir": str(output_dir),
            "resolved_device": str(device),
            "reuse_trained_model": reuse_trained_model,
            "protected_columns": list(reference_metrics["subgroup_metrics"]),
            "repetitions": repetitions,
            "budget": len(test),
        },
        "training": {
            "configuration": checkpoint["config"],
            "history": checkpoint["history"],
            "training_seconds": checkpoint["training_seconds"],
            "checkpoint_path": str(checkpoint_path),
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
    save_results(curves, metadata, output_dir)

    return reference_metrics


def run_dev_experiment(
    disease: str = "dr",
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "AT_benchmarking",
    reuse_trained_model: bool = False,
    checkpoint_dir: Path = MODELS_DIR / "experiment",
    repetitions: int = 25,
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
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("model_selection", "AT_benchmarking"), default="AT_benchmarking"
    )
    parser.add_argument(
        "--FINAL", action="store_true", help="Run final AT benchmarking on test, with confirmation"
    )
    parser.add_argument("--disease", choices=("dr", "amd", "glaucoma"), default="dr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--repetitions", type=int, default=25, help="Number of AT benchmark repetitions"
    )
    parser.add_argument("--seed", type=int, help="Random seed (generated when omitted)")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--reuse-trained-model",
        action="store_true",
        help="Load the saved best checkpoint instead of training",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=MODELS_DIR / "experiment")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.FINAL and args.experiment == "model_selection":
        parser.error("--FINAL is only available for AT_benchmarking")
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    logger.info(f"Experiment seed: {seed}")

    if args.experiment == "model_selection":
        runner = run_model_selection_experiment
    else:
        runner = run_final_experiment if args.FINAL else run_dev_experiment
    results = runner(
        disease=args.disease,
        training_config=TrainingConfig(
            max_epochs=args.epochs, batch_size=args.batch_size, random_state=seed
        ),
        device=args.device,
        reuse_trained_model=args.reuse_trained_model,
        checkpoint_dir=args.checkpoint_dir,
        **({"repetitions": args.repetitions} if args.experiment == "AT_benchmarking" else {}),
    )
    if not isinstance(results, pd.DataFrame):
        pprint(results)
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
