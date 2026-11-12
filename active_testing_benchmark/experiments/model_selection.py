"""Compare CNN architectures on FairVision validation data."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import secrets

from loguru import logger
import mlflow
import pandas as pd
import torch

from active_testing_benchmark.config import MODELS_DIR, PROJ_ROOT, RAW_DATA_DIR
from active_testing_benchmark.experiments.utils import (
    _log_evaluation,
    _train_or_load,
    evaluate_model,
)
from active_testing_benchmark.modeling.data import prepare_data
from active_testing_benchmark.modeling.models import create_image_transform
from active_testing_benchmark.modeling.provenance import log_dataset_provenance
from active_testing_benchmark.modeling.registry import experiment_run
from active_testing_benchmark.modeling.train import (
    TrainingConfig,
    split_dataset,
)


def run_model_selection_experiment(
    disease: str = "dr",
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "model-comparison",
    reuse_trained_model: bool = False,
    checkpoint_dir: Path = MODELS_DIR / "experiment",
    track: bool = True,
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
        with experiment_run(
            "model_selection", model_config, disease, device, enabled=track
        ) as run:
            if run is not None:
                log_dataset_provenance(train=fit, validation=holdout, evaluation=data.test)
                mlflow.log_params(
                    {
                        "train_size": len(fit),
                        "validation_size": len(holdout),
                        "evaluation_size": len(data.test),
                        "checkpoint_selection": "max_validation_auc",
                    }
                )
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
            if run is not None:
                mlflow.set_tag("stage", "evaluation")
            metrics = evaluate_model(model, data.test, config.batch_size, device)
            _log_evaluation(metrics)
            rows.append(
                {
                    "model": model_type,
                    "mlflow_run_id": run.info.run_id if run is not None else None,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=("dr", "amd", "glaucoma"), default="dr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--reuse-trained-model", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=MODELS_DIR / "experiment")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    logger.info(f"Experiment seed: {seed}")
    results = run_model_selection_experiment(
        disease=args.disease,
        training_config=TrainingConfig(
            max_epochs=args.epochs, batch_size=args.batch_size, random_state=seed
        ),
        device=args.device,
        reuse_trained_model=args.reuse_trained_model,
        checkpoint_dir=args.checkpoint_dir,
        track=not args.no_mlflow,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
