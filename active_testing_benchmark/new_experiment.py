"""Compare ResNet-18 and EfficientNet-B0 for model selection, using no test images."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from time import perf_counter

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from active_testing_benchmark.config import PROJ_ROOT, RAW_DATA_DIR
from active_testing_benchmark.modeling.models import create_image_transform, create_model
from active_testing_benchmark.modeling.new import prepare_data
from active_testing_benchmark.modeling.new_train import (
    TrainingConfig,
    TrainingSetup,
    split_dataset,
    train_model,
)


@torch.inference_mode()
def evaluate_model(model, dataset, batch_size, device):
    """Evaluate a restored checkpoint on the external FairVision validation split."""
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        targets = batch["label"].long().to(device)
        logits = model(batch["image"].to(device))
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        labels.append(targets.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())

    y_true = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    predictions = scores.argmax(axis=1)
    num_classes = scores.shape[1]
    auc = None
    if np.array_equal(np.unique(y_true), np.arange(num_classes)):
        auc = (
            roc_auc_score(y_true, scores[:, 1])
            if num_classes == 2
            else roc_auc_score(y_true, scores, multi_class="ovr", average="macro")
        )
    return {
        "validation_loss": loss_sum / len(y_true),
        "validation_accuracy": accuracy_score(y_true, predictions),
        "validation_f1": f1_score(
            y_true,
            predictions,
            average="binary" if num_classes == 2 else "macro",
            labels=np.arange(num_classes),
            zero_division=0,
        ),
        "validation_auc": auc,
    }


def run_experiment(
    disease: str = "dr",
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    output_dir: Path = PROJ_ROOT / "reports" / "active-testing" / "model-comparison",
) -> pd.DataFrame:
    """Select checkpoints on 20% of train, then compare models on FairVision validation."""
    config = replace(training_config or TrainingConfig(), validation_fraction=0.2)
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
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        model_config = replace(config, model_type=model_type)
        model = create_model(model_type, out_dim=len(data.classes), use_pretrained=True).to(device)
        optimizer = AdamW(
            model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay
        )
        setup = TrainingSetup(
            model,
            optimizer=optimizer,
            scheduler=CosineAnnealingLR(optimizer, T_max=config.max_epochs),
        )
        # Reset after model initialization so loader shuffling starts from the same seed.
        torch.manual_seed(config.random_state)
        start = perf_counter()
        model, history = train_model(
            setup, model_config, fit, device=device, validation_data=holdout
        )
        training_seconds = perf_counter() - start
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
                "training_seconds": training_seconds,
                **metrics,
            }
        )
        histories[model_type] = history
        logger.info(f"{model_type}: {rows[-1]}")
        del model, setup, optimizer

    results = (
        pd.DataFrame(rows).sort_values("validation_auc", ascending=False).reset_index(drop=True)
    )
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    results = run_experiment(
        disease=args.disease,
        training_config=TrainingConfig(
            max_epochs=args.epochs, batch_size=args.batch_size, random_state=args.seed
        ),
        device=args.device,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
