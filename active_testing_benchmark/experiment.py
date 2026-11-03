import argparse
import datetime
import json
import pprint
import secrets
import uuid

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing import (
    ActiveSurrogateEstimator,
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
    StratifiedSampleMean,
    UniformSampleMean,
    run_active_evaluation,
)
from active_testing_benchmark.active_testing.surrogate import RandomForestModule
from active_testing_benchmark.config import RESULTS_DIR
from active_testing_benchmark.modeling.data import FairVisionDataset
from active_testing_benchmark.modeling.models import (
    create_image_transform,
    create_model,
)
from active_testing_benchmark.modeling.registry import configure_mlflow
from active_testing_benchmark.modeling.train import (
    TrainConfig,
    evaluate,
    make_class_balanced_weights,
    make_train_loader_and_loss,
    select_device,
    set_seed,
    train_one_epoch,
)

# set up pretty printer
pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)


def run_experiment(
    dev_splits: str | tuple[str, ...] = "train",
    eval_splits: str | tuple[str, ...] = "val",
    disease: str = "dr",
    repetitions: int = 1,
    seed: int | None = None,
    class_balanced_at: bool = True,
):
    """Pilot-select an epoch count, refit, then estimate unweighted and balanced risks.

    Class-balanced risk weights are computed from development/training class counts only.
    """
    if seed is None:
        seed = secrets.randbits(32)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")
    logger.info(f"Experiment master seed: {seed}")

    output_dir = RESULTS_DIR / "active-testing"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.datetime.now().astimezone():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex}"
    output_path = output_dir / f"{disease}-budget-curves-{run_id}.csv"
    while output_path.exists():
        run_id = f"{datetime.datetime.now().astimezone():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex}"
        output_path = output_dir / f"{disease}-budget-curves-{run_id}.csv"

    configure_mlflow()

    config = TrainConfig(
        batch_size=32,
        num_workers=0,
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        seed=seed,
        image_size=224,
        weighted_sampling=False,
        weighted_loss=True,
        sampling_power=1.0,
        loss_power=1.0,
    )

    device = select_device()
    set_seed(config.seed, device=device)

    # Loading of the datasets
    dev_split_values = (dev_splits,) if isinstance(dev_splits, str) else tuple(dev_splits)
    if not dev_split_values:
        raise ValueError("dev_splits must contain at least one split")
    dev_split_names = ", ".join(dev_split_values)
    dev_dataset = FairVisionDataset(
        disease=disease,
        split=dev_split_values,
        transform=create_image_transform(image_size=config.image_size, training=True),
    )
    logger.info(
        f"Loaded {len(dev_dataset):,} samples for development "
        f"({dev_split_names}) dataset for {disease}"
    )

    eval_split_values = (eval_splits,) if isinstance(eval_splits, str) else tuple(eval_splits)
    eval_split_names = ", ".join(eval_split_values)
    active_eval_dataset = FairVisionDataset(
        disease=disease,
        split=eval_split_values,
        transform=create_image_transform(image_size=config.image_size, training=False),
    )
    logger.info(
        f"Loaded {len(active_eval_dataset):,} samples for evaluation "
        f"({eval_split_names}) dataset for {disease}"
    )

    if len(dev_split_values) == 1:
        if len(eval_split_values) == 1 and eval_split_values[0] in {"val", "validation"}:
            reference_validation_size = len(active_eval_dataset)
        else:
            reference_validation_dataset = FairVisionDataset(
                disease=disease,
                split="val",
                transform=create_image_transform(image_size=config.image_size, training=False),
            )
            reference_validation_size = len(reference_validation_dataset)
        validation_fraction = reference_validation_size / (
            len(dev_dataset) + reference_validation_size
        )
        fit_indices, validation_indices = train_test_split(
            np.arange(len(dev_dataset)),
            test_size=validation_fraction,
            random_state=config.seed,
            stratify=dev_dataset.targets,
        )
        pilot_fit_dataset = Subset(dev_dataset, fit_indices.tolist())
        pilot_validation_source = FairVisionDataset(
            disease=disease,
            split=dev_split_values[0],
            transform=create_image_transform(image_size=config.image_size, training=False),
        )
        pilot_validation_dataset = Subset(pilot_validation_source, validation_indices.tolist())
    else:
        pilot_fit_splits = dev_split_values[:-1]
        pilot_validation_split = dev_split_values[-1]
        pilot_fit_dataset = FairVisionDataset(
            disease=disease,
            split=pilot_fit_splits,
            transform=create_image_transform(image_size=config.image_size, training=True),
        )
        pilot_validation_dataset = FairVisionDataset(
            disease=disease,
            split=pilot_validation_split,
            transform=create_image_transform(image_size=config.image_size, training=False),
        )

    pilot_fit_fraction = len(pilot_fit_dataset) / len(dev_dataset)
    pilot_validation_fraction = len(pilot_validation_dataset) / len(dev_dataset)
    logger.info(
        f"Pilot split: {len(pilot_fit_dataset):,} fit samples, "
        f"{len(pilot_validation_dataset):,} validation samples "
        f"({pilot_fit_fraction:.1%} fit, {pilot_validation_fraction:.1%} validation)"
    )

    pilot_model = create_model(
        num_classes=len(dev_dataset.classes), pretrained=True, device=device
    )
    pilot_optimizer = torch.optim.AdamW(
        pilot_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pilot_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        pilot_optimizer, T_max=config.epochs, eta_min=1e-6
    )

    (
        pilot_train_loader,
        pilot_train_criterion,
        pilot_class_counts,
        _,
    ) = make_train_loader_and_loss(
        dataset=pilot_fit_dataset,
        config=config,
        device=device,
    )
    pilot_risk_weights = make_class_balanced_weights(pilot_class_counts).to(device)
    pilot_validation_loader = DataLoader(
        pilot_validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    active_eval_loader = DataLoader(
        active_eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    best_epoch = 1
    best_weighted_validation_loss = float("inf")
    best_unweighted_epoch = 1
    best_unweighted_validation_loss = float("inf")
    selected_unweighted_validation_loss = float("inf")
    pilot_validation_history = []
    for epoch in tqdm(
        range(config.epochs),
        desc="Pilot epochs",
        unit="epoch",
        position=0,
        leave=True,
    ):
        logger.info(f"Starting pilot epoch {epoch + 1}/{config.epochs}")
        train_loss = train_one_epoch(
            model=pilot_model,
            loader=pilot_train_loader,
            criterion=pilot_train_criterion,
            optimizer=pilot_optimizer,
            device=device,
            progress_position=1,
        )

        logger.info(
            f"Pilot training loss at epoch {epoch + 1}: {train_loss:.4f} "
            f"(learning rate {pilot_optimizer.param_groups[0]['lr']:.2g})"
        )

        validation_results = evaluate(
            model=pilot_model,
            loader=pilot_validation_loader,
            device=device,
            protected_column="gender",
            class_weights=pilot_risk_weights,
        )
        validation_loss = validation_results["loss"]
        weighted_validation_loss = validation_results["weighted_loss"]
        logger.info(
            f"Pilot validation losses at epoch {epoch + 1}: "
            f"unweighted CE {validation_loss:.4f}, "
            f"class-balanced CE {weighted_validation_loss:.4f}"
        )
        pilot_validation_history.append(
            {
                "epoch": epoch + 1,
                "unweighted_cross_entropy": validation_loss,
                "class_balanced_cross_entropy": weighted_validation_loss,
            }
        )
        if validation_loss < best_unweighted_validation_loss:
            best_unweighted_validation_loss = validation_loss
            best_unweighted_epoch = epoch + 1
        if weighted_validation_loss < best_weighted_validation_loss:
            best_weighted_validation_loss = weighted_validation_loss
            best_epoch = epoch + 1
            selected_unweighted_validation_loss = validation_loss

        pilot_scheduler.step()

    logger.info(
        f"Selected {best_epoch} pilot epochs "
        f"(class-balanced validation CE {best_weighted_validation_loss:.4f}; "
        f"unweighted CE at selected epoch {selected_unweighted_validation_loss:.4f}; "
        f"unweighted CE minimum at epoch {best_unweighted_epoch}: "
        f"{best_unweighted_validation_loss:.4f})"
    )

    # Reinitialize and refit on every development split for the selected duration.
    logger.info(
        f"Refitting on all {len(dev_dataset):,} development samples for {best_epoch} epochs"
    )
    set_seed(config.seed, device=device)
    train_dataset = dev_dataset
    model = create_model(num_classes=len(train_dataset.classes), pretrained=True, device=device)
    train_loader, train_criterion, train_class_counts, _ = make_train_loader_and_loss(
        dataset=train_dataset,
        config=config,
        device=device,
    )
    class_risk_weights = make_class_balanced_weights(train_class_counts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=best_epoch, eta_min=1e-6
    )

    for epoch in tqdm(
        range(best_epoch),
        desc="Refit epochs",
        unit="epoch",
        position=0,
        leave=True,
    ):
        logger.info(f"Starting refit epoch {epoch + 1}/{best_epoch}")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            progress_position=1,
        )
        logger.info(
            f"Refit training loss at epoch {epoch + 1}: {train_loss:.4f} "
            f"(learning rate {optimizer.param_groups[0]['lr']:.2g})"
        )
        scheduler.step()

    model_seeds = [
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(config.seed).spawn(3)
    ]

    surrogate_1 = RandomForestModule(
        num_classes=len(train_dataset.classes),
        random_state=model_seeds[0],
    )

    surrogate_2 = RandomForestModule(
        num_classes=len(train_dataset.classes),
        random_state=model_seeds[1],
    )

    proxy_model = RandomForestModule(
        num_classes=len(train_dataset.classes),
        random_state=model_seeds[2],
    )

    for surrogate in [surrogate_1, surrogate_2, proxy_model]:
        surrogate.fit(train_loader)

    weighted_loss_fn = (
        nn.CrossEntropyLoss(
            weight=class_risk_weights,
            reduction="sum",
        )
        if class_balanced_at
        else None
    )

    surogates = [surrogate_1, surrogate_2]

    method_configs = [
        (UniformSampleMean, {"batch_size": 10}),
        (StratifiedSampleMean, {"allocation": "balanced", "batch_size": 10}),
        (StratifiedSampleMean, {"allocation": "proportional", "batch_size": 10}),
        (
            KossenActiveTesting,
            {
                "surrogate": surrogate_1,
                "num_classes": len(train_dataset.classes),
                "batch_size": 10,
            },
        ),
        (
            ActiveSurrogateEstimator,
            {
                "surrogate": surogates,
                "num_classes": len(train_dataset.classes),
                "acquisition": "xwed",
                "loss_type": "cross_entropy",
                "surrogate_output": "logits",
                "batch_size": 10,
            },
        ),
        (
            PredictionPoweredActiveTesting,
            {
                "num_classes": len(train_dataset.classes),
                "proxy": proxy_model,
                "surrogate": surogates,
                "lambda_value": 0.5,
                "exploration": 0.05,
                "loss_type": "cross_entropy",
                "batch_size": 10,
            },
        ),
    ]

    dfs = []
    histories = {}

    show_repetition_progress = repetitions > 1
    for rep in tqdm(
        range(repetitions),
        desc="Repetitions",
        unit="repetition",
        position=0,
        leave=True,
        disable=not show_repetition_progress,
    ):
        logger.info(f"Starting repetition {rep + 1}/{repetitions}")

        repition_dfs = execute_active_evaluation(
            config,
            device,
            active_eval_dataset,
            model,
            method_configs,
            histories,
            rep,
            method_position=1 if show_repetition_progress else 0,
            budget_position=2 if show_repetition_progress else 1,
            leave_method_progress=not show_repetition_progress,
            weighted_loss_fn=weighted_loss_fn,
        )

        dfs.extend(repition_dfs)

    # Full finite pool reference risk
    results = evaluate(
        model=model,
        loader=active_eval_loader,
        device=device,
        protected_column="gender",
        class_weights=class_risk_weights if class_balanced_at else None,
    )
    logger.info(
        f"Evaluation results for the trained model:\n"
        f"{pp.pformat({k: v for k, v in results.items() if k != 'predictions'})}"
    )

    df = pd.concat(dfs, ignore_index=True)
    df["full_pool_risk"] = results["loss"]
    df["full_pool_class_balanced_risk"] = results["weighted_loss"]
    # Keep the original column name for existing result consumers.
    df["full_pool_weighted_risk"] = results["weighted_loss"]
    df["full_pool_size"] = len(active_eval_dataset)
    df["class_balanced_at"] = class_balanced_at
    df["class_balanced_risk_weights"] = json.dumps(
        {
            class_name: float(weight)
            for class_name, weight in zip(train_dataset.classes, class_risk_weights.cpu())
        }
    )
    df["training_class_prevalences"] = json.dumps(
        {
            class_name: float(count / train_class_counts.sum())
            for class_name, count in zip(train_dataset.classes, train_class_counts)
        }
    )
    for group, risk in results["subgroup_risks"]["gender"].items():
        df[f"full_pool_risk {group}"] = risk
    for group, risk in results["subgroup_weighted_risks"]["gender"].items():
        df[f"full_pool_class_balanced_risk {group}"] = risk
        df[f"full_pool_weighted_risk {group}"] = risk
    df["pilot_fit_size"] = len(pilot_fit_dataset)
    df["pilot_validation_size"] = len(pilot_validation_dataset)
    df["pilot_fit_fraction"] = len(pilot_fit_dataset) / len(dev_dataset)
    df["pilot_validation_fraction"] = len(pilot_validation_dataset) / len(dev_dataset)
    df["pilot_best_epoch"] = best_epoch
    df["pilot_best_epoch_class_balanced"] = best_epoch
    df["pilot_best_epoch_weighted"] = best_epoch
    df["pilot_best_epoch_unweighted"] = best_unweighted_epoch
    df["pilot_validation_loss"] = best_unweighted_validation_loss
    df["pilot_validation_loss_at_weighted_best"] = selected_unweighted_validation_loss
    df["pilot_validation_class_balanced_loss"] = best_weighted_validation_loss
    df["pilot_validation_weighted_loss"] = best_weighted_validation_loss
    df["pilot_validation_history"] = json.dumps(pilot_validation_history)
    df.to_csv(output_path, index=False, mode="x")
    logger.info(f"Saved experiment budget curves to {output_path}")

    return "Experiment run successfully"


def execute_active_evaluation(
    config,
    device,
    test_dataset,
    model,
    method_configs,
    histories,
    repitition,
    method_position=0,
    budget_position=1,
    leave_method_progress=True,
    weighted_loss_fn=None,
):
    repition_dfs = []

    loss_objectives = [("unweighted_cross_entropy", nn.CrossEntropyLoss(reduction="sum"))]
    if weighted_loss_fn is not None:
        loss_objectives.append(("class_balanced_cross_entropy", weighted_loss_fn))
    evaluations = [
        (objective_name, loss_fn, method_class, method_config)
        for objective_name, loss_fn in loss_objectives
        for method_class, method_config in method_configs
    ]

    method_progress = tqdm(
        evaluations,
        desc=f"Methods (repetition {repitition + 1})",
        unit="method",
        position=method_position,
        leave=leave_method_progress,
    )

    for objective_name, loss_fn, method_class, method_config in method_progress:
        method = method_class(**method_config, seed=config.seed + repitition)
        method_name = method.__class__.__name__
        if isinstance(method, StratifiedSampleMean):
            method_name += f"{method.allocation.capitalize()}"
        method_progress.set_postfix_str(f"{method_name}: {objective_name}")

        active_evaluation_history = run_active_evaluation(
            method=method,
            model=model,
            dataset=test_dataset,
            device=device,
            loss_fn=loss_fn,
            protected_column=["gender"],
            budget=1000,
            progress_position=budget_position,
        )

        history_key = (
            method_name
            if objective_name == "unweighted_cross_entropy"
            else f"{method_name} [{objective_name}]"
        )
        histories[history_key] = active_evaluation_history
        df = pd.DataFrame(
            [
                {
                    "n_labels": e.n_labels,
                    "risk": e.risk,
                    "method": method_name,
                    "risk_objective": objective_name,
                    **{f"risk {g}": r for g, r in e.subgroup_risks.items()},
                    **{f"n_labels_{g}": n for g, n in e.subgroup_n_labels.items()},
                }
                for record in active_evaluation_history
                for e in [record["estimate"]]
            ]
        )
        df["repetition"] = repitition + 1
        df["seed"] = config.seed + repitition
        repition_dfs.append(df)
    return repition_dfs


def run_dev_experiment(
    repetitions: int = 1,
    seed: int | None = None,
    class_balanced_at: bool = True,
):
    run_experiment(
        dev_splits="train",
        eval_splits="val",
        repetitions=repetitions,
        seed=seed,
        class_balanced_at=class_balanced_at,
    )


def run_final_experiment(repetitions: int = 1, class_balanced_at: bool = True):
    assert False, (
        "This function is not to be run yet. It should only be ran when ready for the final "
        "experiment using the full development dataset and the test dataset."
    )

    return run_experiment(
        dev_splits=("train", "val"),
        eval_splits="test",
        repetitions=repetitions,
        class_balanced_at=class_balanced_at,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--class-balanced-at",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate class-balanced cross-entropy alongside unweighted cross-entropy",
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")

    run_dev_experiment(
        repetitions=args.repetitions,
        seed=args.seed,
        class_balanced_at=args.class_balanced_at,
    )


if __name__ == "__main__":
    main()
