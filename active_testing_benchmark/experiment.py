import pprint

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

from active_testing_benchmark.active_testing import (
    ActiveSurrogateEstimator,
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
    StratifiedSampleMean,
    UniformSampleMean,
    run_active_evaluation,
)
from active_testing_benchmark.active_testing.surrogate import RandomForestModule
from active_testing_benchmark.modeling.data import FairVisionDataset
from active_testing_benchmark.modeling.models import (
    create_image_transform,
    create_model,
)
from active_testing_benchmark.modeling.registry import configure_mlflow
from active_testing_benchmark.modeling.train import (
    TrainConfig,
    evaluate,
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
):
    configure_mlflow()

    config = TrainConfig(
        batch_size=32,
        num_workers=0,
        epochs=2,
        learning_rate=1e-4,
        seed=42,
        image_size=224,
        weighted_sampling=False,
        weighted_loss=True,
        sampling_power=1.0,
        loss_power=1.0,
    )

    device = select_device()
    set_seed(config.seed, device=device)

    # Loading of the datasets
    train_dataset = FairVisionDataset(
        disease=disease,
        split=dev_splits,
        transform=create_image_transform(image_size=config.image_size, training=True),
    )
    logger.info(
        f"Loaded {len(train_dataset):,} samples for development ({dev_splits if isinstance(dev_splits, str) else ', '.join(dev_splits)}) dataset for {disease}"
    )

    test_dataset = FairVisionDataset(
        disease=disease,
        split=eval_splits,
        transform=create_image_transform(image_size=config.image_size, training=False),
    )
    logger.info(
        f"Loaded {len(test_dataset):,} samples for evaluation ({eval_splits if isinstance(eval_splits, str) else ', '.join(eval_splits)}) dataset for {disease}"
    )

    model = create_model(num_classes=len(train_dataset.classes), pretrained=True, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loader, train_criterion, _, _ = make_train_loader_and_loss(
        dataset=train_dataset,
        config=config,
        device=device,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    for epoch in range(config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            debug=5,
        )

        logger.info(f"Training loss at epoch {epoch + 1}: {train_loss:.4f}")

    results = evaluate(model=model, loader=val_loader, device=device, protected_column="gender")

    logger.info(
        f"Evaluation results for the trained model:\n"
        f"{pp.pformat({k: v for k, v in results.items() if k != 'predictions'})}"
    )

    surrogate_1 = RandomForestModule(
        num_classes=len(train_dataset.classes),
        random_state=config.seed,
    )

    surrogate_2 = RandomForestModule(
        num_classes=len(train_dataset.classes),
        random_state=config.seed * 2,
    )

    proxy_model = RandomForestModule(
        num_classes=len(train_dataset.classes),
        random_state=config.seed * 3,
    )

    for i, surrogate in enumerate([surrogate_1, surrogate_2, proxy_model]):
        surrogate.fit(train_loader)

        eval_results = evaluate(
            model=surrogate, loader=val_loader, device=device, protected_column="gender"
        )
        logger.info(
            f"Surrogate {i + 1} evaluation results:\n"
            f"{pp.pformat({k: v for k, v in eval_results.items() if k != 'predictions'})}"
        )

    surogates = [surrogate_1, surrogate_2]

    method_configs = {
        UniformSampleMean: {"batch_size": 10},
        StratifiedSampleMean: {"allocation": "balanced", "batch_size": 10},
        StratifiedSampleMean: {"allocation": "proportional", "batch_size": 10},
        KossenActiveTesting: {
            "surrogate": surrogate_1,
            "num_classes": len(train_dataset.classes),
            "batch_size": 10,
        },
        ActiveSurrogateEstimator: {
            "surrogate": surogates,
            "num_classes": len(train_dataset.classes),
            "acquisition": "xwed",
            "loss_type": "cross_entropy",
            "surrogate_output": "logits",
            "batch_size": 10,
        },
        PredictionPoweredActiveTesting: {
            "num_classes": len(train_dataset.classes),
            "proxy": proxy_model,
            "surrogate": surogates,
            "lambda_value": 0.5,
            "exploration": 0.05,
            "loss_type": "cross_entropy",
            "batch_size": 10,
        },
    }

    dfs = []
    histories = {}
    repititions = 2

    for i, rep in enumerate(range(repititions)):
        logger.info(f"Starting repetition {rep + 1}/{repititions}")

        methods = [
            method_class(**method_config, seed=config.seed + i)
            for method_class, method_config in method_configs.items()
        ]

        for method in methods:
            active_evaluation_history = run_active_evaluation(
                method=method,
                model=model,
                dataset=test_dataset,
                device=device,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                protected_column=["gender"],
                budget=990,
            )

            method_name = method.__class__.__name__
            if isinstance(method, StratifiedSampleMean):
                method_name += f"{method.allocation.capitalize()}"

            histories[method_name] = active_evaluation_history

        for method_name, history in histories.items():
            df = pd.DataFrame(
                [
                    {
                        "n_labels": e.n_labels,
                        "risk": e.risk,
                        "method": method_name,
                        **{f"risk {g}": r for g, r in e.subgroup_risks.items()},
                        **{f"n_labels_{g}": n for g, n in e.subgroup_n_labels.items()},
                    }
                    for record in history
                    for e in [record["estimate"]]
                ]
            )
            df["repetition"] = rep + 1
            df["seed"] = config.seed + i
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("whitegrid")

    _, ax = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

    for i, g in enumerate(["", "male", "female"]):
        sns.lineplot(
            data=df,
            x="n_labels",
            y=f"risk{' ' + g if g else ''}",
            hue="method",
            ax=ax[i],
        )

        ax[i].set_title(f"{g.capitalize() if g else 'Overall'} Risk")
        ax[i].set_xlabel("Number of Labels")
        ax[i].set_ylabel("Estimated Risk")

        # ax[i].set_yscale("log")

    # fig.suptitle(f"Active Evaluation of {disease.upper()} Model")

    plt.tight_layout()
    plt.show()
    print(df)
    return "Experiment run successfully"


def run_dev_experiment():
    run_experiment(dev_splits="train", eval_splits="val")


def run_final_experiment():
    assert False, (
        "This function is not to be run yet. It should only be ran when ready for the final experiment using the full development dataset and the test dataset."
    )

    dev_dataset = FairVisionDataset(disease="dr", split=("train", "val"))
    test_dataset = FairVisionDataset(disease="dr", split="test")

    return run_experiment(dev_dataset=dev_dataset, eval_dataset=test_dataset)


def main():
    run_dev_experiment()


if __name__ == "__main__":
    main()
