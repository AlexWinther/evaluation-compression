import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from active_testing_benchmark.active_testing import (
    UniformSampleMean,
    run_active_evaluation,
)
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


def run_experiment(
    dev_splits: str | tuple[str, ...] = "train",
    eval_splits: str | tuple[str, ...] = "val",
    disease: str = "dr",
):
    configure_mlflow()

    config = TrainConfig(
        batch_size=32,
        num_workers=0,
        epochs=1,
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

    model = create_model(
        num_classes=len(train_dataset.classes), pretrained=True, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loader, train_criterion, counts, loss_weights = make_train_loader_and_loss(
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

    results = evaluate(model=model, loader=val_loader, device=device)

    logger.info(
        f"Evaluation results for the trained model: "
        f"loss={results['loss']:.4f}, "
        f"accuracy={results['accuracy']:.4f}, "
        f"roc_auc={results['roc_auc']:.4f}"
    )

    # Run active evaluation on the test dataset
    active_evaluation_history = run_active_evaluation(
        method=UniformSampleMean(batch_size=config.batch_size),
        model=model,
        dataset=test_dataset,
        device=device,
    )

    logger.debug(
        f"Active evaluation results for the trained model:\n"
        f"{active_evaluation_history[0]}\n\n"
    )

    return "Experiment run successfully"


def run_dev_experiment():
    run_experiment(dev_splits="train", eval_splits="val")


def run_final_experiment():
    assert (
        False
    ), "This function is not to be run yet. It should only be ran when ready for the final experiment using the full development dataset and the test dataset."

    dev_dataset = FairVisionDataset(disease="dr", split=("train", "val"))
    test_dataset = FairVisionDataset(disease="dr", split="test")

    return run_experiment(dev_dataset=dev_dataset, eval_dataset=test_dataset)


def main():
    run_dev_experiment()


if __name__ == "__main__":
    main()
