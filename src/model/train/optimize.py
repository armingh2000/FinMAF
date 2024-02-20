import torch
import torch.nn as nn
import src.configs as configs
from tqdm import tqdm
from ray import train
import os
from ray.train import Checkpoint


cyclic_loss_function = getattr(nn, configs.cyclic_loss)()
acyclic_loss_function = getattr(nn, configs.acyclic_loss)()
cyclic_loss_weight = configs.cyclic_loss_weight
acyclic_loss_weight = configs.acyclic_loss_weight


def get_loss(pred, target):
    cyclic_features = pred[:, :5]
    acyclic_features = pred[:, 5:]

    cyclic_target = target[:, :5]
    acyclic_target = target[:, 5:]

    cyclic_loss = cyclic_loss_function(cyclic_features, cyclic_target)
    acyclic_loss = acyclic_loss_function(acyclic_features, acyclic_target)

    return cyclic_loss * cyclic_loss_weight + acyclic_loss * acyclic_loss_weight


def train_func(model, config, train_loader, val_loader, logger):
    optimizer = getattr(torch.optim, config[optimizer])(
        model.parameters(), lr=config["learning_rate"]
    )
    start_epoch = 0

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(start_epoch, config["epochs"]):
        logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}")
        model.train()
        for iteration, data in tqdm(
            enumerate(train_loader, 0), unit="batch", total=len(train_loader)
        ):
            sequences, targets = data
            optimizer.zero_grad()
            y_pred = model(sequences)
            loss = get_loss(y_pred, targets)
            loss.backward()
            optimizer.step()

            if (iteration + 1) % (len(train_loader) // 10) == 0:
                logger.info(f"Iteration {iteration + 1}/{len(train_loader)}")
                val_loss, val_accuracy = evaluate(model, val_loader, logger)
                logger.info(f"Validation Loss: {val_loss:.4f}")

                train.report(
                    {"iteration loss": val_loss, "iteration accuracy": val_accuracy}
                )

        logger.info(f"Epoch {epoch + 1}/{config['epochs']} complete")
        epoch_loss, epoch_accuracy = evaluate(model, val_loader, logger)

        with train.checkpoint_dir(epoch) as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

            # Report metrics back to Ray Tune
            train.report(
                {
                    "epoch": epoch,
                    "epoch loss": epoch_loss,
                    "epoch accuracy": epoch_accuracy,
                },
                checkpoint=Checkpoint.from_directory(checkpoint_dir),
            )

    logger.info("Training complete.")


def get_accuracy(pred, target):
    return (pred == target).float().mean()


def evaluate(model, val_loader, logger):
    logger.info("Evaluating model ...")
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            y_pred = model(sequences)
            val_loss += get_loss(y_pred, targets).item()
            val_accuracy += get_accuracy(y_pred, targets).item()

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    return val_loss, val_accuracy


# def test_model(model, test_loader, logger):
#     model.eval()
#     test_loss = 0.0
#     loss_function = nn.MSELoss()

#     with torch.no_grad():
#         for sequences, targets in test_loader:
#             y_pred = model(sequences)
#             test_loss += loss_function(y_pred, targets).item()

#     test_loss /= len(test_loader)
#     logger.info(f"Test Loss: {test_loss:.4f}")
