import torch
import torch.nn as nn
import src.configs as configs
from tqdm import tqdm

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


def train(model, train_loader, val_loader, logger):
    optimizer = getattr(torch.optim, configs.optimizer)(
        model.parameters(), lr=configs.learning_rate
    )

    for epoch in range(configs.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{configs.epochs}")
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

            if (iteration + 1) % 5 == 0:
                evaluate(model, val_loader, logger)

        evaluate(model, val_loader, logger)
        logger.info(f"Epoch {epoch + 1}/{configs.epochs} complete.")

    logger.info("Training complete.")


def evaluate(model, val_loader, logger):
    logger.info("Evaluating model ...")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            y_pred = model(sequences)
            val_loss += get_loss(y_pred, targets).item()

    val_loss /= len(val_loader)
    logger.info(f"Validation Loss: {val_loss:.4f}")


def test_model(model, test_loader, logger):
    model.eval()
    test_loss = 0.0
    loss_function = nn.MSELoss()

    with torch.no_grad():
        for sequences, targets in test_loader:
            y_pred = model(sequences)
            test_loss += loss_function(y_pred, targets).item()

    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
