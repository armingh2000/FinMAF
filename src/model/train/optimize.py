import torch
import torch.nn as nn
import src.configs as configs
from src.model.train.dataset import prepare_dataset_chunks


def train(model, train_loader, val_loader, test_loader, logger):
    cyclic_loss_function = getattr(nn, configs.cyclic_loss)()
    acyclic_loss_function = getattr(nn, configs.acyclic_loss)()
    optimizer = getattr(torch.optim, configs.optimizer)(
        model.parameters(), lr=configs.learning_rate
    )

    for epoch in range(configs.epochs):
        model.train()
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            y_pred = model(sequences)
            loss = loss_function(y_pred, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                y_pred = model(sequences)
                val_loss += loss_function(y_pred, targets).item()

        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch + 1}/{configs.epochs} - Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}"
        )

    print("Training complete.")


def evaluate(model, test_loader):
    pass


def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    loss_function = nn.MSELoss()

    with torch.no_grad():
        for sequences, targets in test_loader:
            y_pred = model(sequences)
            test_loss += loss_function(y_pred, targets).item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
