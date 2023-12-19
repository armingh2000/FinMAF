import torch
from torch.utils.data import random_split

dataset = None

# Assuming 'dataset' is your StockHistoryDataset instance
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


def train_and_evaluate(model, train_loader, val_loader, learning_rate=0.001, epochs=10):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
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
            f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}"
        )

    print("Training complete.")


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


# Define your LSTM model
model = StockLSTM(input_size, hidden_layer_size, output_size, num_layers)

# Train and evaluate the model
train_and_evaluate(model, train_loader, val_loader, learning_rate=0.001, epochs=10)

# Test the model
test_model(model, test_loader)
