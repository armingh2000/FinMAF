import torch
import src.configs as configs
import os


def penalized_loss(predictions, targets, threshold=0.01):
    # Compute the absolute difference
    diff = torch.abs(predictions - targets)

    # Penalize large differences more
    loss = torch.where(diff < threshold, 0, diff)
    return loss.sum()


def train(model, train_dataloader, test_dataloader):
    # loss_fn = nn.HuberLoss(reduction="sum")
    loss_fn = penalized_loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs.learning_rate, weight_decay=1e-3
    )

    best_test_loss = float("inf")  # Initialize the best test loss to infinity

    for epoch in range(configs.epochs):
        model.train()  # Ensure the model is in training mode

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(configs.device), y.to(configs.device)

            pred = model(X, True)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{configs.epochs}], "
                    f"Batch [{batch + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Evaluate the model on the test set at the end of each epoch
        print(f"Evaluating after epoch {epoch + 1}...")
        test_loss = test(model, test_dataloader, loss_fn)

        # Save the model if the test loss is the best yet
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_path = os.path.join(configs.model_path)
            torch.save(model.state_dict(), save_path)
            print(
                f"New best model saved with test loss {best_test_loss:.4f} at {save_path}"
            )


def test(model, test_dataloader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for X, y in test_dataloader:
            X, y = X.to(configs.device), y.to(configs.device)

            pred = model(X, False)  # Assuming False for evaluation
            loss = loss_fn(pred, y)

            test_loss += loss.item() * X.size(0)  # Accumulate total loss
            total_samples += X.size(0)

    average_loss = test_loss / total_samples
    print(f"Test Loss: {average_loss:.4f}")
    return average_loss
