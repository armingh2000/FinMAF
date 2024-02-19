from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import src.configs as configs
from ray.tune import CLIReporter
from model import StockLSTM
from functools import partial
from src.log import setup_logger, revert_streams


def training_function(config, train_loader, val_loader, logger, checkpoint_dir=None):
    # Update configs with Ray Tune's config
    configs.cyclic_loss_weight = config["cyclic_loss_weight"]
    configs.acyclic_loss_weight = 1 - config["cyclic_loss_weight"]
    configs.optimizer = config["optimizer"]
    configs.learning_rate = config["learning_rate"]
    configs.epochs = config["epochs"]
    configs.hidden_size = config["hidden_size"]
    configs.num_layers = config["num_layers"]
    configs.batch_size = config["batch_size"]

    model = StockLSTM()

    # Your existing train function here, adjusted if necessary to fit this format
    # region: fix train function
    train(model, train_loader, val_loader, logger, checkpoint)
    # endregion


def tune_hyperparameters(
    train_loader, val_loader, num_samples=10, max_num_epochs=10, gpus_per_trial=0
):
    # Load and pass the logger here if needed
    logger = setup_logger()

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])

    wrapped_training_function = partial(
        training_function,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )

    result = tune.run(
        wrapped_training_function,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")


if __name__ == "__main__":
    # Load your data here
    train_loader, val_loader = get_data_loaders(
        batch_size=64
    )  # Adjust batch size or loading logic as needed

    # Call the hyperparameter tuning function with the data loaders
    tune_hyperparameters(
        train_loader, val_loader, num_samples=10, max_num_epochs=30, gpus_per_trial=1
    )

    revert_streams()
