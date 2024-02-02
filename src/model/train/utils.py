import torch
import os
import src.configs as configs


def dump_torch_object(object, path, logger):
    # Check if the object is a PyTorch model (nn.Module)
    if isinstance(object, torch.nn.Module):
        logger.info(f"Saving model to {path}")
        # Save only the model's state dictionary
        torch.save(object.state_dict(), path)
    else:
        # Save the entire object
        logger.info(f"Saving object to {path}")
        torch.save(object, path)


def load_torch_object(path, logger, model_class=None):
    if not os.path.exists(path):
        logger.error("File not found: ", path)
        return None

    if model_class and issubclass(model_class, torch.nn.Module):
        # If a model class is provided and it is a subclass of nn.Module
        logger.info(f"Loading model from {path}")
        model = model_class()  # Initialize the model
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
            return model
        except Exception as e:
            logger.error("Error loading the model state dict: ", e)
            return None
    else:
        # Load the object normally
        logger.info(f"Loading object from {path}")
        try:
            return torch.load(path)
        except Exception as e:
            logger.error("Error loading the object: ", e)
            return None


def save_checkpoint(model, optimizer, epoch, loss, logger):
    logger.info(f"Saving checkpoint: Epoch {epoch} | Loss: {loss:.4f}")
    # Save the model's state dictionary separately
    model_state_dict_path = (
        configs.model_checkpoint_dir_path / f"model_state_dict_{epoch}_{loss:.4f}.pth"
    )
    dump_torch_object(model, model_state_dict_path, logger)

    # Save other training state information
    training_state = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    training_state_path = (
        configs.training_state_checkpoint_dir_path
        / f"training_state_{epoch}_{loss:.4f}.pth"
    )
    dump_torch_object(training_state, training_state_path, logger)

    logger.info(f"Checkpoint saved: Epoch {epoch}")


def load_checkpoint(checkpoint_path=None, model=None, optimizer=None, logger=None):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # Load specific checkpoint file
        logger.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    else:
        # Find the most recent checkpoint in the directory
        model_checkpoint_dir = configs.model_checkpoint_dir_path
        training_state_checkpoint_dir = configs.training_state_checkpoint_dir_path

        model_checkpoints = list(model_checkpoint_dir.glob("model_state_dict_*.pth"))
        training_state_checkpoints = list(
            training_state_checkpoint_dir.glob("training_state_*.pth")
        )

        if not model_checkpoints or not training_state_checkpoints:
            logger.info("No checkpoints found")
            return None, None, None

        # Sort the checkpoints by epoch to find the most recent one
        latest_model_checkpoint = max(
            model_checkpoints, key=lambda path: int(path.stem.split("_")[-2])
        )
        latest_training_state_checkpoint = max(
            training_state_checkpoints, key=lambda path: int(path.stem.split("_")[-2])
        )

        logger.info(f"Loading the latest model checkpoint '{latest_model_checkpoint}'")
        model_state_dict = torch.load(latest_model_checkpoint)
        model.load_state_dict(model_state_dict)

        logger.info(
            f"Loading the latest training state checkpoint '{latest_training_state_checkpoint}'"
        )
        training_state = torch.load(latest_training_state_checkpoint)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        epoch = training_state["epoch"]
        loss = training_state["loss"]

    logger.info(f"Checkpoint loaded successfully: Epoch {epoch}, Loss {loss:.4f}")
    return model, optimizer, epoch
