import torch
import os


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
