import os
import h5py
import numpy as np
import torch


def mkpath(path):
    """
    Checks if the given path is a file or a directory and creates it if it doesn't exist.

    :param path: str - The file or directory path to check and create.
    """
    if not os.path.exists(path):
        if "." in os.path.basename(
            path
        ):  # Assuming it's a file if there's an extension
            # Create the parent directories if they don't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Create an empty file
            with open(path, "w") as f:
                pass
        else:
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
        print(f"Created path: {path}")
    else:
        print(f"Path already exists: {path}")


def dump_dictionary(dictionary, filepath):
    mkpath(filepath)
    with h5py.File(filepath, "w") as file:
        for key, value in dictionary.items():
            # Check if value is a numpy array
            if isinstance(value, np.ndarray):
                ds = file.create_dataset(key, data=value)
                ds.attrs["type"] = "numpy_array" if value.shape else "numpy_scalar"
            # Check if value is a torch.Tensor
            elif isinstance(value, torch.Tensor):
                tensor_np = value.numpy()  # Convert tensor to numpy array
                ds = file.create_dataset(key, data=tensor_np)
                ds.attrs["type"] = "torch_tensor"
            # Check if value is an integer
            elif isinstance(value, int):
                ds = file.create_dataset(key, (), dtype="i")  # Create a scalar dataset
                ds[()] = value  # Assign the integer value
                ds.attrs["type"] = "int"
            else:
                raise TypeError(f"Unsupported type for key '{key}': {type(value)}")


def load_dictionary(filepath, logger=None, data=None):
    if not os.path.exists(filepath):
        raise ValueError(f"Filepath {filepath} does not exist")

    logger_number = 0
    data_number = 0

    if logger is not None:
        logger_number = 1

    if data is not None:
        data_number = 1

    assert (
        not logger_number ^ data_number
    ), "Must specify both 'logger' and 'data' or neither"

    if data_number and logger_number:
        logger.info(f"Loading {data} ...")

    with h5py.File(filepath, "r") as file:
        dictionary = {}
        for key in file:
            item = file[key]
            item_type = item.attrs["type"]
            if item_type == "numpy_array":
                dictionary[key] = item[:]
            elif item_type == "numpy_scalar":
                dictionary[key] = np.array(item[()])
            elif item_type == "torch_tensor":
                dictionary[key] = torch.from_numpy(
                    item[:]
                )  # Convert numpy array back to torch.Tensor
            elif item_type == "int":
                dictionary[key] = item[()]
            else:
                raise ValueError(f"Unknown type for key '{key}': {item_type}")

        return dictionary
