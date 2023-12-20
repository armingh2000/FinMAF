import h5py
import numpy as np
from src.utils import mkpath
import os


def dump_embeddings(embeddings, filepath):
    mkpath(filepath)
    with h5py.File(filepath, "w") as file:
        for stock, embedding in embeddings.items():
            file.create_dataset(stock, data=np.array(embedding))


def load_embeddings(filepath):
    if not os.path.exists(filepath):
        raise ValueError(f"Filepath {filepath} does not exist")

    with h5py.File(filepath, "r") as file:
        embeddings = {}
        # Iterate over all datasets in the HDF5 file
        for stock in file:
            # Load the embedding for each stock
            embeddings[stock] = file[stock][:]

        return embeddings


def dump_dataset_data(data):
    with h5py.File("your_data.h5", "w") as hdf:
        for i, (array1, array2) in enumerate(data):
            group = hdf.create_group(f"tuple_{i}")
            group.create_dataset("array1", data=array1)
            group.create_dataset("array2", data=array2)


def load_dataset_data():
    loaded_data = []

    with h5py.File("your_data.h5", "r") as hdf:
        for group_name in hdf:
            group = hdf[group_name]
            array1 = np.array(group["array1"])
            array2 = np.array(group["array2"])
            loaded_data.append((array1, array2))

    return loaded_data
