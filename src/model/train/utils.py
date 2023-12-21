import h5py
import numpy as np
from src.utils import mkpath
import os
import src.configs as configs


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
    with h5py.File(configs.embedding_dataset_data_path, "w") as hdf:
        for i, (sequence, target) in enumerate(data):
            group = hdf.create_group(f"tuple_{i}")
            group.create_dataset("sequence", data=sequence)
            group.create_dataset("target", data=target)


def load_dataset_data():
    loaded_data = []

    with h5py.File(configs.embedding_dataset_data_path, "r") as hdf:
        for group_name in hdf:
            group = hdf[group_name]
            sequence = np.array(group["sequence"])
            target = np.array(group["target"])
            loaded_data.append((sequence, target))

    return loaded_data
