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
