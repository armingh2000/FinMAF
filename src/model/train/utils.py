import torch
import os


def dump_torch_object(object, path):
    torch.save(object, path)


def load_torch_object(path):
    if not os.path.exists(path):
        return None

    return torch.load(path)
