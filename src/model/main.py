from src.model.dataset import load_processed_symbols, FinMAFDataset
from src.model.model import *
from torch.utils.data import DataLoader, random_split
import src.configs as configs
from src.model.train import train
from src.model.utils import make_paths

if __name__ == "__main__":
    make_paths()

    dfs = load_processed_symbols()

    dataset = FinMAFDataset(dfs)
    # torch.save(dataset, configs.dataset_path)
    # dataset = torch.load(configs.dataset_path)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size)

    data_dim = train_dataset[0][0].shape
    model = FinMAF(data_dim)

    train(model, train_dataloader, test_dataloader)
