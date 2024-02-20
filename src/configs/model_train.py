from .shared import project_root
from ray import tune


bert_embedding_path = project_root / "data/historical/train/bert_embeddings.h5"
pca_embedding_path = project_root / "data/historical/train/pca_embeddings.h5"
# use_cuda = torch.cuda.is_available()
n_pca_components = 10
lstm_sequence_length = 60
embedding_dataset_data_path = (
    project_root / "data/historical/train/embedding_dataset_data.h5"
)
stock_durations_path = project_root / "data/historical/train/stock_durations.h5"

# Model
hidden_size = 128
batch_first = True
num_layers = 1
input_size = 11
output_size = 11

# Optimize
batch_size = 16
shuffle = True
epochs = 100
learning_rate = 0.001
cyclic_loss = "MSELoss"
acyclic_loss = "L1Loss"
cyclic_loss_weight = 0.3
acyclic_loss_weight = 0.7
optimizer = "Adam"
model_checkpoint_dir_path = project_root / "data/historical/train/model_checkpoint/"
training_state_checkpoint_dir_path = (
    project_root / "data/historical/train/train_state_checkpoint/"
)

# Dataset
train_dataset_path = project_root / "data/historical/train/train_dataset.pt"
val_dataset_path = project_root / "data/historical/train/val_dataset.pt"
test_dataset_path = project_root / "data/historical/train/test_dataset.pt"
train_split = 0.0001
val_split = 0.00001
torch_seed = 57885161  # prime number

# Ray Tune Configs
ray_tune_config = {
    "batch_size": tune.choice([16, 32, 64]),
    "epochs": tune.uniform(5, 20),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "cyclic_loss_weight": tune.loguniform(1e-4, 1e-1),
    "optimizer": tune.grid(["Adam", "SGD", "RMSprop"]),
    "hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "num_layers": tune.grid([1, 2, 4, 8]),
}
