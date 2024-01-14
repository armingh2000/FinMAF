from .shared import project_root
import torch


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

# Train
batch_size = 16
shuffle = True
epochs = 10
learning_rate = 0.001
cyclic_loss = "MSELoss"
acyclic_loss = "L1Loss"
cyclic_loss_weight = 0.7
acyclic_loss_weight = 0.3
optimizer = "Adam"

# Dataset
train_dataset_path = project_root / "data/historical/train/train_dataset.pt"
val_dataset_path = project_root / "data/historical/train/val_dataset.pt"
test_dataset_path = project_root / "data/historical/train/test_dataset.pt"
train_split = 0.7
val_split = 0.15
generator_seed = 57885161  # prime number
