from .shared import project_root
import torch


bert_embedding_path = project_root / "data/historical/train/bert_embeddings.h5"
pca_embedding_path = project_root / "data/historical/train/pca_embeddings.h5"
# use_cuda = torch.cuda.is_available()
n_pca_components = 10
lstm_sequence_length = 60
embedding_dataset_data_path = project_root / "data/historical/train/embedding_data.h5"
