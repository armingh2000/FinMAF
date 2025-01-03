import torch
from torch import nn
import src.configs as configs
import math


class FinMAF(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

        self.is_train = False

        # Positional Encoding and Initial Embedding
        self.pe = PositionalEncoding(data_dim[1])
        self.embedding = nn.Linear(data_dim[1], configs.attention_dimension)

        # Multi-head Attention Layers
        self.multihead_attention1 = nn.MultiheadAttention(
            embed_dim=configs.attention_dimension,
            num_heads=configs.attention_heads,
            dtype=torch.float32,
        )

        # Normalization Layers for Attention Outputs
        self.attention_norm1 = nn.LayerNorm(configs.attention_dimension)

        # Expanded Convolutional Layers
        self.CNN_1 = nn.Conv1d(
            configs.attention_dimension, 64, kernel_size=7, stride=1, padding=3
        )
        self.CNN_2 = nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2)
        self.CNN_3 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)

        # Pooling and Batch Normalization
        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(16)

        # Fully Connected Layers with Increased Complexity
        self.FC_1 = nn.Linear(
            240, 256
        )  # Adjusted input dimension based on expanded CNN
        self.FC_2 = nn.Linear(256, 128)
        self.FC_3 = nn.Linear(128, 5)

        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Dropout Layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, for_train):
        if for_train != self.is_train:
            self.is_train = for_train
            self.train() if for_train else self.eval()
            torch.enable_grad() if for_train else torch.no_grad()

        # Positional Encoding and Embedding
        x_pe = self.pe(x)
        x_em = self.embedding(x_pe)

        # Multi-head Attention
        x_sa = x_em + self.multihead_attention1(x_em, x_em, x_em)[0]
        x_sa = self.attention_norm1(x_sa)

        # Convolutional Layers
        x_cnn1 = self.CNN_1(x_sa.permute(0, 2, 1))
        x_cnn1 = self.relu(self.batch_norm1(x_cnn1))
        x_cnn1 = self.maxpool1(x_cnn1)
        x_cnn1 = self.dropout(x_cnn1)

        x_cnn2 = self.CNN_2(x_cnn1)
        x_cnn2 = self.relu(self.batch_norm2(x_cnn2))
        x_cnn2 = self.maxpool2(x_cnn2)
        x_cnn2 = self.dropout(x_cnn2)

        x_cnn3 = self.CNN_3(x_cnn2)
        x_cnn3 = self.relu(self.batch_norm3(x_cnn3))
        x_cnn3 = self.dropout(x_cnn3)

        # Flatten Outputs
        x_cnn_flat = self.flatten(x_cnn3)

        # Fully Connected Layers
        x_fc1 = self.FC_1(x_cnn_flat)
        x_fc1 = self.relu(x_fc1)
        x_fc1 = self.dropout(x_fc1)
        x_fc2 = self.FC_2(x_fc1)
        x_fc2 = self.relu(x_fc2)
        x_fc2 = self.dropout(x_fc2)
        x_fc3 = self.FC_3(x_fc2)

        return x_fc3


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        position = torch.arange(configs.window_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, configs.window_size, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe)
        self.pe = self.pe.to(torch.float32)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe

        return self.dropout(x)
