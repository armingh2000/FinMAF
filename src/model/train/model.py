import torch.nn as nn
import src.configs as configs


class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            configs.input_size,
            configs.hidden_size,
            configs.num_layers,
            batch_first=configs.batch_first,
        )
        self.fc = nn.Linear(configs.hidden_size, configs.output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out
