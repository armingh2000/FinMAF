import torch

month_days_count = {
    1: 31,  # January
    2: 28,  # February
    3: 31,  # March
    4: 30,  # April
    5: 31,  # May
    6: 30,  # June
    7: 31,  # July
    8: 31,  # August
    9: 30,  # September
    10: 31,  # October
    11: 30,  # November
    12: 31,  # December
}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.manual_seed(2)

attention_dimension = 128
attention_heads = 4
batch_size = 64
epochs = 10
learning_rate = 0.0005
data_dim = (60, 17)
