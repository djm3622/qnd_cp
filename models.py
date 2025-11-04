# models.py
# LSTM-based high-level decoder (NN2) and optional NN1 filter
import torch
import torch.nn as nn

class HLD_NN2_LSTM(nn.Module):
    """
    Predict which logical operator to add {I, Xbar, Zbar, Ybar} given the time series of syndromes
    before simple decoder corrections are applied.
    Inputs: sequence length T, features F = n_x + n_z
    Output: 4 real values (MSE-targeted probabilities as in paper)
    """
    def __init__(self, input_dim: int, hidden_dims=(16, 4), num_classes: int = 4):
        super().__init__()
        # Paper explores small LSTM sizes; a (16, 4) 2-layer stack was used for d=3 during tuning. :contentReference[oaicite:2]{index=2}
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.relu = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.fc = nn.Linear(hidden_dims[1], num_classes)
        self.sigmoid = nn.Sigmoid()  # for [0,1] outputs used with MSE to approximate probs

    def forward(self, x):
        # x: [B, T, F]
        y, _ = self.lstm1(x)
        y = self.relu(y)
        y, _ = self.lstm2(y)
        # take last time step
        y_last = y[:, -1, :]
        out = self.fc(y_last)
        out = self.sigmoid(out)
        return out

class HLD_NN1_LSTM(nn.Module):
    """
    Optional NN1: tag detection events as data vs measurement induced per check stream.
    We frame as per-timestep per-check Bernoulli. We predict P(is_data_event).
    """
    def __init__(self, input_dim: int, hidden_dims=(16, 4), out_dim: int = None):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.relu = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.fc = nn.Linear(hidden_dims[1], out_dim if out_dim is not None else input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y, _ = self.lstm1(x)
        y = self.relu(y)
        y, _ = self.lstm2(y)
        y_last = y[:, -1, :]
        out = self.fc(y_last)
        out = self.sigmoid(out)
        return out
