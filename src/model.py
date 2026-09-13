"""The model. Deliberately boring: one LSTM layer, dropout, linear head.

Every arm uses exactly this. Arms differ only in the loss (if a physics arm wins, it can't be because it had a bigger network).
"""

import torch.nn as nn


class LSTMFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """(B, T, F) -> (B, T). One discharge value per day."""
        out, _ = self.lstm(x)
        return self.head(self.drop(out)).squeeze(-1)
