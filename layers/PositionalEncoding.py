import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.module):

    def __init__(self, d_model:int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, 1).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() & (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term ) # even positional encoding
        pe[:, 1::2] = torch.cos(pos * div_term) # odd positional encoding

        pe.unsqueeze(0) # adds a new dimension at the start of pe (1, seq_len, d_model) so that we can broadcast over batch

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
