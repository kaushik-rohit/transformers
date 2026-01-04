import torch
import torch.nn as nn


class FeedForwardBlock(nn):

    def __init__(self, d_model: int, dropout:int, dff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        # (batch, seq, d_model)
        return self.linear2(self.dropout(self.linear1(x)))