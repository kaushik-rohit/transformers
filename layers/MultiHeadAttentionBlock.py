import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model / h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]

        attention_scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout:
            attention_scores = dropout(attention_scores)

        return attention_scores @ v, attention_scores


    def forward(self, q, k, v, mask):
        # q will be (batch, seq, d_model)

        q_ = self.w_q(q)
        k_ = self.w_k(k)
        v_ = self.w_v(v)

        # (batch, h, seq, d_k)
        q_ = q_.view(q_.shape[0], q_.shape[1], self.h, self.d_k).transpose(1, 2)
        k_ = k_.view(k_.shape[0], k_.shape[1], self.h, self.d_k).transpose(1, 2)
        v_ = v_.view(v_.shape[0], v_.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(q, k, v, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
