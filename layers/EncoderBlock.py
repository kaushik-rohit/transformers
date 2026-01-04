import torch.nn as nn

from layers import FeedForwardBlock, LayerNormalization, MultiHeadAttentionBlock, ResidualConnection


class EncoderBlock(nn.Module):

    def __init__(self, attention_block: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, mask))
        x = self.residual_connections[1](y, self.feed_forward)

        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


