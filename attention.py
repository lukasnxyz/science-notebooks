'''
Attention Is All You Need
paper: https://arxiv.org/pdf/1706.03762
(block only)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self,
                 block_size: int,
                 head_size: int,
                 n_embd: int,
                 dropout: float = 0.2,
    ):
        '''
        parameters:
        block_size (int): max sequence length (context length 128, 256, 512, 1024, etc.)
        head_size (int): dim of attention head output (n_embd // n_heads, sqrt(d_k))
        n_embd (int): number of vector embedding dimensions
        dropout (float): 0-1 random dropout change
        '''

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # https://pytorch.org/docs/stable/generated/torch.tril.html
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        _, T, C = X.shape # batch, t, channels

        # store the actual values in k and q
        k = self.key(X)
        q = self.query(X)
        v = self.value(X)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # matrix magic from karpathy
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation of vals
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 block_size: int,
                 num_heads: int,
                 head_size: int,
                 n_embd: int,
                 dropout: float=0.2,
    ):
        '''
        parameters:
        block_size (int): max sequence length (context length 128, 256, 512, 1024, etc.)
        num_heads (int): number of heads of attention (8 in the original paper)
        head_size (int): dim of attention head output (n_embd // n_heads, sqrt(d_k))
        n_embd (int): number of vector embedding dimensions
        dropout (float): 0-1 random dropout change
        '''

        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(block_size, head_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float=0.2):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.net(X)

class Block(nn.Module):
    def __init__(self, block_size: int, n_head: int, n_embd: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(block_size, head_size, n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, X):
        X = X + self.sa(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))
        return X
