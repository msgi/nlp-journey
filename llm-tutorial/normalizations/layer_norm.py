import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """The layer normalization method used in the original paper 'Attention is All You Need'.

    Args:
        dim: the dimmension of the embedding.
        eps: just to avoid the /0 error.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x.shape: (batch, seq_len, hidden_size)
        # mean.shape: (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # print(mean.shape)
        # std.shape: (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)
        # print(std.shape)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm =   x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 
        
        return norm * self.weight + self.bias



if __name__ == "__main__":
    sample = torch.ones((2, 8, 512))
    ln = LayerNorm(512)
    
    print(ln(sample).shape)
    rms_norm = RMSNorm(512)
    print(rms_norm(sample).shape)
