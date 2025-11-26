import torch
import torch.nn as nn
import numpy as np
import math


# relu(x) = max(0, x)
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.tensor(0.0))


# sigmoid(x) = 1 / (1 + e^(-x))
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


# tanh(x) =
def tanh(x: torch.Tensor) -> torch.Tensor:
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


# softmax(x) = \frac{e^{x_i}}{\sum_i^n e^{x_i}}
def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])  # 防止数据爆炸
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


# leaky_relu(x) = if(x < 0, alpha * x, x)
def leaky_relu(x: torch.Tensor, alpha=0.01) -> torch.Tensor:
    return torch.where(x > 0, x, alpha * x)


# elu(x) = if(x < 0, alpha * (e^x - 1), x)
def elu(x: torch.Tensor, alpha=1.0) -> torch.Tensor:
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))


# swish(x) = x / (1 + e^(-x))
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * sigmoid(x)


# \text{GELU}(x) \approx 0.5 \times x \times \left( 1 + \tanh\left[ \sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^3) \right] \right)
def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh( torch.sqrt(torch.tensor(2.0) / torch.pi)  * (x + 0.044715 * torch.pow(x, 3.0))))
    


if __name__ == "__main__":
    data = torch.tensor([[2.0, 3.0, 4.0, -1, -3], [10.0, 20, 30, -20, -30]])
    mat = torch.tensor([1, 2, 3, 4])

    print(gelu(data))
    print(softmax(mat))
