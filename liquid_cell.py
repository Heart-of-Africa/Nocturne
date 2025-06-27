
import torch
import torch.nn as nn

class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, alpha=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.U = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.alpha = alpha

    def forward(self, h, x):
        dh = -self.alpha * h + torch.tanh(self.W @ h + self.U @ x + self.b)
        return h + dh
