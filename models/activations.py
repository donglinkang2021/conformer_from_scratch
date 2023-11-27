# PyTorch
import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        """Swish activation function: x * sigmoid(x)"""
        super(Swish, self).__init__()
    
    def forward(self, x):
        """
        Parameters
        ----------
        @param x : torch.Tensor
            Input tensor
        """
        return x * x.sigmoid()

class Glu(nn.Module):
    
    def __init__(self, dim):
        """Gated Linear Unit activation function: x_in * sigmoid(x_gate)"""
        super(Glu, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Parameters
        ----------
        @param x : torch.Tensor
            Input tensor
        """
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()