import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        # Add input to hidden layer connections
        for h in hidden_sizes:
            self.layers.append(nn.Linear(input_size, h))
            self.layers.append(nn.ReLU())
            input_size = h
        # Add hidden to output layer connections
        self.layers.append(nn.Linear(input_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
