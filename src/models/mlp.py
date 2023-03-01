import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Mlp, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        for h in hidden_sizes:
            self.layers.append(nn.Linear(input_size, h))
            self.layers.append(nn.ReLU())
            input_size = h
        self.layers.append(nn.Linear(input_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
