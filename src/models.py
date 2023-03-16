import torch


class Mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Mlp, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.layers = torch.nn.ModuleList()
        for h in hidden_sizes:
            self.layers.append(torch.nn.Linear(input_size, h))
            self.layers.append(torch.nn.ReLU())
            input_size = h
        self.layers.append(torch
        .nn.Linear(input_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LR(torch.nn.Module):
    def __init__(self, n, m, k, scale=1.0):
        super().__init__()

        L = scale*torch.randn(n, k, dtype=torch.float32, requires_grad=True)
        R = scale*torch.randn(k, m, dtype=torch.float32, requires_grad=True)

        self.L = torch.nn.Parameter(L)
        self.R = torch.nn.Parameter(R)
        
    def forward(self, indices):
        rows, cols = indices
        if cols is not None:
            prod = self.L[rows, :] * self.R[:, cols].T
            return torch.sum(prod, dim=-1)
        return torch.matmul(self.L[rows, :], self.R.T)


class PARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0):
        super().__init__()
        
        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale*torch.randn(dim, k, dtype=torch.double, requires_grad=True)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices):
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(self.n_factors):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        return torch.sum(prod, dim=-1)
