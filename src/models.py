import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, model='gaussian'):
        super(PolicyNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.model == 'gaussian':
            return x, torch.clamp(self.log_sigma, min=-2.0, max=0.0)
        return x


class ValueNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(ValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PolicyLR(torch.nn.Module):
    def __init__(self, n, m, k, scale=1.0, model='gaussian'):
        super().__init__()

        L = scale*torch.randn(n, k, dtype=torch.float32, requires_grad=True)
        R = scale*torch.randn(k, m, dtype=torch.float32, requires_grad=True)

        self.L = torch.nn.Parameter(L)
        self.R = torch.nn.Parameter(R)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, indices):
        rows, cols = indices
        if cols is not None:
            prod = self.L[rows, :] * self.R[:, cols].T
            res = torch.sum(prod, dim=-1)
        else:
            res = torch.matmul(self.L[rows, :], self.R.T)
        if self.model == 'gaussian':
            return res, torch.clamp(self.log_sigma, min=-2.5, max=0.0)
        return res


class ValueLR(torch.nn.Module):
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


class PolicyPARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0, model='gaussian'):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale*(torch.randn(dim, k, dtype=torch.double, requires_grad=True) - 1.0)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, indices):
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            res = torch.matmul(prod, self.factors[-1].T)
        else:
            res = torch.sum(prod, dim=-1)
        if self.model == 'gaussian':
            return res, torch.clamp(self.log_sigma, min=-2.5, max=0.0)
        return res


class ValuePARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale*(torch.randn(dim, k, dtype=torch.double, requires_grad=True) - 1.0)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices):
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            return torch.matmul(prod, self.factors[-1].T)
        return torch.sum(prod, dim=-1)


class PolicyTensor(torch.nn.Module):
    def __init__(self, dims, model='gaussian'):
        super().__init__()

        X = torch.randn(dims, dtype=torch.double, requires_grad=True)
        self.X = torch.nn.Parameter(X)
        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, indices):
        res = self.X[indices]
        if self.model == 'gaussian':
            return res, torch.clamp(self.log_sigma, min=-2.5, max=0.0)
        return res


class ValueTensor(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()

        X = torch.randn(dims, dtype=torch.double, requires_grad=True)
        self.X = torch.nn.Parameter(X)

    def forward(self, indices):
        return self.X[indices]


class PolicyLM(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, model='gaussian'):
        super(PolicyNetwork, self).__init__()
        self.lm = torch.nn.Linear(num_inputs, num_outputs)
        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        x = self.lm(x)
        if self.model == 'gaussian':
            return x, torch.clamp(self.log_sigma, min=-2.0, max=0.0)
        return x


class ValueLM(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ValueNetwork, self).__init__()
        self.lm = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.lm(x)
