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


class PolicyPARAFAC(torch.nn.Module):
    def __init__(self, dims, k, num_outputs, scale=1.0, bias=0.0, model='gaussian'):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) + bias)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

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
    def __init__(self, dims, k, scale=1.0, bias=0.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) + bias)
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


class PolicyRBF(torch.nn.Module):
    def __init__(self, num_inputs, num_rbf_features, num_outputs, model='gaussian'):
        super(PolicyRBF, self).__init__()
        self.num_rbf_features = num_rbf_features
        self.centers = torch.randn(num_rbf_features, num_inputs).double()
        self.linear = torch.nn.Linear(num_rbf_features, num_outputs)
        
        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

    def radial_basis(self, x):
        if x.ndim == 1:
            x = torch.unsqueeze(x, dim=0)
        dist = torch.cdist(x, self.centers)
        return torch.exp(-dist.pow(2))

    def forward(self, x):
        rbf_feats = self.radial_basis(x)
        x = self.linear(rbf_feats).squeeze()
        if self.model == 'gaussian':
            return x, torch.clamp(self.log_sigma, min=-2.0, max=0.0)
        return x


class ValueRBF(torch.nn.Module):
    def __init__(self, num_inputs, num_rbf_features, num_outputs):
        super(ValueRBF, self).__init__()
        self.num_rbf_features = num_rbf_features
        self.centers = torch.randn(num_rbf_features, num_inputs).double()
        self.linear = torch.nn.Linear(num_rbf_features, num_outputs)

    def radial_basis(self, x):
        dist = torch.cdist(x, self.centers)
        return torch.exp(-dist.pow(2))

    def forward(self, x):
        rbf_feats = self.radial_basis(x)
        return self.linear(rbf_feats)
