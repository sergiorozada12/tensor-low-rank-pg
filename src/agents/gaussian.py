import numpy as np
import torch


class GaussianPolicyNN:
    def __init__(self, env, mu, v, lr_actor=1e-2, lr_critic=1e-2, bool_output=False):
        self.mu = mu
        self.value = v

        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)
        self.opt_actor = torch.optim.Adam(list(self.mu.parameters()) + [self.log_sigma], lr=lr_actor) 
        self.opt_critic = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

    def pi(self, s_t):
        s_t = torch.as_tensor(s_t).double()
        mu = self.mu(s_t).squeeze()

        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        pi = torch.distributions.Normal(mu, sigma)
        return pi

    def v(self, s_t):
        s_t_tensor = torch.as_tensor(s_t).double()
        return self.value(s_t_tensor).squeeze()

    def act(self, s_t):
        a_t = self.pi(s_t).sample()
        return torch.clamp(a_t, 0.0, 1.0) if self.bool_output else a_t

    def learn(self, states, actions, returns):
        actions = torch.as_tensor(actions).double()
        returns = torch.as_tensor(returns).double()
        states = np.array(states)

        values = self.v(states)
        with torch.no_grad():
            advantages = returns - values

        # Actor
        log_prob = self.pi(states).log_prob(actions)
        loss_action = torch.mean(-log_prob*advantages)
        self.opt_actor.zero_grad()
        loss_action.backward()
        self.opt_actor.step()

        # Critic
        loss_fn = torch.nn.MSELoss()
        loss_value = loss_fn(values, returns)
        self.opt_critic.zero_grad()
        loss_value.backward()
        self.opt_critic.step()
