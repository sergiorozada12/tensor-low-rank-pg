import numpy as np
import torch

from src.models.mlp import Mlp
from src.agents.agents import GaussianAgent
from src.utils import Buffer


class ReinforceGaussianNN:
    def __init__(self, state_dim, hidden_dims, action_dim, gamma=0.99, lr_actor=1e-2, lr_critic=1e-2):
        self.gamma = gamma

        actor = Mlp(state_dim, hidden_dims, action_dim).double()
        critic = Mlp(state_dim, hidden_dims, action_dim).double()
        self.policy = GaussianAgent(actor, critic)

        mu_params = list(self.policy.actor.parameters())
        std_params = [self.policy.log_sigma]
        
        self.opt_actor = torch.optim.Adam(mu_params + std_params, lr_actor)
        self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(), lr_critic)

        self.MseLoss = torch.nn.MSELoss()
        self.buffer = Buffer()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state).double()
            action, action_logprob = self.policy.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.numpy()

    def calculate_returns(self):
        result = np.empty_like(self.buffer.rewards)
        result[-1] = self.buffer.rewards[-1]
        for t in range(len(self.buffer.rewards)-2, -1, -1):
            result[t] = self.buffer.rewards[t] + self.gamma*result[t+1]
        return result

    def update(self):
        returns = self.calculate_returns()
        returns = torch.as_tensor(returns).double().detach().squeeze()

        states = torch.stack(self.buffer.states, dim=0).detach()
        actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        logprobs = self.policy.evaluate_logprob(states, actions)

        state_values = self.policy.evaluate_value(states)
        advantages = returns - state_values.detach()

        loss_actor = -logprobs*advantages
        loss_critic = self.MseLoss(state_values, returns)

        # Actor
        self.opt_actor.zero_grad()
        loss_actor.mean().backward()
        self.opt_actor.step()

        # Critic
        self.opt_critic.zero_grad()
        loss_critic.mean().backward()
        self.opt_critic.step()

        self.buffer.clear()
