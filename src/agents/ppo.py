from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn

from src.utils import Buffer
from src.models.mlp import Mlp
from src.agents.agents import GaussianAgent


class PPOGaussianNN:
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int],
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        epochs: int,
        eps_clip: float
    ) -> None:

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.buffer = Buffer()

        actor = Mlp(state_dim, hidden_dims, action_dim).double()
        critic = Mlp(state_dim, hidden_dims, action_dim).double()
        actor_old = Mlp(state_dim, hidden_dims, action_dim).double()
        critic_old = Mlp(state_dim, hidden_dims, action_dim).double()

        self.policy = GaussianAgent(actor, critic)
        self.policy_old = GaussianAgent(actor_old, critic_old)
        
        mu_params = list(self.policy.actor.parameters())
        std_params = [self.policy.log_sigma]
        
        self.opt_actor = torch.optim.Adam(mu_params + std_params, lr_actor)
        self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(), lr_critic)

        self.policy_old.load_state_dict(self.policy.state_dict())        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state).double()
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.numpy()

    def calculate_returns(self) -> List[float]:
        # GAE in MC fashion
        returns = []
        return_actual = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.terminals)):
            if done:
                return_actual = 0
            return_actual = reward + self.gamma*return_actual
            returns.insert(0, return_actual)
        return returns

    def update(self):
        rewards = self.calculate_returns()
        rewards = torch.as_tensor(rewards).double().detach().squeeze()

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()
        
        self.buffer.clear()

        for _ in range(self.epochs):
            logprobs = self.policy.evaluate_logprob(old_states, old_actions)
            state_values = self.policy.evaluate_value(old_states)
            
            ratio = torch.exp(logprobs - old_logprobs)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            advantage = rewards - state_values.detach()

            minorizer_raw = ratio * advantage
            minorizer_clamped = ratio_clamped * advantage

            loss_actor = -torch.min(minorizer_raw, minorizer_clamped)
            loss_critic = self.MseLoss(rewards, state_values)

            # Actor
            self.opt_actor.zero_grad()
            loss_actor.mean().backward()
            self.opt_actor.step()

            # Critic
            self.opt_critic.zero_grad()
            loss_critic.mean().backward()
            self.opt_critic.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
