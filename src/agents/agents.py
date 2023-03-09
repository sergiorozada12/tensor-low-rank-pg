from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class GaussianAgent(nn.Module):
    def __init__(self, actor, critic) -> None:
        super(GaussianAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state).double()

        # Parameters
        mu = self.actor(state).squeeze()
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)

        # Distribution
        pi = torch.distributions.Normal(mu, torch.diag(sigma))
        return pi

    def evaluate_logprob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action)
        return action_logprob.squeeze()
    
    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()


class SoftmaxAgent(nn.Module):
    def __init__(self, actor, critic) -> None:
        super(SoftmaxAgent, self).__init__()

        self.actor = actor
        self.critic = critic

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state).double()

        # Parameters
        logits = self.actor(state).squeeze()

        # Distribution
        pi = torch.distributions.categorical.Categorical(logits=logits)
        return pi

    def evaluate_logprob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action)
        return action_logprob.squeeze()
    
    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()
