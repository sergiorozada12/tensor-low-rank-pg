from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class GaussianAgent(nn.Module):
    def __init__(self, actor, critic, discretizer_actor=None, discretizer_critic=None) -> None:
        super(GaussianAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)

        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state).double()

        # Parameters
        if self.discretizer_actor:
            rows, cols = self.discretizer_actor.get_index(state)
            mu = self.actor(rows, cols).squeeze()
        else:
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
        if self.discretizer_critic:
            rows, cols = self.discretizer_critic.get_index(state)
            value = self.critic(rows, cols)
            return value.squeeze()
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()


class SoftmaxAgent(nn.Module):
    def __init__(self, actor, critic, discretizer_actor=None, discretizer_critic=None) -> None:
        super(SoftmaxAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state).double()

        # Parameters
        if self.discretizer_actor:
            rows, cols = self.discretizer_actor.get_index(state)
            logits = self.actor(rows, cols).squeeze()
        else:
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
        if self.discretizer_critic:
            rows, cols = self.discretizer_critic.get_index(state)
            value = self.critic(rows, cols)
            return value.squeeze()
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()
