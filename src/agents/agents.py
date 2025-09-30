from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class GaussianAgent(nn.Module):
    def __init__(
        self, actor, critic, discretizer_actor=None, discretizer_critic=None
    ) -> None:
        super(GaussianAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state).double()

        # Parameters
        if self.discretizer_actor:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_actor.get_index(state)
            mu, log_sigma = self.actor(indices)
        else:
            mu, log_sigma = self.actor(state)
        sigma = log_sigma.exp()

        # Distribution
        pi = torch.distributions.Normal(mu.squeeze(), sigma.squeeze())
        return pi

    def evaluate_logprob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action)
        return action_logprob.squeeze()

    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        if self.discretizer_critic:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_critic.get_index(state)
            value = self.critic(indices)
            return value.squeeze()
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()


class SoftmaxAgent(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        n_a,
        discretizer_actor=None,
        discretizer_critic=None,
        beta=1.0,
        max_p=0.99,
    ) -> None:
        super(SoftmaxAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic

        self.beta = beta
        self.max_logit = 1
        self.min_logit = np.log(
            ((np.exp(self.max_logit) / max_p) - np.exp(self.max_logit)) / (n_a - 1)
        )

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state).double()

        # Parameters
        if self.discretizer_actor:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_actor.get_index(state)
            logits = self.actor(indices).squeeze()
        else:
            logits = self.actor(state).squeeze()

        # Distribution
        logits = torch.clamp(self.beta * logits, self.min_logit, self.max_logit)
        pi = torch.distributions.categorical.Categorical(logits=logits)
        return pi

    def evaluate_logprob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action)
        return action_logprob.squeeze()

    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        if self.discretizer_critic:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_critic.get_index(state)
            value = self.critic(indices)
            return value.squeeze()
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()
