from typing import Tuple, List
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from src.utils import Buffer
from src.agents.agents import GaussianAgent, SoftmaxAgent


class PPOGaussianNN:
    def __init__(
        self,
        actor,
        critic,
        discretizer_actor=None,
        discretizer_critic=None,
        lr_actor: float=1e-3,
        lr_critic: float=1e-3,
        gamma: float=0.99,
        epochs: int=1000,
        eps_clip: float=0.2,
    ) -> None:

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = GaussianAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = GaussianAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)
        
        mu_params = list(self.policy.actor.parameters())
        std_params = [self.policy.log_sigma]
        
        self.opt_actor = torch.optim.Adam(mu_params + std_params, lr_actor)
        self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(), lr_critic)

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


class PPOSoftmaxNN:
    def __init__(
        self,
        actor,
        critic,
        discretizer_actor=None,
        discretizer_critic=None,
        lr_actor: float=1e-3,
        lr_critic: float=1e-3,
        gamma: float=0.99,
        epochs: int=1000,
        eps_clip: float=0.2,
    ) -> None:

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = SoftmaxAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = SoftmaxAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)

        self.opt_actor = torch.optim.Adam(self.policy.actor.parameters(), lr_actor)
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

        return action.item()

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
