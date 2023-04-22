from typing import Tuple, List
from copy import deepcopy

import numpy as np
import scipy
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters

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
        gamma: float=0.99,
        tau: float=0.97,
        epochs: int=1000,
        eps_clip: float=0.2,
    ) -> None:

        self.gamma = gamma
        self.tau = tau
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = GaussianAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = GaussianAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr_actor)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state).double()
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.numpy()

    def calculate_returns(self, values) -> List[float]:
        returns = []
        advantages=[]

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            mask = 1 - self.buffer.terminals[i]

            actual_return = reward + self.gamma*prev_return*mask
            actual_delta = reward + self.gamma*prev_value*mask - values[i]
            actual_advantage = actual_delta + self.gamma*self.tau*prev_advantage*mask        

            returns.insert(0, actual_return)
            advantages.insert(0, actual_advantage)

            prev_return = actual_return
            prev_value = values[i]
            prev_advantage = actual_advantage

        returns = torch.as_tensor(returns).double().detach().squeeze()
        advantages = torch.as_tensor(advantages).double().detach().squeeze()
        advantages = (advantages - advantages.mean())/advantages.std()

        return returns, advantages

    def update(self):
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()

        # Critic - GAE estimation
        values = self.policy.evaluate_value(old_states)
        rewards, advantages = self.calculate_returns(values.data.numpy())

        def loss_critic(params):
            vector_to_parameters(torch.tensor(params), self.policy.critic.parameters())
            self.policy.critic.zero_grad()

            values = self.policy.evaluate_value(old_states)
            loss = (values - rewards).pow(2).mean()
            loss.backward()

            grads = parameters_to_vector([param.grad for param in self.policy.critic.parameters()])
            grad_flat = torch.cat([grad.view(-1) for grad in grads]).data.double().numpy()
            return loss.data.double().numpy(), grad_flat

        # Critic - LBFGS training
        params_critic = torch.cat([param.data.view(-1) for param in self.policy.critic.parameters()])
        params_critic, _, _ = scipy.optimize.fmin_l_bfgs_b(loss_critic, params_critic.double().numpy(), maxiter=25)
        vector_to_parameters(torch.tensor(params_critic), self.policy.critic.parameters())

        # Actor - Stochastic Gradient Ascent
        for _ in range(self.epochs):
            logprobs = self.policy.evaluate_logprob(old_states, old_actions)
            ratio = torch.exp(logprobs - old_logprobs)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            minorizer_raw = ratio * advantages
            minorizer_clamped = ratio_clamped * advantages

            loss_actor = -torch.min(minorizer_raw, minorizer_clamped)

            self.opt_actor.zero_grad()
            loss_actor.mean().backward()
            self.opt_actor.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()


class PPOSoftmaxNN:
    def __init__(
        self,
        actor,
        critic,
        discretizer_actor=None,
        discretizer_critic=None,
        lr_actor: float=1e-3,
        gamma: float=0.99,
        tau: float=0.97,
        epochs: int=1000,
        eps_clip: float=0.2,
    ) -> None:

        self.gamma = gamma
        self.tau = tau
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = SoftmaxAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = SoftmaxAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.opt_actor = torch.optim.Adam(self.policy.actor.parameters(), lr_actor)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state).double()
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def calculate_returns(self, values) -> List[float]:
        returns = []
        advantages=[]

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            mask = 1 - self.buffer.terminals[i]

            actual_return = reward + self.gamma*prev_return*mask
            actual_delta = reward + self.gamma*prev_value*mask - values[i]
            actual_advantage = actual_delta + self.gamma*self.tau*prev_advantage*mask        

            returns.insert(0, actual_return)
            advantages.insert(0, actual_advantage)

            prev_return = actual_return
            prev_value = values[i]
            prev_advantage = actual_advantage

        returns = torch.as_tensor(returns).double().detach().squeeze()
        advantages = torch.as_tensor(advantages).double().detach().squeeze()
        advantages = (advantages - advantages.mean())/advantages.std()

        return returns, advantages

    def update(self):
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()

        # Critic - GAE estimation
        values = self.policy.evaluate_value(old_states)
        rewards, advantages = self.calculate_returns(values.data.numpy())

        def loss_critic(params):
            vector_to_parameters(torch.tensor(params), self.policy.critic.parameters())
            self.policy.critic.zero_grad()

            values = self.policy.evaluate_value(old_states)
            loss = (values - rewards).pow(2).mean()
            loss.backward()

            grads = parameters_to_vector([param.grad for param in self.policy.critic.parameters()])
            grad_flat = torch.cat([grad.view(-1) for grad in grads]).data.double().numpy()
            return loss.data.double().numpy(), grad_flat

        # Critic - LBFGS training
        params_critic = torch.cat([param.data.view(-1) for param in self.policy.critic.parameters()])
        params_critic, _, _ = scipy.optimize.fmin_l_bfgs_b(loss_critic, params_critic.double().numpy(), maxiter=25)
        vector_to_parameters(torch.tensor(params_critic), self.policy.critic.parameters())

        # Actor - Stochastic Gradient Ascent
        for _ in range(self.epochs):
            logprobs = self.policy.evaluate_logprob(old_states, old_actions)
            ratio = torch.exp(logprobs - old_logprobs)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            minorizer_raw = ratio * advantages
            minorizer_clamped = ratio_clamped * advantages

            loss_actor = -torch.min(minorizer_raw, minorizer_clamped)

            self.opt_actor.zero_grad()
            loss_actor.mean().backward()
            self.opt_actor.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
