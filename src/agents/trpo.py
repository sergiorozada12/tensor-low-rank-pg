# kudos to https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/trpoagent.py
from typing import Tuple, List
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters

from src.utils import Buffer
from src.agents.agents import GaussianAgent, SoftmaxAgent


class TRPOGaussianNN:
    def __init__(
        self,
        actor,
        critic,
        discretizer_actor=None,
        discretizer_critic=None,
        lr_critic: float=1e-3,
        gamma: float=0.99,
        delta: float=.01,
        cg_dampening: float=0.001,
        cg_tolerance: float=1e-10,
        cg_iteration: float=10,
    ) -> None:

        self.gamma = gamma
        self.delta = delta
        self.cg_dampening = cg_dampening
        self.cg_tolerance = cg_tolerance
        self.cg_iteration = cg_iteration

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = GaussianAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = GaussianAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)

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

    def kl_penalty(self, states):
        mu1 = self.policy_old.actor(states).detach()
        log_sigma1 = self.policy_old.log_sigma.detach()
        mu2 = self.policy.actor(states)
        log_sigma2 = self.policy.log_sigma

        kl = ((log_sigma2 - log_sigma1) + 0.5 * (log_sigma1.exp().pow(2)
            + (mu1 - mu2).pow(2)) / log_sigma2.exp().pow(2) - 0.5)

        return kl.sum(1).mean()

    def loss_actor(
            self,
            states,
            actions,
            old_logprobs,
            advantages
    ):
        logprobs = self.policy.evaluate_logprob(states, actions)
        ratio = torch.exp(logprobs - old_logprobs)
        return torch.mean(ratio * advantages)

    def line_search(
            self,
            gradients,
            states,
            actions,
            old_logprobs,
            advantages,
    ):
        step_size = (2*self.delta/gradients.double().dot(self.fvp(gradients, states).double())).sqrt()
        step_size_decay = 1.5
        line_search_attempts = 10
        
        policy = deepcopy(self.policy)
        for i in range(line_search_attempts):
            self.policy = deepcopy(policy)

            params = parameters_to_vector(self.policy.actor.parameters())
            new_params = params + step_size*gradients
            vector_to_parameters(
                new_params,
                self.policy.actor.parameters()
            )
            self.policy.log_sigma = self.policy.log_sigma.detach() + step_size*self.policy.log_sigma.grad
            self.policy.log_sigma.requires_grad_()

            with torch.no_grad():
                loss_actor = self.loss_actor(states, actions, old_logprobs, advantages)
                kl_penalty = self.kl_penalty(states)

            # Shrink gradient if KL constraint not met or reward lower
            if kl_penalty > self.delta or loss_actor < 0:
                step_size /= step_size_decay
            else:
                self.policy_old = deepcopy(self.policy)
                return
            

    def fvp(self, vector, states):
        vector = vector.clone().requires_grad_()

        # Gradient of KL w.r.t. network param
        self.policy.actor.zero_grad()
        kl_penalty = self.kl_penalty(states)
        grad_kl = torch.autograd.grad(kl_penalty, self.policy.actor.parameters(), create_graph=True)
        grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        # Gradient of the gradient vector dot product w.r.t. param
        grad_vector_dot = grad_kl.double().dot(vector.double())
        fisher_vector_product = torch.autograd.grad(grad_vector_dot, self.policy.actor.parameters())
        fisher_vector_product = torch.cat([out.view(-1) for out in fisher_vector_product]).detach()

        # Apply CG dampening and return fisher vector product
        return fisher_vector_product + self.cg_dampening*vector.detach()

    def conjugate_gradient(self, b, states):
        p = b.clone()
        r = b.clone().double()
        x = torch.zeros(*p.shape).double()
        rdotr = r.dot(r)
        for _ in range(self.cg_iteration):
            fvp = self.fvp(p, states).double()
            v = rdotr / p.double().dot(fvp)
            x += v * p.double()
            r -= v * fvp
            new_rdotr = r.dot(r)
            mu = new_rdotr / rdotr
            p = (r + mu * p.double()).float()
            rdotr = new_rdotr
            if rdotr < self.cg_tolerance:
                break
        return x.float()
    
    def update(self):
        rewards = self.calculate_returns()
        rewards = torch.as_tensor(rewards).double().detach().squeeze()

        states = torch.stack(self.buffer.states, dim=0).detach()
        actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()
        
        self.buffer.clear()

        # Critic
        state_values = self.policy.evaluate_value(states)
        advantage = rewards - state_values.detach()
        loss_critic = self.MseLoss(rewards, state_values)
        self.opt_critic.zero_grad()
        loss_critic.mean().backward()
        self.opt_critic.step()

        # Actor
        self.loss_actor(states, actions, old_logprobs, advantage).backward()
        gradients = parameters_to_vector([param.grad for param in self.policy.actor.parameters()])

        # Conjugate Gradient Descent
        gradients = self.conjugate_gradient(gradients, states)
        
        # Line search backtracking
        self.line_search(gradients, states, actions, old_logprobs, advantage)


class TRPOSoftmaxNN:
    def __init__(
        self,
        actor,
        critic,
        discretizer_actor=None,
        discretizer_critic=None,
        lr_critic: float=1e-3,
        gamma: float=0.99,
        delta: float=.01,
        cg_dampening: float=0.001,
        cg_tolerance: float=1e-10,
        cg_iteration: float=10,
    ) -> None:

        self.gamma = gamma
        self.delta = delta
        self.cg_dampening = cg_dampening
        self.cg_tolerance = cg_tolerance
        self.cg_iteration = cg_iteration

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = SoftmaxAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = SoftmaxAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)

        self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(), lr_critic)
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

    def kl_penalty(self, states, actions, old_logprobs):
        logprobs = self.policy.evaluate_logprob(states, actions)
        return torch.nn.functional.kl_div(logprobs, old_logprobs, log_target=True)

    def loss_actor(
            self,
            states,
            actions,
            old_logprobs,
            advantages
    ):
        logprobs = self.policy.evaluate_logprob(states, actions)
        ratio = torch.exp(logprobs - old_logprobs)
        return torch.mean(ratio * advantages)

    def line_search(
            self,
            gradients,
            states,
            actions,
            old_logprobs,
            advantages,
    ):
        step_size = (2*self.delta/gradients.double().dot(self.fvp(
            gradients,
            states,
            actions,
            old_logprobs).double())).sqrt()
        step_size_decay = 1.5
        line_search_attempts = 10
        
        policy = deepcopy(self.policy)
        for _ in range(line_search_attempts):
            self.policy = deepcopy(policy)

            params = parameters_to_vector(self.policy.actor.parameters())
            new_params = params + step_size*gradients
            vector_to_parameters(
                new_params,
                self.policy.actor.parameters()
            )

            with torch.no_grad():
                loss_actor = self.loss_actor(states, actions, old_logprobs, advantages)
                kl_penalty = self.kl_penalty(states, actions, old_logprobs)

            # Shrink gradient if KL constraint not met or reward lower
            if kl_penalty > self.delta or loss_actor < 0:
                step_size /= step_size_decay
            else:
                self.policy_old = deepcopy(self.policy)
                return
            

    def fvp(self, vector, states, actions, old_logprobs):
        vector = vector.clone().requires_grad_()

        # Gradient of KL w.r.t. network param
        self.policy.actor.zero_grad()
        kl_penalty = self.kl_penalty(states, actions, old_logprobs)
        grad_kl = torch.autograd.grad(kl_penalty, self.policy.actor.parameters(), create_graph=True)
        grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        # Gradient of the gradient vector dot product w.r.t. param
        grad_vector_dot = grad_kl.double().dot(vector.double())
        fisher_vector_product = torch.autograd.grad(grad_vector_dot, self.policy.actor.parameters())
        fisher_vector_product = torch.cat([out.view(-1) for out in fisher_vector_product]).detach()

        # Apply CG dampening and return fisher vector product
        return fisher_vector_product + self.cg_dampening*vector.detach()

    def conjugate_gradient(self, b, states, actions, old_logprobs):
        p = b.clone()
        r = b.clone().double()
        x = torch.zeros(*p.shape).double()
        rdotr = r.dot(r)
        for _ in range(self.cg_iteration):
            fvp = self.fvp(p, states, actions, old_logprobs).double()
            v = rdotr / p.double().dot(fvp)
            x += v * p.double()
            r -= v * fvp
            new_rdotr = r.dot(r)
            mu = new_rdotr / rdotr
            p = (r + mu * p.double()).float()
            rdotr = new_rdotr
            if rdotr < self.cg_tolerance:
                break
        return x.float()

    def update(self):
        rewards = self.calculate_returns()
        rewards = torch.as_tensor(rewards).double().detach().squeeze()

        states = torch.stack(self.buffer.states, dim=0).detach()
        actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()

        self.buffer.clear()

        # Critic
        state_values = self.policy.evaluate_value(states)
        advantage = rewards - state_values.detach()
        loss_critic = self.MseLoss(rewards, state_values)
        self.opt_critic.zero_grad()
        loss_critic.mean().backward()
        self.opt_critic.step()

        # Actor
        self.loss_actor(states, actions, old_logprobs, advantage).backward()
        gradients = parameters_to_vector([param.grad for param in self.policy.actor.parameters()])

        # Conjugate Gradient Descent
        gradients = self.conjugate_gradient(gradients, states, actions, old_logprobs)
        
        # Line search backtracking
        self.line_search(gradients, states, actions, old_logprobs, advantage)
