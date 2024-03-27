# kudos to @ikostrikov https://github.com/ikostrikov/pytorch-trpo
from typing import Tuple, List
from copy import deepcopy

import numpy as np
import torch
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
        gamma: float=0.99,
        tau=0.97,
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
        self.discretizer_actor = discretizer_actor
        self.tau = tau

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = GaussianAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = GaussianAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)

        self.opt_critic = torch.optim.LBFGS(
            self.policy.critic.parameters(),
            history_size=100,
            max_iter=25,
            line_search_fn='strong_wolfe',
        )

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

    def kl_penalty(self, states):
        if self.discretizer_actor:
            states = states.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_actor.get_index(states)
            mu1, log_sigma1 = self.policy_old.actor(indices)
            mu2, log_sigma2 = self.policy.actor(indices)

            mu1 = mu1.detach().unsqueeze(1)
            mu2 = mu2.unsqueeze(1)
            log_sigma1 = log_sigma1.detach()
        else:
            mu1, log_sigma1 = self.policy_old.actor(states)
            mu2, log_sigma2 = self.policy.actor(states)

            mu1 = mu1.detach()
            log_sigma1 = log_sigma1.detach()

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
        if logprobs.dim() > 1:
            logprobs = logprobs.sum(dim=1)
        ratio = torch.exp(logprobs - old_logprobs)
        return torch.mean(ratio * advantages)

    def line_search(
        self,
        states,
        actions,
        old_logprobs,
        advantages,
        params,
        params_flat,
        gradients,
        expected_improve_rate,
        max_backtracks=10,
        accept_ratio=.1
    ):
        with torch.no_grad():
            loss = self.loss_actor(states, actions, old_logprobs, advantages)

        weights = 0.5**np.arange(max_backtracks)
        for weight in weights:
            params_new = params_flat + weight*gradients
            vector_to_parameters(params_new, params)

            with torch.no_grad():
                loss_new = self.loss_actor(states, actions, old_logprobs, advantages)

            actual_improve = loss_new - loss
            expected_improve = expected_improve_rate*weight
            ratio = actual_improve/expected_improve

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return params_new
        return params_flat

    def fvp(self, vector, states, params):
        vector = vector.clone().requires_grad_()

        self.policy.actor.zero_grad()
        kl_penalty = self.kl_penalty(states)
        grad_kl = torch.autograd.grad(kl_penalty, params, create_graph=True)
        
        grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_vector_dot = grad_kl.dot(vector)
        fisher_vector_product = torch.autograd.grad(grad_vector_dot, params)
        fisher_vector_product = torch.cat([out.view(-1) for out in fisher_vector_product]).detach()

        return fisher_vector_product + self.cg_dampening*vector.detach()

    def conjugate_gradient(self, b, states, params):    
        x = torch.zeros(*b.shape)
        d = b.clone()
        r = b.clone()
        rr = r.dot(r)
        for _ in range(self.cg_iteration):
            Hd = self.fvp(d, states, params)
            alpha = rr / (d.dot(Hd) + 1e-10)
            x = x + alpha * d
            r = r - alpha * Hd
            rr_new = r.dot(r)
            beta = rr_new / (rr + 1e-10)
            d = r + beta * d
            rr = rr_new
            if rr < self.cg_tolerance:
                break
        return x

    def zero_grad(self, model, idx=None):
        if idx is None:
            return

        for i, param in enumerate(model.parameters()):
            if i != idx:
                param.grad.zero_()

    def update_critic(self, idx=None):
        states = torch.stack(self.buffer.states, dim=0).detach()

        # GAE estimation
        values = self.policy.evaluate_value(states)
        rewards, advantages = self.calculate_returns(values.data.numpy())

        # LBFGS training
        def closure():
            self.opt_critic.zero_grad()
            values = self.policy.evaluate_value(states)
            loss = (values - rewards).pow(2).mean()
            if loss.abs() <= 1e10:
                loss.backward()
                self.zero_grad(self.policy.critic, idx)
            return loss
        self.opt_critic.step(closure)

        return advantages

    def update_actor(self, advantages, idx=None):
        params = list(self.policy.actor.parameters())
        params_old = list(self.policy_old.actor.parameters())
        if idx is not None:
            params = [params[idx]]
            params_old = [params_old[idx]]

        states = torch.stack(self.buffer.states, dim=0).detach()
        actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()
        if old_logprobs.dim() > 1:
            old_logprobs = old_logprobs.sum(dim=1)

        # Actor - Gradient estimation
        self.loss_actor(states, actions, old_logprobs, advantages).backward()
        self.zero_grad(self.policy.actor, idx)

        grads = parameters_to_vector([param.grad for param in params])
        params_flat = parameters_to_vector([param for param in params])

        # Actor - Conjugate Gradient Ascent
        direction = self.conjugate_gradient(grads, states, params)
        direction_hessian_norm = direction.dot(self.fvp(direction, states, params))
        lagrange_multiplier = torch.sqrt(2*self.delta/(direction_hessian_norm + 1e-10))

        grads_opt = lagrange_multiplier*direction

        # Actor - Line search backtracking
        expected_improvement = grads.dot(grads_opt)
        params_flat = self.line_search(
            states,
            actions,
            old_logprobs,
            advantages,
            params,
            params_flat,
            grads_opt,
            expected_improvement
        )
        vector_to_parameters(params_flat, params)
        vector_to_parameters(params_flat, params_old)


class TRPOSoftmaxNN:
    def __init__(
        self,
        actor,
        critic,
        n_a,
        discretizer_actor=None,
        discretizer_critic=None,
        gamma: float=0.99,
        tau: float=0.97,
        delta: float=.01,
        cg_dampening: float=0.001,
        cg_tolerance: float=1e-10,
        cg_iteration: float=10,
        beta: float=1.0,
        max_p: float=0.99,
    ) -> None:

        self.gamma = gamma
        self.tau = tau
        self.delta = delta
        self.cg_dampening = cg_dampening
        self.cg_tolerance = cg_tolerance
        self.cg_iteration = cg_iteration

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = SoftmaxAgent(actor, critic, n_a, discretizer_actor, discretizer_critic, beta, max_p)
        self.policy_old = SoftmaxAgent(actor_old, critic_old, n_a, discretizer_actor, discretizer_critic, beta, max_p)

        self.opt_critic = torch.optim.LBFGS(
            self.policy.critic.parameters(),
            history_size=100,
            max_iter=25,
            line_search_fn='strong_wolfe',
        )

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

    def kl_penalty(self, states, actions, old_logprobs):
        logprobs = self.policy.evaluate_logprob(states, actions)
        kl = torch.nn.functional.kl_div(logprobs, old_logprobs, log_target=True)   
        return kl

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
        states,
        actions,
        old_logprobs,
        advantages,
        params,
        params_flat,
        gradients,
        expected_improve_rate,
        max_backtracks=10,
        accept_ratio=.1
    ):
        with torch.no_grad():
            loss = self.loss_actor(states, actions, old_logprobs, advantages)

        weights = 0.5**np.arange(max_backtracks)
        for weight in weights:
            params_new = params_flat + weight*gradients
            vector_to_parameters(params_new, params)

            with torch.no_grad():
                loss_new = self.loss_actor(states, actions, old_logprobs, advantages)

            actual_improve = loss_new - loss
            expected_improve = expected_improve_rate*weight
            ratio = actual_improve/expected_improve

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return params_new
        return params_flat

    def fvp(self, vector, states, params,actions, old_logprobs):
        vector = vector.clone().requires_grad_()

        self.policy.actor.zero_grad()
        kl_penalty = self.kl_penalty(states,actions, old_logprobs)
        grad_kl = torch.autograd.grad(kl_penalty, params, create_graph=True)
        
        grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_vector_dot = grad_kl.dot(vector)
        fisher_vector_product = torch.autograd.grad(grad_vector_dot, params)
        fisher_vector_product = torch.cat([out.view(-1) for out in fisher_vector_product]).detach()

        return fisher_vector_product + self.cg_dampening*vector.detach()

    def conjugate_gradient(self, b, states, params,actions, old_logprobs):    
        x = torch.zeros(*b.shape)
        d = b.clone()
        r = b.clone()
        rr = r.dot(r)
        for _ in range(self.cg_iteration):
            Hd = self.fvp(d, states, params,actions, old_logprobs)
            alpha = rr / (d.dot(Hd) + 1e-10)
            x = x + alpha * d
            r = r - alpha * Hd
            rr_new = r.dot(r)
            beta = rr_new / (rr + 1e-10)
            d = r + beta * d
            rr = rr_new
            if rr < self.cg_tolerance:
                break
        return x

    def zero_grad(self, model, idx=None):
        if idx is None:
            return

        for i, param in enumerate(model.parameters()):
            if i != idx:
                param.grad.zero_()

    def update_critic(self, idx=None):
        states = torch.stack(self.buffer.states, dim=0).detach()

        # GAE estimation
        values = self.policy.evaluate_value(states)
        rewards, advantages = self.calculate_returns(values.data.numpy())

        # LBFGS training
        def closure():
            self.opt_critic.zero_grad()
            values = self.policy.evaluate_value(states)
            loss = (values - rewards).pow(2).mean()
            if loss.abs() <= 1e10:
                loss.backward()
                self.zero_grad(self.policy.critic, idx)
            return loss
        self.opt_critic.step(closure)

        return advantages

    def update_actor(self, advantages, idx=None):
        params = list(self.policy.actor.parameters())
        params_old = list(self.policy_old.actor.parameters())

        if idx is not None:
            params = [params[idx]]
            params_old = [params_old[idx]]

        states = torch.stack(self.buffer.states, dim=0).detach()
        actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()

        # Actor - Gradient estimation
        self.loss_actor(states, actions, old_logprobs, advantages).backward()
        self.zero_grad(self.policy.actor, idx)

        grads = parameters_to_vector([param.grad for param in params])
        params_flat = parameters_to_vector([param for param in params])

        # Actor - Conjugate Gradient Ascent
        direction = self.conjugate_gradient(grads, states, params,actions, old_logprobs)
        direction_hessian_norm = direction.dot(self.fvp(direction, states, params,actions, old_logprobs))
        lagrange_multiplier = torch.sqrt(2*self.delta/(direction_hessian_norm + 1e-10))

        grads_opt = lagrange_multiplier*direction

        # Actor - Line search backtracking
        expected_improvement = grads.dot(grads_opt)
        params_flat = self.line_search(
            states,
            actions,
            old_logprobs,
            advantages,
            params,
            params_flat,
            grads_opt,
            expected_improvement
        )
        vector_to_parameters(params_flat, params)
        vector_to_parameters(params_flat, params_old)
