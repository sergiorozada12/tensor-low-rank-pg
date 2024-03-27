from typing import List, Tuple
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import random

import multiprocessing

from src.train import Trainer
from src.utils import Buffer, Discretizer


torch.set_num_threads(1)



class WirelessCommunicationsEnv:
    """
    SETUP DESCRIPTION
    - Wireless communication setup, focus in one user sharing the media with other users
    - Finite horizon transmission (T slots)
    - User is equipped with a battery and queue where info is stored
    - K orthogonal channels, user can select power for each time instant
    - Rate given by Shannon's capacity formula

    STATES
    - Amount of energy in the battery: bt (real and positive)
    - Amount of packets in the queue: queuet (real and positive)
    - Normalized SNR (aka channel gain) for each of the channels: gkt (real and positive)
    - Channel being currently occupied: ok (binary)

    ACTIONS
    - Accessing or not each of the K channels pkt
    - Tx power for each of the K channels
    - We can merge both (by setting pkt=0)
    """

    def __init__(
        self,
        T: int = 10,  # Number of time slots
        K: int = 3,  # Number of channels
        snr_max: float = 10,  # Max SNR
        snr_min: float = 2,  # Min SNR
        snr_autocorr: float = 0.7,  # Autocorrelation coefficient of SNR
        P_occ: np.ndarray = np.array(
            [  # Prob. of transition of occupancy
                [0.3, 0.5],
                [0.7, 0.5],
            ]
        ),
        occ_initial: List[int] = [1, 1, 1],  # Initial occupancy state
        batt_harvest: float = 3,  # Battery to harvest following a Bernoulli
        P_harvest: float = 0.5,  # Probability of harvest energy
        batt_initial: float = 5,  # Initial battery
        batt_max_capacity: float = 50,  # Maximum capacity of the battery
        batt_weight: float = 1.0,  # Weight for the reward function
        queue_initial: float = 20,  # Initial size of the queue
        queue_arrival: float = 20, # Arrival messages
        queue_max_capacity: float = 20,
        t_queue_arrival: int = 10, # Refilling of the queue
        queue_weight: float = 1e-1,  # Weight for the reward function
        loss_busy: float = 0.80,  # Loss in the channel when busy
    ) -> None:
        self.T = T
        self.K = K

        self.snr = np.linspace(snr_max, snr_min, K)
        self.snr_autocorr = snr_autocorr

        self.occ_initial = occ_initial
        self.P_occ = P_occ

        self.batt_harvest = batt_harvest
        self.batt_initial = batt_initial
        self.P_harvest = P_harvest
        self.batt_max_capacity = batt_max_capacity
        self.batt_weight = batt_weight

        self.queue_initial = queue_initial
        self.queue_weight = queue_weight
        self.t_queue_arrival = t_queue_arrival
        self.queue_arrival = queue_arrival
        self.queue_max_capacity = queue_max_capacity

        self.loss_busy = loss_busy

    def step(self, p: np.ndarray):
        p = np.clip(p, 0, 2)
        if np.sum(p) > self.batt[self.t]:
            p = self.batt[self.t] * p / np.sum(p)

        self.c[:, self.t] = np.log2(1 + self.g[:, self.t] * p)
        self.c[:, self.t] *= (1 - self.loss_busy) * self.occ[:, self.t] + (
            1 - self.occ[:, self.t]
        )

        self.t += 1

        self.h[:, self.t] = np.sqrt(0.5 * self.snr) * (
            np.random.randn(self.K) + 1j * np.random.randn(self.K)
        )
        self.h[:, self.t] *= np.sqrt(1 - self.snr_autocorr)
        self.h[:, self.t] += np.sqrt(self.snr_autocorr) * self.h[:, self.t - 1]
        self.g[:, self.t] = np.abs(self.h[:, self.t]) ** 2
        
        # self.P_occ[1, 1] -> prob getting unocc
        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[1, 1]) * self.occ[
            :, self.t - 1
        ]

        # self.P_occ[0, 0] -> prob keeping unocc
        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[0, 0]) * (
            1 - self.occ[:, self.t - 1]
        )

        energy_harv = self.batt_harvest * (self.P_harvest > np.random.rand())
        self.batt[self.t] = self.batt[self.t - 1] - np.sum(p) + energy_harv
        self.batt[self.t] = np.clip(self.batt[self.t], 0, self.batt_max_capacity)

        packets = 0
        if self.batt[self.t - 1] > 0:
            packets = np.sum(self.c[:, self.t - 1])
        self.queue[self.t] = self.queue[self.t - 1] - packets
        
        if self.t % self.t_queue_arrival == 0:
            noise = np.random.randint(5) - 2
            self.queue[self.t] += self.queue_arrival + noise
            
        self.queue[self.t] = np.clip(self.queue[self.t], 0, self.queue_max_capacity)

        r = (self.batt_weight * np.log(1 + self.batt[self.t]) - self.queue_weight * self.queue[self.t])
        done = self.t == self.T

        return self._get_obs(self.t), r, done, None, None

    def reset(self):
        self.t = 0
        self.h = np.zeros((self.K, self.T + 1), dtype=np.complex64)
        self.g = np.zeros((self.K, self.T + 1))
        self.c = np.zeros((self.K, self.T + 1))
        self.occ = np.zeros((self.K, self.T + 1))
        self.queue = np.zeros(self.T + 1)
        self.batt = np.zeros(self.T + 1)

        self.h[:, 0] = np.sqrt(0.5 * self.snr) * (
            np.random.randn(self.K) + 1j * np.random.randn(self.K)
        )
        self.g[:, 0] = np.abs(self.h[:, 0]) ** 2
        self.occ[:, 0] = self.occ_initial
        self.queue[0] = self.queue_initial
        self.batt[0] = self.batt_initial

        return self._get_obs(0), None

    def _get_obs(self, t):
        return np.concatenate(
            [self.g[:, t], self.occ[:, t], [self.queue[t], self.batt[t]]]
    )

class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, model='gaussian'):
        super(PolicyNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.model == 'gaussian':
            return x, torch.clamp(self.log_sigma, min=-2.0, max=0.0)
        return x

class ValueNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(ValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PolicyPARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0, bias=0.0, model='gaussian'):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) + bias) # CHANGE INIT
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, dims[-1])) # CHANGE DIMS OF SIGMA

    def forward(self, indices):
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            res = torch.matmul(prod, self.factors[-1].T)
        else:
            res = torch.sum(prod, dim=-1)
        if self.model == 'gaussian':
            #print('mu: ', res)
            #print('logsigma: ', self.log_sigma)
            return res, torch.clamp(self.log_sigma, min=-2.5, max=0.0)
        return res

class ValuePARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0, bias=0.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) + bias) # CHANGE INIT
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices):
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            return torch.matmul(prod, self.factors[-1].T)
        #print(torch.sum(prod, dim=-1))
        return torch.sum(prod, dim=-1)

class GaussianAgent(nn.Module):
    def __init__(self, actor, critic, discretizer_actor=None, discretizer_critic=None) -> None:
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

    def evaluate_logprob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
            loss.backward()
            self.zero_grad(self.policy.critic, idx)
            return loss
        self.opt_critic.step(closure)

        return advantages

    def update_actor(self, advantages, idx=None):
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze().sum(dim=1) # CHANGE HERE

        # Stochastic Gradient Ascent
        for _ in range(self.epochs):
            logprobs = self.policy.evaluate_logprob(old_states, old_actions).sum(dim=1) # CHANGE HERE
            ratio = torch.exp(logprobs - old_logprobs)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            minorizer_raw = ratio * advantages
            minorizer_clamped = ratio_clamped * advantages

            loss_actor = -torch.min(minorizer_raw, minorizer_clamped)

            self.opt_actor.zero_grad()
            loss_actor.mean().backward()
            self.zero_grad(self.policy.actor, idx)
            self.opt_actor.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Discretizer:
    def __init__(
        self,
        min_points,
        max_points,
        buckets,
        dimensions=None,
        ):

        self.min_points = np.array(min_points)
        self.max_points = np.array(max_points)
        self.buckets = np.array(buckets)

        self.range = self.max_points - self.min_points
        self.spacing = self.range / self.buckets

        self.dimensions = dimensions
        if dimensions:
            self.n_states = np.round(self.buckets).astype(int)
            self.row_n_states = [self.n_states[dim] for dim in self.dimensions[0]]
            self.col_n_states = [self.n_states[dim] for dim in self.dimensions[1]]

            self.N = np.prod(self.row_n_states)
            self.M = np.prod(self.col_n_states)

            self.row_offset = [int(np.prod(self.row_n_states[i + 1:])) for i in range(len(self.row_n_states))]
            self.col_offset = [int(np.prod(self.col_n_states[i + 1:])) for i in range(len(self.col_n_states))]

    def get_index(self, state):
        state = np.clip(state, a_min=self.min_points, a_max=self.max_points)
        scaling = (state - self.min_points) / self.range
        idx = np.round(scaling * (self.buckets - 1)).astype(int)

        if not self.dimensions:
            return idx

        row_idx = idx[:, self.dimensions[0]]
        row = np.sum(row_idx*self.row_offset, axis=1)

        col = None
        if self.dimensions[1]:
            col_idx = idx[:, self.dimensions[1]]
            col = np.sum(col_idx*self.col_offset, axis=1)

        return [row, col]

def run_dqn(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    env = WirelessCommunicationsEnv(
        T=1_000,
        K=2,
        snr_max=10,
        snr_min=2,
        snr_autocorr=0.7,
        P_occ=np.array(
            [  
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        ),
        occ_initial=[1, 1],
        batt_harvest=1.0, 
        P_harvest=0.2, 
        batt_initial=5,
        batt_max_capacity=10,  
        batt_weight=1.0, 
        queue_initial=10,
        queue_arrival=5,
        queue_max_capacity=20,
        t_queue_arrival=10,
        queue_weight=0.2,
        loss_busy=0.8,  
    )
    
    actor = PolicyNetwork(6, [64], 2, model='gaussian').double()
    critic = ValueNetwork(6, [64], 1).double()

    agent = PPOGaussianNN(
        actor=actor,
        critic=critic,
        gamma=0.9,
        tau=0.99,
        lr_actor=1e-3,
        epochs=1,
        eps_clip=0.2,
    )

    trainer = Trainer(
        actor_opt='sgd',
        critic_opt='sgd',
    )

    _, G, _ = trainer.train(
        env=env,
        agent=agent,
        epochs=5_000,
        max_steps=1_000,
        update_freq=5_000,
        initial_offset=0,
    )

    return G

def run_tlr(n):
    
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    env = WirelessCommunicationsEnv(
        T=1_000,
        K=2,
        snr_max=10,
        snr_min=2,
        snr_autocorr=0.7,
        P_occ=np.array(
            [  
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        ),
        occ_initial=[1, 1],
        batt_harvest=1.0, 
        P_harvest=0.2, 
        batt_initial=5,
        batt_max_capacity=10,  
        batt_weight=1.0, 
        queue_initial=10,
        queue_arrival=5,
        queue_max_capacity=20,
        t_queue_arrival=10,
        queue_weight=0.2,
        loss_busy=0.8,  
    )

    discretizer = Discretizer(
        min_points=[0, 0, 0, 0, 0, 0],
        max_points=[20, 20, 1, 1, 20, 10],
        buckets=[10, 10, 2, 2, 10, 10],
    )

    actor = PolicyPARAFAC([10, 10, 2, 2, 10, 10, 2], k=10, model='gaussian', scale=0.5, bias=1.0).double()
    critic = ValuePARAFAC([10, 10, 2, 2, 10, 10], k=10, scale=1.0).double()

    agent = PPOGaussianNN(
        actor=actor,
        critic=critic,
        discretizer_actor=discretizer,
        discretizer_critic=discretizer,
        gamma=0.9,
        tau=0.99,
        lr_actor=1e-2, # 1e-3
        epochs=1,
        eps_clip=0.2,
    )

    trainer = Trainer(
        actor_opt='sgd',
        critic_opt='sgd',
    )

    try:
        _, G, _ = trainer.train(
            env=env,
            agent=agent,
            epochs=5_000,
            max_steps=1_000,
            update_freq=5_000,
            initial_offset=0,
        )
        return G
    except:
        return [0.0] * 5_000

EXP = 100
PROC = 50

if __name__ == "__main__":
    with multiprocessing.Pool(processes=PROC) as pool:
        results = pool.map(
            run_tlr,
            range(EXP)
        )
    results = np.array(results)
    np.save('wireless_tlr.npy', results)
    
    """
    with multiprocessing.Pool(processes=PROC) as pool:
        results = pool.map(
            run_dqn,
            range(EXP)
        )
    results = np.array(results)
    np.save('wireless_dqn.npy', results)
    """