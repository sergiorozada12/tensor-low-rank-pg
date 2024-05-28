
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
import gymnasium as gym
import torch

from src.environments.pendulum import PendulumEnvDiscrete
from src.models import PolicyNetwork, ValueNetwork
from src.agents.ppo import PPOGaussianNN, PPOSoftmaxNN
from src.train import Trainer


def train_continuous():
    env = gym.make('Pendulum-v1', g=9.81)

    actor = PolicyNetwork(3, [64], 1, model='gaussian').double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = PPOGaussianNN(
        actor,
        critic,
        lr_actor=1e-3,
        gamma=0.9,
        tau=0.9,
        epochs=10,
        eps_clip=0.3 
    )

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    agent, _, _ = trainer.train(env, agent, epochs=1_000, max_steps=1000, update_freq=10_000, initial_offset=0)
    return agent


def train_discrete():
    env = PendulumEnvDiscrete()

    actor = PolicyNetwork(3, [64], 3, model='softmax').double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = PPOSoftmaxNN(
        actor,
        critic,
        n_a=3,
        gamma=0.9,
        tau=0.9,
        lr_actor=1e-2,
        epochs=1,
        max_p=0.99,
        beta=1.0,
        eps_clip=0.01,
    )

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    agent, _, _ = trainer.train(env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0)
    return agent


def get_tensor(agent):
    d1 = np.linspace(-1, 1, 100)
    d2 = np.linspace(-1, 1, 100)
    d3 = np.linspace(-5, 5, 100)

    X = np.zeros([d1.size, d2.size, d3.size])
    for i1, s1 in enumerate(d1):
        for i2, s2 in enumerate(d2):
            for i3, s3 in enumerate(d3):
                mu, _ = agent.policy.actor(torch.tensor([s1, s2, s3]))
                X[i1, i2, i3] = mu.detach().item()
    X_debiased = X - X.mean()
    return X_debiased


def tensor_decomposition(X, maxrank, step, filename):
    ranks, errors = [], []
    for k in np.arange(1, maxrank, step):
        factors = parafac(X, rank=k, init='random')
        X_hat = tl.cp_to_tensor(factors)
        
        error = np.linalg.norm(X_hat.flatten() - X.flatten(), 2)
        normalizer = np.linalg.norm(X.flatten(), 2)
        ranks.append(k)
        errors.append(100*error/normalizer)

    np.save(f"results/ranks_pend_{filename}.npy", ranks)
    np.save(f"results/errors_pend_{filename}.npy", errors)


def run_ranks_exploration():
    agent_cont = train_continuous()
    agent_disc = train_discrete()

    X_cont = get_tensor(agent_cont)
    X_disc = get_tensor(agent_disc)

    tensor_decomposition(X_cont, 11, 1, 'cont')
    tensor_decomposition(X_disc, 35, 3, 'disc')
