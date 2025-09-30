import numpy as np

import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
import gymnasium as gym
import torch

from src.environments.pendulum import PendulumEnvDiscrete
from src.environments.cartpole import CartpoleEnvContinuous
from src.environments.goddard import GoddardEnvContinuous, GoddardEnvDiscrete
from src.models import PolicyNetwork, ValueNetwork
from src.agents.ppo import PPOGaussianNN, PPOSoftmaxNN
from src.train import Trainer


def get_tensor(agent, is_discrete, ranges, grid_sizes, n_actions=3):
    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes] * len(ranges)
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(ranges, grid_sizes)]

    if is_discrete:
        X = np.zeros((*grid_sizes, n_actions))
    else:
        X = np.zeros((*grid_sizes,))

    for idx in np.ndindex(*grid_sizes):
        state_vals = [axes[d][idx[d]] for d in range(len(grid_sizes))]
        state = torch.tensor(state_vals)

        if is_discrete:
            logits = agent.policy.actor(state)
            X[idx] = logits.detach().numpy()
        else:
            mu, _ = agent.policy.actor(state)
            X[idx] = mu.detach().item()

    X = X.astype(np.float32)
    return X - X.mean()


def tensor_decomposition(X, maxrank, step, filename):
    ranks, errors_nfe, errors_max = [], [], []
    for k in np.arange(1, maxrank, step):
        factors = parafac(X, rank=k, init="svd")
        X_hat = tl.cp_to_tensor(factors)

        error = np.linalg.norm(X_hat.flatten() - X.flatten(), 2)
        normalizer = np.linalg.norm(X.flatten(), 2)
        errors_nfe.append(100 * error / normalizer)

        error = (X_hat.flatten() - X.flatten()).max()
        normalizer = X.flatten().max()
        errors_max.append(100 * error / normalizer)

        ranks.append(k)

    np.save(f"results/ranks_{filename}.npy", ranks)
    np.save(f"results/errors_nfe_{filename}.npy", errors_nfe)
    np.save(f"results/errors_max_{filename}.npy", errors_max)


def train_pendulum_continuous():
    env = gym.make("Pendulum-v1", g=9.81)

    actor = PolicyNetwork(3, [64], 1, model="gaussian").double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = PPOGaussianNN(
        actor, critic, lr_actor=1e-3, gamma=0.9, tau=0.9, epochs=10, eps_clip=0.3
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=1_000, max_steps=1000, update_freq=10_000, initial_offset=0
    )
    return agent


def train_pendulum_discrete():
    env = PendulumEnvDiscrete()

    actor = PolicyNetwork(3, [64], 3, model="softmax").double()
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

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0
    )
    return agent


def train_mountaincar_continuous():
    env = gym.make("MountainCarContinuous-v0")

    actor = PolicyNetwork(2, [8], 1).double()
    critic = ValueNetwork(2, [8], 1).double()

    agent = PPOGaussianNN(
        actor, critic, lr_actor=1e-3, gamma=0.99, tau=0.9, epochs=10, eps_clip=0.2
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=1_000, max_steps=10000, update_freq=10_000, initial_offset=10
    )
    return agent


def train_mountaincar_discrete():
    env = gym.make("MountainCar-v0")

    actor = PolicyNetwork(2, [8], 3, model="softmax").double()
    critic = ValueNetwork(2, [8], 1).double()

    agent = PPOSoftmaxNN(
        actor,
        critic,
        lr_actor=1e-2,
        gamma=0.99,
        tau=0.99,
        epochs=20,
        eps_clip=0.2,
        beta=0.1,
        n_a=3,
        max_p=0.9,
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env,
        agent,
        epochs=1_000,
        max_steps=10_000,
        update_freq=20_000,
        initial_offset=10,
    )
    return agent


def train_cartpole_continuous():
    env = CartpoleEnvContinuous()

    actor = PolicyNetwork(4, [64], 1).double()
    critic = ValueNetwork(4, [64], 1).double()

    agent = PPOGaussianNN(
        actor, critic, lr_actor=1e-2, gamma=0.99, tau=0.9, epochs=1, eps_clip=0.1
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=200, update_freq=1_000, initial_offset=10
    )
    return agent


def train_cartpole_discrete():
    env = gym.make("CartPole-v0")

    actor = PolicyNetwork(4, [64], 2, model="softmax").double()
    critic = ValueNetwork(4, [64], 1).double()

    agent = PPOSoftmaxNN(
        actor,
        critic,
        lr_actor=1e-2,
        gamma=0.99,
        tau=0.99,
        epochs=1,
        eps_clip=0.1,
        beta=0.1,
        n_a=2,
        max_p=0.9,
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=200, update_freq=1_000, initial_offset=10
    )
    return agent


def train_goddard_continuous():
    env = GoddardEnvContinuous()

    actor = PolicyNetwork(3, [64], 1).double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = PPOGaussianNN(
        actor, critic, lr_actor=1e-3, gamma=0.99, tau=0.99, epochs=1, eps_clip=0.1
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=10_000, max_steps=1_000, update_freq=1_000, initial_offset=10
    )
    return agent


def train_goddard_discrete():
    env = GoddardEnvDiscrete()

    actor = PolicyNetwork(3, [64], 10, model="softmax").double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = PPOSoftmaxNN(
        actor,
        critic,
        lr_actor=1e-3,
        gamma=0.99,
        tau=0.9,
        epochs=1,
        eps_clip=0.5,
        beta=1.0,
        n_a=10,
        max_p=0.9,
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    agent, _, _ = trainer.train(
        env, agent, epochs=10_000, max_steps=1_000, update_freq=1_000, initial_offset=10
    )
    return agent


def run_ranks_exploration():
    # Pendulum
    agent_cont = train_pendulum_continuous()
    agent_disc = train_pendulum_discrete()

    X_cont = get_tensor(
        agent_cont,
        is_discrete=False,
        ranges=[(-1.0, 1.0), (-1.0, 1.0), (-5.0, 5.0)],
        grid_sizes=50,
        n_actions=3,
    )
    X_disc = get_tensor(
        agent_disc,
        is_discrete=True,
        ranges=[(-1.0, 1.0), (-1.0, 1.0), (-5.0, 5.0)],
        grid_sizes=50,
        n_actions=3,
    )

    tensor_decomposition(X_cont, maxrank=16, step=1, filename="pendulum_cont")
    tensor_decomposition(X_disc, maxrank=16, step=1, filename="pendulum_disc")

    # Mountaincar
    agent_cont = train_mountaincar_continuous()
    agent_disc = train_mountaincar_discrete()

    X_cont = get_tensor(
        agent_cont,
        is_discrete=False,
        ranges=[(-1.2, 0.6), (-0.07, 0.07)],
        grid_sizes=100,
        n_actions=3,
    )
    X_disc = get_tensor(
        agent_disc,
        is_discrete=True,
        ranges=[(-1.2, 0.6), (-0.07, 0.07)],
        grid_sizes=100,
        n_actions=3,
    )

    tensor_decomposition(X_cont, maxrank=16, step=1, filename="mountaincar_cont")
    tensor_decomposition(X_disc, maxrank=16, step=1, filename="mountaincar_disc")

    # Cartpole
    agent_cont = train_cartpole_continuous()
    agent_disc = train_cartpole_discrete()

    X_cont = get_tensor(
        agent_cont,
        is_discrete=False,
        ranges=[(-4.8, 4.8), (-0.5, 0.5), (-0.42, 0.42), (-0.9, 0.9)],
        grid_sizes=20,
        n_actions=2,
    )
    X_disc = get_tensor(
        agent_disc,
        is_discrete=True,
        ranges=[(-4.8, 4.8), (-0.5, 0.5), (-0.42, 0.42), (-0.9, 0.9)],
        grid_sizes=20,
        n_actions=2,
    )

    tensor_decomposition(X_cont, maxrank=16, step=1, filename="cartpole_cont")
    tensor_decomposition(X_disc, maxrank=16, step=1, filename="cartpole_disc")

    # Goddard
    agent_cont = train_goddard_continuous()
    agent_disc = train_goddard_discrete()

    X_cont = get_tensor(
        agent_cont,
        is_discrete=False,
        ranges=[(0.0, 0.2), (1.00, 1.03), (0.6, 1.0)],
        grid_sizes=50,
        n_actions=10,
    )
    X_disc = get_tensor(
        agent_disc,
        is_discrete=True,
        ranges=[(0.0, 0.2), (1.00, 1.03), (0.6, 1.0)],
        grid_sizes=50,
        n_actions=10,
    )

    tensor_decomposition(X_cont, maxrank=16, step=1, filename="goddard_cont")
    tensor_decomposition(X_disc, maxrank=16, step=1, filename="goddard_disc")
