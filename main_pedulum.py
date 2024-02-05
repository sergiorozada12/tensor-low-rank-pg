import multiprocessing
import numpy as np
import gymnasium as gym

from src.models import (
    PolicyNetwork,
    ValueNetwork,
    PolicyLR,
    ValueLR,
    PolicyPARAFAC,
    ValuePARAFAC
)
from src.utils import Discretizer
from src.agents.reinforce import ReinforceGaussianNN, ReinforceSoftmaxNN
from src.agents.ppo import PPOGaussianNN, PPOSoftmaxNN
from src.agents.trpo import TRPOGaussianNN, TRPOSoftmaxNN
from src.train import Trainer


def run_experiment_reinforce_nn(experiment_index):
    env = gym.make('Pendulum-v1', g=9.81)
    actor = PolicyNetwork(3, [64], 1).double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = ReinforceGaussianNN(
        actor,
        critic,
        gamma=0.9,
        tau=0.9,
        lr_actor=1e-3,
        epochs=10,
    )

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    _, G_reinforce_nn, _ = trainer.train(env, agent, epochs=2000, max_steps=1000, update_freq=10_000, initial_offset=0)

    return G_reinforce_nn


if __name__ == "__main__":
    num_experiments = 100

    with multiprocessing.Pool(processes=1) as pool:
        results = pool.map(
            run_experiment_reinforce_nn,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('pendulum_continuous_reinforce_nn.npy', results)
