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
    env = gym.make("MountainCarContinuous-v0")
    actor = PolicyNetwork(2, [8], 1).double()
    critic = ValueNetwork(2, [8], 1).double()

    agent = ReinforceGaussianNN(
        actor,
        critic,
        gamma=0.99,
        tau=0.9,
        lr_actor=1e-3,
        epochs=10,
    )

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10000, update_freq=10_000, initial_offset=10)
    return G

def run_experiment_ppo_nn(experiment_index):
    env = gym.make("MountainCarContinuous-v0")
    actor = PolicyNetwork(2, [8], 1).double()
    critic = ValueNetwork(2, [8], 1).double()

    agent = PPOGaussianNN(
        actor,
        critic,
        lr_actor=1e-3,
        gamma=0.99,
        tau=0.9,
        epochs=10,
        eps_clip=0.2 
    )

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10000, update_freq=10_000, initial_offset=10)
    return G

def run_experiment_trpo_nn(experiment_index):
    env = gym.make("MountainCarContinuous-v0")
    actor = PolicyNetwork(2, [8], 1).double()
    critic = ValueNetwork(2, [8], 1).double()

    agent = TRPOGaussianNN(
        actor,
        critic,
        gamma=0.99,
        delta=0.01,
        tau=0.9,
        cg_dampening=0.1,
        cg_tolerance=1e-10,
        cg_iteration=10,
    )

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    _, G, _ = trainer.train(
        env,
        agent,
        epochs=5_00,
        max_steps=10_000,
        update_freq=10_000,
        initial_offset=10,
    )
    return G

def run_experiment_reinforce_ten(experiment_index):
    try:
        env = gym.make("MountainCarContinuous-v0")

        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[10, 10],
        )

        actor = PolicyPARAFAC([10, 10], 1, 0.1).double()
        critic = ValuePARAFAC([10, 10], 1, 0.1).double()

        agent = ReinforceGaussianNN(
            actor,
            critic,
            discretizer,
            discretizer,
            gamma=0.99,
            tau=0.9,
            lr_actor=1e-3,
            epochs=10
        )

        trainer = Trainer(actor_opt='bcd', critic_opt='bcd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10000, update_freq=10_000, initial_offset=10)
        print(np.array(G).shape, flush=True)
        return G
    except:
        print('hey', flush=True)
        return [0]*500

def run_experiment_ppo_ten(experiment_index):
    try:
        env = gym.make("MountainCarContinuous-v0")

        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[10, 10],
        )

        actor = PolicyPARAFAC([10, 10], 1, 0.1).double()
        critic = ValuePARAFAC([10, 10], 1, 0.1).double()

        agent = PPOGaussianNN(
            actor,
            critic,
            discretizer,
            discretizer,
            lr_actor=1e-3,
            gamma=0.99,
            tau=0.9,
            epochs=10,
            eps_clip=0.2 
        )

        trainer = Trainer(actor_opt='bcd', critic_opt='bcd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10000, update_freq=10_000, initial_offset=10)
        print(np.array(G).shape, flush=True)
        return G
    except:
        print('hey', flush=True)
        return [0]*500


def run_experiment_trpo_ten(experiment_index):
    try:
        env = gym.make("MountainCarContinuous-v0")

        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[10, 10],
        )

        actor = PolicyPARAFAC([10, 10], 1, 0.1).double()
        critic = ValuePARAFAC([10, 10], 1, 1.0).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            discretizer,
            discretizer,
            gamma=0.99,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='bcd', critic_opt='bcd')
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=500,
            max_steps=10_000,
            update_freq=10_000,
            initial_offset=10,
        )
        print(np.array(G).shape, flush=True)
        return G
    except:
        print('hey', flush=True)
        return [0]*500

if __name__ == "__main__":
    num_experiments = 100

    with multiprocessing.Pool(processes=num_experiments) as pool:
        results = pool.map(
            run_experiment_reinforce_ten,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('mountaincar_continuous_reinforce_ten.npy', results)

    with multiprocessing.Pool(processes=num_experiments) as pool:
        results = pool.map(
            run_experiment_ppo_ten,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('mountaincar_continuous_ppo_ten.npy', results)

    with multiprocessing.Pool(processes=num_experiments) as pool:
        results = pool.map(
            run_experiment_trpo_ten,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('mountaincar_continuous_trpo_ten.npy', results)
