import multiprocessing
import random
import numpy as np
import gymnasium as gym
import torch

from src.models import (
    PolicyNetwork,
    ValueNetwork,
    PolicyPARAFAC,
    ValuePARAFAC,
    PolicyRBF,
    ValueRBF
)
from src.utils import Discretizer
from src.agents.reinforce import ReinforceGaussianNN
from src.agents.ppo import PPOGaussianNN
from src.agents.trpo import TRPOGaussianNN
from src.train import Trainer


torch.set_num_threads(1)


def run_experiment_reinforce_nn(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
        return G
    except:
        return [0]*500


def run_experiment_trpo_ten(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
        return G
    except:
        return [0]*500


def run_experiment_reinforce_rbf(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = gym.make("MountainCarContinuous-v0")
    actor = PolicyRBF(2, 20, 1, model='gaussian').double()
    critic = ValueRBF(2, 20, 1).double()

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

def run_experiment_ppo_rbf(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = gym.make("MountainCarContinuous-v0")
    actor = PolicyRBF(2, 20, 1, model='gaussian').double()
    critic = ValueRBF(2, 20, 1).double()

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

def run_experiment_trpo_rbf(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = gym.make("MountainCarContinuous-v0")
    actor = PolicyRBF(2, 20, 1, model='gaussian').double()
    critic = ValueRBF(2, 20, 1).double()

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


def run_paralell(func, filename, num_experiments, num_processes):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(
            func,
            range(num_experiments)
        )
    results = np.array(results)
    np.save(f"results/{filename}.npy", results)


def run_mountaincar_cont_experiments(nexps, nprocs):
    run_paralell(run_experiment_reinforce_nn, 'mountaincar_continuous_reinforce_nn', nexps, nprocs)
    run_paralell(run_experiment_reinforce_rbf, 'mountaincar_continuous_reinforce_rbf', nexps, nprocs)
    run_paralell(run_experiment_reinforce_ten, 'mountaincar_continuous_reinforce_ten', nexps, nprocs)

    run_paralell(run_experiment_ppo_nn, 'mountaincar_continuous_ppo_nn', nexps, nprocs)
    run_paralell(run_experiment_ppo_rbf, 'mountaincar_continuous_ppo_rbf', nexps, nprocs)
    run_paralell(run_experiment_ppo_ten, 'mountaincar_continuous_ppo_ten', nexps, nprocs)

    run_paralell(run_experiment_trpo_nn, 'mountaincar_continuous_trpo_nn', nexps, nprocs)
    run_paralell(run_experiment_trpo_rbf, 'mountaincar_continuous_trpo_rbf', nexps, nprocs)
    run_paralell(run_experiment_trpo_ten, 'mountaincar_continuous_trpo_ten', nexps, nprocs)
