import multiprocessing
import random
import numpy as np
import torch
import gymnasium as gym

from src.models import (
    PolicyNetwork,
    ValueNetwork,
    PolicyPARAFAC,
    ValuePARAFAC,
    PolicyRBF,
    ValueRBF
)
from src.utils import Discretizer
from src.agents.reinforce import ReinforceSoftmaxNN
from src.agents.ppo import PPOSoftmaxNN
from src.agents.trpo import TRPOSoftmaxNN
from src.train import Trainer


torch.set_num_threads(1)


def run_experiment_reinforce_nn(experiment_index):
    try:
        env = gym.make("MountainCar-v0")
        actor = PolicyNetwork(2, [8], 3, model='softmax').double()
        critic = ValueNetwork(2, [8], 1).double()

        agent = ReinforceSoftmaxNN(
            actor,
            critic,
            gamma=0.99,
            tau=0.99,
            lr_actor=1e-2,
            epochs=20,
            beta=0.1,
            n_a=3,
            max_p=0.9,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10_000, update_freq=20_000, initial_offset=10)
        return G
    except:
        return [0]*500


def run_experiment_ppo_nn(experiment_index):
    try:
        env = gym.make("MountainCar-v0")
        actor = PolicyNetwork(2, [8], 3, model='softmax').double()
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

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10_000, update_freq=20_000, initial_offset=10)
        return G
    except:
        return [0]*500


def run_experiment_trpo_nn(experiment_index):
    try:
        env = gym.make("MountainCar-v0")

        actor = PolicyNetwork(2, [8], 3, model='softmax').double()
        critic = ValueNetwork(2, [8], 1).double()

        agent = TRPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            gamma=0.99,
            delta=0.0001,
            tau=0.99,
            cg_dampening=0.2,
            cg_tolerance=1e-10,
            cg_iteration=10,
            beta=0.1, # PLAY HERE
            max_p=0.9
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=10_000,
            max_steps=10_000,
            update_freq=20_000, # PLAY HERE
            initial_offset=10,
        )
        return G
    except:
        return [0]*10_000


def run_experiment_reinforce_ten(experiment_index):
    try:
        env = gym.make("MountainCar-v0")

        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[10, 10],
        )

        actor = PolicyPARAFAC([10, 10, 3], 1, scale=0.01, model='softmax', bias=-1).double()
        critic = ValuePARAFAC([10, 10], 1, 1.0, bias=-1).double()

        agent = ReinforceSoftmaxNN(
            actor,
            critic,
            n_a=3,
            discretizer_actor=discretizer,
            discretizer_critic=discretizer,
            gamma=0.99,
            tau=0.99,
            lr_actor=1e-2,
            epochs=20,
            beta=0.1,
            max_p=0.9,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10_000, update_freq=20_000, initial_offset=10)
        return G
    except:
        return [0]*500


def run_experiment_ppo_ten(experiment_index):
    try:
        env = gym.make("MountainCar-v0")

        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[10, 10],
        )

        actor = PolicyPARAFAC([10, 10, 3], 1, scale=0.01, model='softmax', bias=-1).double()
        critic = ValuePARAFAC([10, 10], 1, scale=1.0, bias=-1).double()

        agent = PPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            discretizer_actor=discretizer,
            discretizer_critic=discretizer,
            gamma=0.99,
            tau=0.99,
            eps_clip=0.2,
            lr_actor=1e-2,
            epochs=20,
            beta=0.1,
            max_p=0.9,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10_000, update_freq=20_000, initial_offset=10)
        return G
    except:
        return [0]*500


def run_experiment_trpo_ten(experiment_index):
    try:
        env = gym.make("MountainCar-v0")

        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[10, 10],
        )

        actor = PolicyPARAFAC([10, 10, 3], 1, scale=1.0, model='softmax', bias=-1).double()
        critic = ValuePARAFAC([10, 10], 1, 1.0).double()


        agent = TRPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            discretizer_actor=discretizer,
            discretizer_critic=discretizer,
            gamma=0.99,
            delta=0.0001,
            tau=0.99,        
            cg_dampening=0.2,
            cg_tolerance=1e-10,
            cg_iteration=10,
            beta=0.1, # PLAY HERE
            max_p=0.9
        )

        trainer = Trainer(actor_opt='bcd', critic_opt='sgd')
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=10_000,
            max_steps=10_000,
            update_freq=20_000, # PLAY HERE
            initial_offset=10,
        )
        return G
    except:
        return [0]*10_000


def run_experiment_reinforce_rbf(experiment_index):
    try:
        env = gym.make("MountainCar-v0")
        actor = PolicyRBF(2, 50, 3, model='softmax').double()
        critic = ValueRBF(2, 50, 1).double()

        agent = ReinforceSoftmaxNN(
            actor,
            critic,
            gamma=0.99,
            tau=0.99,
            lr_actor=1e-2,
            epochs=20,
            beta=0.1,
            n_a=3,
            max_p=0.9,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10_000, update_freq=20_000, initial_offset=10)
        return G
    except:
        return [0]*500


def run_experiment_ppo_rbf(experiment_index):
    try:
        env = gym.make("MountainCar-v0")
        actor = PolicyRBF(2, 50, 3, model='softmax').double()
        critic = ValueRBF(2, 50, 1).double()

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

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(env, agent, epochs=500, max_steps=10_000, update_freq=20_000, initial_offset=10)
        return G
    except:
        return [0]*500


def run_experiment_trpo_rbf(experiment_index):
    try:
        env = gym.make("MountainCar-v0")

        actor = PolicyRBF(2, 50, 3, model='softmax').double()
        critic = ValueRBF(2, 50, 1).double()

        agent = TRPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            gamma=0.99,
            delta=0.0001,
            tau=0.99,
            cg_dampening=0.2,
            cg_tolerance=1e-10,
            cg_iteration=10,
            beta=0.1, # PLAY HERE
            max_p=0.9
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=10_000,
            max_steps=10_000,
            update_freq=20_000, # PLAY HERE
            initial_offset=10,
        )
        return G
    except:
        return [0]*10_000


def run_paralell(func, filename, num_experiments, num_processes):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(
            func,
            range(num_experiments)
        )
    results = np.array(results)
    np.save(f"results/{filename}.npy", results)


def run_mountaincar_disc_experiments(nexps, nprocs):
    run_paralell(run_experiment_reinforce_nn, 'mountaincar_discrete_reinforce_nn', nexps, nprocs)
    run_paralell(run_experiment_reinforce_rbf, 'mountaincar_discrete_reinforce_rbf', nexps, nprocs)
    run_paralell(run_experiment_reinforce_ten, 'mountaincar_discrete_reinforce_ten', nexps, nprocs)

    run_paralell(run_experiment_ppo_nn, 'mountaincar_discrete_ppo_nn', nexps, nprocs)
    run_paralell(run_experiment_ppo_rbf, 'mountaincar_discrete_ppo_rbf', nexps, nprocs)
    run_paralell(run_experiment_ppo_ten, 'mountaincar_discrete_ppo_ten', nexps, nprocs)

    run_paralell(run_experiment_trpo_nn, 'mountaincar_discrete_trpo_nn', nexps, nprocs)
    run_paralell(run_experiment_trpo_rbf, 'mountaincar_discrete_trpo_rbf', nexps, nprocs)
    run_paralell(run_experiment_trpo_ten, 'mountaincar_discrete_trpo_ten', nexps, nprocs)
