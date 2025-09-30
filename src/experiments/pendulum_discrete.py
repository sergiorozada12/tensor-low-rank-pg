import multiprocessing
import random
import numpy as np
import torch

from src.environments.pendulum import PendulumEnvDiscrete

from src.models import (
    PolicyNetwork,
    ValueNetwork,
    PolicyPARAFAC,
    ValuePARAFAC,
    PolicyRBF,
    ValueRBF,
)
from src.utils import Discretizer
from src.agents.reinforce import ReinforceSoftmaxNN
from src.agents.ppo import PPOSoftmaxNN
from src.agents.trpo import TRPOSoftmaxNN
from src.train import Trainer


torch.set_num_threads(1)


def run_experiment_reinforce_nn(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = PendulumEnvDiscrete()

    actor = PolicyNetwork(3, [64], 3, model="softmax").double()
    critic = ValueNetwork(3, [64], 1).double()

    agent = ReinforceSoftmaxNN(
        actor,
        critic,
        n_a=3,
        gamma=0.9,
        tau=0.9,
        lr_actor=1e-2,
        epochs=1,
        max_p=0.99,
        beta=1.0,
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    _, G, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1000, update_freq=10_000, initial_offset=0
    )

    return G


def run_experiment_ppo_nn(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

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
    _, G, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0
    )

    return G


def run_experiment_trpo_nn(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    try:
        env = PendulumEnvDiscrete()

        actor = PolicyNetwork(3, [64], 3, model="softmax").double()
        critic = ValueNetwork(3, [64], 1).double()

        agent = TRPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            gamma=0.9,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.2,
            cg_tolerance=1e-10,
            cg_iteration=10,
            max_p=0.99,
            beta=1.0,
        )

        trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=4_000,
            max_steps=1_000,
            update_freq=50_000,
            initial_offset=0,
        )
        return G
    except:
        return [0] * 4_000


def run_experiment_reinforce_ten(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = PendulumEnvDiscrete()

    discretizer = Discretizer(
        min_points=[-1, -1, -5],
        max_points=[1, 1, 5],
        buckets=[10, 10, 10],
    )

    actor = PolicyPARAFAC([10, 10, 10, 3], 10, 0.1, model="softmax").double()
    critic = ValuePARAFAC([10, 10, 10], 10, 1.0).double()

    agent = ReinforceSoftmaxNN(
        actor,
        critic,
        n_a=3,
        discretizer_actor=discretizer,
        discretizer_critic=discretizer,
        gamma=0.9,
        tau=0.9,
        lr_actor=1e-2,
        epochs=10,
        max_p=0.99,
        beta=1.0,
    )

    trainer = Trainer(actor_opt="bcd", critic_opt="sgd")
    _, G, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0
    )
    return G


def run_experiment_ppo_ten(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = PendulumEnvDiscrete()

    discretizer = Discretizer(
        min_points=[-1, -1, -5],
        max_points=[1, 1, 5],
        buckets=[10, 10, 10],
    )

    actor = PolicyPARAFAC([10, 10, 10, 3], 10, 0.1, model="softmax").double()
    critic = ValuePARAFAC([10, 10, 10], 10, 1.0).double()

    agent = PPOSoftmaxNN(
        actor,
        critic,
        n_a=3,
        discretizer_actor=discretizer,
        discretizer_critic=discretizer,
        gamma=0.9,
        tau=0.9,
        lr_actor=1e-1,
        epochs=1,
        max_p=0.99,
        beta=1.0,
        eps_clip=0.01,
    )

    trainer = Trainer(actor_opt="bcd", critic_opt="sgd")
    _, G, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0
    )

    return G


def run_experiment_trpo_ten(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    try:
        env = PendulumEnvDiscrete()

        discretizer = Discretizer(
            min_points=[-1, -1, -5],
            max_points=[1, 1, 10],
            buckets=[10, 10, 10],
        )

        actor = PolicyPARAFAC([10, 10, 10, 3], 10, 0.1, model="softmax").double()
        critic = ValuePARAFAC([10, 10, 10], 10, 0.1).double()

        agent = TRPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            discretizer_actor=discretizer,
            discretizer_critic=discretizer,
            gamma=0.9,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.2,
            cg_tolerance=1e-10,
            cg_iteration=10,
            max_p=0.99,
            beta=1.0,
        )

        trainer = Trainer(actor_opt="bcd", critic_opt="sgd")
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=4_000,
            max_steps=1_000,
            update_freq=50_000,
            initial_offset=0,
        )

        return G
    except:
        return [0] * 4_000


def run_experiment_reinforce_rbf(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = PendulumEnvDiscrete()

    actor = PolicyRBF(3, 100, 3, model="softmax").double()
    critic = ValueRBF(3, 100, 1).double()

    agent = ReinforceSoftmaxNN(
        actor,
        critic,
        n_a=3,
        gamma=0.9,
        tau=0.9,
        lr_actor=1e-2,
        epochs=1,
        max_p=0.99,
        beta=1.0,
    )

    trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
    _, G, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1000, update_freq=10_000, initial_offset=0
    )

    return G


def run_experiment_ppo_rbf(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    env = PendulumEnvDiscrete()

    actor = PolicyRBF(3, 100, 3, model="softmax").double()
    critic = ValueRBF(3, 100, 1).double()

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
    _, G, _ = trainer.train(
        env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0
    )

    return G


def run_experiment_trpo_rbf(experiment_index):
    random.seed(experiment_index)
    np.random.seed(experiment_index)
    torch.manual_seed(experiment_index)

    try:
        env = PendulumEnvDiscrete()

        actor = PolicyRBF(3, 100, 3, model="softmax").double()
        critic = ValueRBF(3, 100, 1).double()

        agent = TRPOSoftmaxNN(
            actor,
            critic,
            n_a=3,
            gamma=0.9,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.2,
            cg_tolerance=1e-10,
            cg_iteration=10,
            max_p=0.99,
            beta=1.0,
        )

        trainer = Trainer(actor_opt="sgd", critic_opt="sgd")
        _, G, _ = trainer.train(
            env,
            agent,
            epochs=4_000,
            max_steps=1_000,
            update_freq=50_000,
            initial_offset=0,
        )
        return G
    except:
        return [0] * 4_000


def run_paralell(func, filename, num_experiments, num_processes):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(func, range(num_experiments))
    results = np.array(results)
    np.save(f"results/{filename}.npy", results)


def run_pendulum_disc_experiments(nexps, nprocs):
    run_paralell(
        run_experiment_reinforce_nn, "pendulum_discrete_reinforce_nn", nexps, nprocs
    )
    run_paralell(
        run_experiment_reinforce_rbf, "pendulum_discrete_reinforce_rbf", nexps, nprocs
    )
    run_paralell(
        run_experiment_reinforce_ten, "pendulum_discrete_reinforce_ten", nexps, nprocs
    )

    run_paralell(run_experiment_ppo_nn, "pendulum_discrete_ppo_nn", nexps, nprocs)
    run_paralell(run_experiment_ppo_rbf, "pendulum_discrete_ppo_rbf", nexps, nprocs)
    run_paralell(run_experiment_ppo_ten, "pendulum_discrete_ppo_ten", nexps, nprocs)

    run_paralell(run_experiment_trpo_nn, "pendulum_discrete_trpo_nn", nexps, nprocs)
    run_paralell(run_experiment_trpo_rbf, "pendulum_discrete_trpo_rbf", nexps, nprocs)
    run_paralell(run_experiment_trpo_ten, "pendulum_discrete_trpo_ten", nexps, nprocs)
