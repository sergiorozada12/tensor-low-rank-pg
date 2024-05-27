import random
import numpy as np
import torch
import multiprocessing

from src.environments.wireless import WirelessCommunicationsEnv
from src.train import Trainer
from src.agents.ppo import PPOGaussianNN
from src.utils import Discretizer
from src.models import (
    PolicyNetwork,
    ValueNetwork,
    PolicyPARAFAC,
    ValuePARAFAC,
    PolicyRBF,
    ValueRBF
)


torch.set_num_threads(1)


def run_dqn_small(n):
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


def run_dqn_big(n):
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
        lr_actor=1e-2,
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


def run_rbf(n):
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

    actor = PolicyRBF(6, 100, 2, model='gaussian').double()
    critic = ValueRBF(6, 100, 1).double()

    agent = PPOGaussianNN(
        actor=actor,
        critic=critic,
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

    _, G, _ = trainer.train(
        env=env,
        agent=agent,
        epochs=5_000,
        max_steps=1_000,
        update_freq=5_000,
        initial_offset=0,
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


def run_wireless_experiments(nexps, nprocs):
    run_paralell(run_dqn_small, 'wireless_nn_small', nexps, nprocs)
    run_paralell(run_dqn_big, 'wireless_nn_big', nexps, nprocs)
    run_paralell(run_tlr, 'wireless_ten', nexps, nprocs)
    run_paralell(run_rbf, 'wireless_rbf', nexps, nprocs)
