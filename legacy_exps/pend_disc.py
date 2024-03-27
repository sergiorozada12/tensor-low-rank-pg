import multiprocessing
import numpy as np
import gymnasium as gym

from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize

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


class PendulumEnvDiscrete(PendulumEnv):
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        
        u = 2*u - 2.0
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}


def run_experiment_reinforce_nn(experiment_index):
    env = PendulumEnvDiscrete()

    actor = PolicyNetwork(3, [64], 3, model='softmax').double()
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

    trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
    _, G, _ = trainer.train(env, agent, epochs=2_000, max_steps=1000, update_freq=10_000, initial_offset=0)

    return G

def run_experiment_ppo_nn(experiment_index):
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
    _, G, _ = trainer.train(env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0)

    return G

def run_experiment_trpo_nn(experiment_index):
    try:
        env = PendulumEnvDiscrete()

        actor = PolicyNetwork(3, [64], 3, model='softmax').double()
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

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
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
        return [0]*4_000

def run_experiment_reinforce_ten(experiment_index):
    env = PendulumEnvDiscrete()

    discretizer = Discretizer(
        min_points=[-1, -1, -5],
        max_points=[1, 1, 5],
        buckets=[10, 10, 10],
    )

    actor = PolicyPARAFAC([10, 10, 10, 3], 10, 0.1, model='softmax').double()
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

    trainer = Trainer(actor_opt='bcd', critic_opt='sgd')
    _, G, _ = trainer.train(env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0)
    return G

def run_experiment_ppo_ten(experiment_index):
    env = PendulumEnvDiscrete()

    discretizer = Discretizer(
        min_points=[-1, -1, -5],
        max_points=[1, 1, 5],
        buckets=[10, 10, 10],
    )

    actor = PolicyPARAFAC([10, 10, 10, 3], 10, 0.1, model='softmax').double()
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

    trainer = Trainer(actor_opt='bcd', critic_opt='sgd')
    _, G, _ = trainer.train(env, agent, epochs=2_000, max_steps=1_000, update_freq=10_000, initial_offset=0)

    return G

def run_experiment_trpo_ten(experiment_index):
    try:
        env = PendulumEnvDiscrete()

        discretizer = Discretizer(
            min_points=[-1, -1, -5],
            max_points=[1, 1, 10],
            buckets=[10, 10, 10],
        )

        actor = PolicyPARAFAC([10, 10, 10, 3], 10, 0.1, model='softmax').double()
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

        trainer = Trainer(actor_opt='bcd', critic_opt='sgd')
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
        return [0]*4_000


if __name__ == "__main__":
    num_experiments = 100

    with multiprocessing.Pool(processes=num_experiments) as pool:
        results = pool.map(
            run_experiment_reinforce_nn,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('pendulum_discrete_reinforce_nn.npy', results)




    with multiprocessing.Pool(processes=num_experiments) as pool:
        results = pool.map(
            run_experiment_ppo_ten,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('pendulum_discrete_ppo_ten.npy', results)



    with multiprocessing.Pool(processes=num_experiments) as pool:
        results = pool.map(
            run_experiment_trpo_ten,
            range(num_experiments)
        )

    results = np.array(results)
    np.save('pendulum_discrete_trpo_ten.npy', results)
