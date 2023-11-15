import src.train as train
import src.utils as utils
import src.models as models
from src.environments import pendulum
from src.agents import reinforce, trpo, ppo
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

if __name__ == "__main__":

    res_nn_reinforce, res_nn_TRPO, res_nn_PPO = [], [], []
    res_matrix_reinforce, res_matrix_TRPO, res_matrix_PPO = [], [], []
    res_tensor_reinforce, res_tensor_TRPO, res_tensor_PPO = [], [], []

    env = gym.make('Pendulum-v1', g=9.81)
    num_experiments = 100

    discretizer_actor_matrix = utils.Discretizer(
        min_points= [-1, -1, -8],
        max_points= [1, 1, 8],
        buckets= [10, 10, 10],
        dimensions= [[0],[1 ,2]]
    )

    discretizer_critic_matrix = utils.Discretizer(
        min_points= [-1, -1, -8],
        max_points= [1, 1, 8],
        buckets= [16, 16, 10],
        dimensions= [[0],[1 ,2]]
    )

    discretizer_actor_tensor = utils.Discretizer(
        [-1, -1, -8],
        [1, 1, 8],
        [10, 10, 16]
    )

    discretizer_critic_tensor = utils.Discretizer(
        [-1, -1, -8],
        [1, 1, 8], 
        [16, 16, 16]
    )

    Trainer = train.Trainer("sgd", "sgd")

    #RREINFORCE
    for _ in range(num_experiments):
        #NN
        actor = models.PolicyNetwork(3,[32],1).double()
        critc = models.ValueNetwork(3,[32],1).double()

        agent = reinforce.ReinforceGaussianNN(
            actor, 
            critc, 
            gamma=.9, 
            tau=.9, 
            lr_actor=1e-5
        )

        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=3000, 
            max_steps=1500, 
            update_freq=15000, 
            initial_offset=0
        )

        res_nn_reinforce.append(totals)

        #Matrix

        actor = models.PolicyLR(
            n=discretizer_actor_matrix.N, 
            m=discretizer_actor_matrix.M, 
            k=4
        )
        critc = models.ValueLR(
            n=discretizer_critic_matrix.N, 
            m=discretizer_critic_matrix.M, 
            k=4
        )

        agent = reinforce.ReinforceGaussianNN(
            actor, 
            critc,
            discretizer_actor= discretizer_actor_matrix,
            discretizer_critic=discretizer_critic_matrix, 
            gamma=.9, 
            tau=.9, 
            lr_actor=1e-5
        )

        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=3000, 
            max_steps=1500, 
            update_freq=15000, 
            initial_offset=0
        )

        res_matrix_reinforce.append(totals)
        
        #Tensor

        actor = models.PolicyPARAFAC(
            [10, 10, 16], 
            k=8
        )
        critc = models.ValuePARAFAC( 
            [16, 16, 16], 
            k=8
        )

        agent = reinforce.ReinforceGaussianNN(
            actor, 
            critc,
            discretizer_actor= discretizer_actor_tensor,
            discretizer_critic=discretizer_critic_tensor, 
            gamma=.9, 
            tau=.9, 
            lr_actor=1e-5
        )
        
        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=3000, 
            max_steps=1500, 
            update_freq=20000, 
            initial_offset=0
        )

        res_tensor_reinforce.append(totals)


        

    #TRPO
    for _ in range(num_experiments):
        #NN
        actor = models.PolicyNetwork(3,[32],1).double()
        critc = models.ValueNetwork(3,[32],1).double()

        agent = trpo.TRPOGaussianNN(
            actor, 
            critc, 
            gamma=.9, 
            tau=.9, 
            delta=.05, 
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10)

        
        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=3000, 
            max_steps=1000, 
            update_freq=15000, 
            initial_offset=0
        )

        res_nn_TRPO.append(totals)

        #Matrix

        actor = models.PolicyLR(
            n=discretizer_actor_matrix.N, 
            m=discretizer_actor_matrix.M, 
            k=4
        )
        critc = models.ValueLR(
            n=discretizer_critic_matrix.N, 
            m=discretizer_critic_matrix.M, 
            k=4
        )

        agent = trpo.TRPOGaussianNN(
            actor, 
            critc, 
            discretizer_actor= discretizer_actor_matrix,
            discretizer_critic=discretizer_critic_matrix, 
            gamma=.9, 
            tau=.9, 
            delta=.05, 
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10
        )

        
        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=1000, 
            max_steps=1000, 
            update_freq=15000, 
            initial_offset=0
        )

        res_matrix_TRPO.append(totals)
        
        #Tensor

        actor = models.PolicyPARAFAC(
            [10, 10, 10], 
            k=8
        )
        critc = models.ValuePARAFAC(
            [16, 16, 10], 
            k=8
        )

        agent = trpo.TRPOGaussianNN(
            actor, 
            critc, 
            discretizer_actor= discretizer_actor_tensor,
            discretizer_critic=discretizer_critic_tensor, 
            gamma=.9 ,
            tau=.9, 
            delta=.05, 
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10
        )

        
        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=1000, 
            max_steps=2000, 
            update_freq=15000, 
            initial_offset=0
        )
        res_tensor_TRPO.append(totals)

    #PPO
    for _ in range(num_experiments):
        #NN
        actor = models.PolicyNetwork(3,[32],1).double()
        critc = models.ValueNetwork(3,[32],1).double()

        agent = ppo.PPOGaussianNN(
            actor, 
            critc, 
            gamma=.9, 
            tau=.9, 
            lr_actor=1e-4, 
            epochs=1000, 
            eps_clip=0.2
        )

        actor = models.PolicyLR(
            n=discretizer_actor_matrix.N, 
            m=discretizer_actor_matrix.M, 
            k=4
        )
        critc = models.ValueLR(
            n=discretizer_critic_matrix.N, 
            m=discretizer_critic_matrix.M, 
            k=4
        )

        
        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=1000, 
            max_steps=1000, 
            update_freq=10000, 
            initial_offset=0
        )

        res_nn_PPO.append(totals)

        #Matrix
        
        agent = ppo.PPOGaussianNN(
            actor, 
            critc,
            discretizer_actor= discretizer_actor_matrix,
            discretizer_critic=discretizer_critic_matrix, 
            gamma=.9, 
            tau=.9, 
            lr_actor=1e-5, 
            epochs=1000, 
            eps_clip=0.2)


        
        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=2500, 
            max_steps=1000, 
            update_freq=10000, 
            initial_offset=0
        )

        res_matrix_PPO.append(totals)
        
        #Tensor

        actor = models.PolicyPARAFAC(
            [10, 10, 10], 
            k=8
        )
        critc = models.ValuePARAFAC(
            [16, 16, 10], 
            k=8
        )

        agent = ppo.PPOGaussianNN(
            actor,
            critc,
            discretizer_actor= discretizer_actor_tensor,
            discretizer_critic=discretizer_actor_tensor, 
            gamma=.9, 
            tau=.9, 
            lr_actor=1e-5, 
            epochs=1000, 
            eps_clip=0.2
        )


        _, totals,_ = Trainer.train(
            env, 
            agent, 
            epochs=7000, 
            max_steps=1000, 
            update_freq=10000, 
            initial_offset=0
        )
    
