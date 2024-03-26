import numpy as  np
import matplotlib
import matplotlib.pyplot as plt

from src.utils import OOMFormatter


def plot_control():
    pend_cont_rein_nn = np.load("results/pendulum_continuous_reinforce_nn.npy")
    pend_cont_rein_ten = np.load("results/pendulum_continuous_reinforce_ten.npy")
    pend_cont_ppo_nn = np.load("results/pendulum_continuous_ppo_nn.npy")
    pend_cont_ppo_ten = np.load("results/pendulum_continuous_ppo_ten.npy")
    pend_cont_trpo_nn = np.load("results/pendulum_continuous_trpo_nn.npy")
    pend_cont_trpo_ten = np.load("results/pendulum_continuous_trpo_ten.npy")

    pend_disc_rein_nn = np.load("results/pendulum_discrete_reinforce_nn.npy")
    pend_disc_rein_ten = np.load("results/pendulum_discrete_reinforce_ten.npy")
    pend_disc_ppo_nn = np.load("results/pendulum_discrete_ppo_nn.npy")
    pend_disc_ppo_ten = np.load("results/pendulum_discrete_ppo_ten.npy")
    pend_disc_trpo_nn = np.load("results/pendulum_discrete_trpo_nn.npy")
    pend_disc_trpo_ten = np.load("results/pendulum_discrete_trpo_ten.npy")

    mount_cont_rein_nn = np.load("results/mountaincar_continuous_reinforce_nn.npy")
    mount_cont_rein_ten = np.load("results/mountaincar_continuous_reinforce_ten.npy")
    mount_cont_ppo_nn = np.load("results/mountaincar_continuous_ppo_nn.npy")
    mount_cont_ppo_ten = np.load("results/mountaincar_continuous_ppo_ten.npy")
    mount_cont_trpo_nn = np.load("results/mountaincar_continuous_trpo_nn.npy")
    mount_cont_trpo_ten = np.load("results/mountaincar_continuous_trpo_ten.npy")

    mount_disc_rein_nn = np.load("results/mountaincar_discrete_reinforce_nn.npy")
    mount_disc_rein_ten = np.load("results/mountaincar_discrete_reinforce_ten.npy")
    mount_disc_ppo_nn = np.load("results/mountaincar_discrete_ppo_nn.npy")
    mount_disc_ppo_ten = np.load("results/mountaincar_discrete_ppo_ten.npy")    
    mount_disc_trpo_nn = np.load("results/mountaincar_discrete_trpo_nn.npy")
    mount_disc_trpo_ten = np.load("results/mountaincar_discrete_trpo_ten.npy")

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(4, 3, figsize=(10, 10))  # Adjust figsize as needed

        # Pendulum Continuous
        ax = axs[0, 0]

        med = np.median(pend_cont_rein_nn, axis=0)
        p25 = np.percentile(pend_cont_rein_nn, q=25, axis=0)
        p75 = np.percentile(pend_cont_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 642 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)    

        med = np.median(pend_cont_rein_ten, axis=0)
        p25 = np.percentile(pend_cont_rein_ten, q=25, axis=0)
        p75 = np.percentile(pend_cont_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 300 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 2_000)
        ax.set_ylabel("Return - PendulumCont")
        ax.grid()

        ax.legend()

        ax = axs[0, 1]

        med = np.median(pend_cont_ppo_nn, axis=0)
        p25 = np.percentile(pend_cont_ppo_nn, q=25, axis=0)
        p75 = np.percentile(pend_cont_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 642 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_cont_ppo_ten, axis=0)
        p25 = np.percentile(pend_cont_ppo_ten, q=25, axis=0)
        p75 = np.percentile(pend_cont_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 300 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 1_000)
        ax.grid()

        ax.legend()

        ax = axs[0, 2]

        med = np.median(pend_cont_trpo_nn, axis=0)
        p25 = np.percentile(pend_cont_trpo_nn, q=25, axis=0)
        p75 = np.percentile(pend_cont_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 642 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_cont_trpo_ten, axis=0)
        p25 = np.percentile(pend_cont_trpo_ten, q=25, axis=0)
        p75 = np.percentile(pend_cont_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 300 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.grid()

        ax.legend()

        # Pendulum Discrete
        ax = axs[1, 0]

        med = np.median(pend_disc_rein_nn, axis=0)
        p25 = np.percentile(pend_disc_rein_nn, q=25, axis=0)
        p75 = np.percentile(pend_disc_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 772 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_disc_rein_ten, axis=0)
        p25 = np.percentile(pend_disc_rein_ten, q=25, axis=0)
        p75 = np.percentile(pend_disc_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 630 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 2_000)
        ax.set_ylabel("Return - PendulumDisc")
        ax.grid()

        ax.legend()

        ax = axs[1, 1]

        med = np.median(pend_disc_ppo_nn, axis=0)
        p25 = np.percentile(pend_disc_ppo_nn, q=25, axis=0)
        p75 = np.percentile(pend_disc_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 772 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_disc_ppo_ten, axis=0)
        p25 = np.percentile(pend_disc_ppo_ten, q=25, axis=0)
        p75 = np.percentile(pend_disc_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 630 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 2_000)
        ax.grid()

        ax.legend()

        ax = axs[1, 2]

        med = np.median(pend_disc_trpo_nn, axis=0)
        p25 = np.percentile(pend_disc_trpo_nn, q=25, axis=0)
        p75 = np.percentile(pend_disc_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 772 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_disc_trpo_ten, axis=0)
        p25 = np.percentile(pend_disc_trpo_ten, q=25, axis=0)
        p75 = np.percentile(pend_disc_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 630 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 4_000)
        ax.grid()

        ax.legend()

        # Mountaincar Continuous
        ax = axs[2, 0]

        med = np.median(mount_cont_rein_nn, axis=0)
        p25 = np.percentile(mount_cont_rein_nn, q=25, axis=0)
        p75 = np.percentile(mount_cont_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 62 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_cont_rein_ten, axis=0)
        p25 = np.percentile(mount_cont_rein_ten, q=25, axis=0)
        p75 = np.percentile(mount_cont_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 40 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_ylabel("Return - MountaincarCont")
        ax.grid()

        ax.legend()

        ax = axs[2, 1]

        med = np.median(mount_cont_ppo_nn, axis=0)
        p25 = np.percentile(mount_cont_ppo_nn, q=25, axis=0)
        p75 = np.percentile(mount_cont_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 62 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(mount_cont_ppo_ten, axis=0)
        p25 = np.percentile(mount_cont_ppo_ten, q=25, axis=0)
        p75 = np.percentile(mount_cont_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 40 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.grid()
        
        ax.legend()

        ax = axs[2, 2]

        med = np.median(mount_cont_trpo_nn, axis=0)
        p25 = np.percentile(mount_cont_trpo_nn, q=25, axis=0)
        p75 = np.percentile(mount_cont_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 62 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_cont_trpo_ten, axis=0)
        p25 = np.percentile(mount_cont_trpo_ten, q=25, axis=0)
        p75 = np.percentile(mount_cont_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 40 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 200)
        ax.grid()

        ax.legend()

        # Mountaincar Discrete
        ax = axs[3, 0]

        med = np.median(mount_disc_rein_nn, axis=0)
        p25 = np.percentile(mount_disc_rein_nn, q=25, axis=0)
        p75 = np.percentile(mount_disc_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 84 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_disc_rein_ten, axis=0)
        p25 = np.percentile(mount_disc_rein_ten, q=25, axis=0)
        p75 = np.percentile(mount_disc_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 43 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_xlabel("Episodes - AC")
        ax.set_ylabel("Return - MountaincarDisc")
        ax.grid()

        ax.legend()

        ax = axs[3, 1]

        med = np.median(mount_disc_ppo_nn, axis=0)
        p25 = np.percentile(mount_disc_ppo_nn, q=25, axis=0)
        p75 = np.percentile(mount_disc_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 84 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_disc_ppo_ten, axis=0)
        p25 = np.percentile(mount_disc_ppo_ten, q=25, axis=0)
        p75 = np.percentile(mount_disc_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 43 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_xlabel("Episodes - PPO")
        ax.grid()

        ax.legend()

        ax = axs[3, 2]

        med = np.median(mount_disc_trpo_nn, axis=0)
        p25 = np.percentile(mount_disc_trpo_nn, q=25, axis=0)
        p75 = np.percentile(mount_disc_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 84 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_disc_trpo_ten, axis=0)
        p25 = np.percentile(mount_disc_trpo_ten, q=25, axis=0)
        p75 = np.percentile(mount_disc_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 43 params.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 10_000)
        ax.set_xlabel("Episodes - TRPO")
        ax.grid()

        ax.legend()

        plt.tight_layout()
        fig.savefig('fig_1.jpg', dpi=300)
