import numpy as  np
import matplotlib
import matplotlib.pyplot as plt

from src.utils import OOMFormatter


def plot_control():
    pend_cont_rein_nn = np.load("results/pendulum_continuous_reinforce_nn.npy")
    pend_cont_rein_ten = np.load("results/pendulum_continuous_reinforce_ten.npy")
    pend_cont_rein_rbf = np.load("results/pendulum_continuous_reinforce_rbf.npy")
    pend_cont_ppo_nn = np.load("results/pendulum_continuous_ppo_nn.npy")
    pend_cont_ppo_ten = np.load("results/pendulum_continuous_ppo_ten.npy")
    pend_cont_ppo_rbf = np.load("results/pendulum_continuous_ppo_rbf.npy")
    pend_cont_trpo_nn = np.load("results/pendulum_continuous_trpo_nn.npy")
    pend_cont_trpo_ten = np.load("results/pendulum_continuous_trpo_ten.npy")
    pend_cont_trpo_rbf = np.load("results/pendulum_continuous_trpo_rbf.npy")

    pend_disc_rein_nn = np.load("results/pendulum_discrete_reinforce_nn.npy")
    pend_disc_rein_ten = np.load("results/pendulum_discrete_reinforce_ten.npy")
    pend_disc_rein_rbf = np.load("results/pendulum_discrete_reinforce_rbf.npy")
    pend_disc_ppo_nn = np.load("results/pendulum_discrete_ppo_nn.npy")
    pend_disc_ppo_ten = np.load("results/pendulum_discrete_ppo_ten.npy")
    pend_disc_ppo_rbf = np.load("results/pendulum_discrete_ppo_rbf.npy")
    pend_disc_trpo_nn = np.load("results/pendulum_discrete_trpo_nn.npy")
    pend_disc_trpo_ten = np.load("results/pendulum_discrete_trpo_ten.npy")
    pend_disc_trpo_rbf = np.load("results/pendulum_discrete_trpo_rbf.npy")

    mount_cont_rein_nn = np.load("results/mountaincar_continuous_reinforce_nn.npy")
    mount_cont_rein_ten = np.load("results/mountaincar_continuous_reinforce_ten.npy")
    mount_cont_rein_rbf = np.load("results/mountaincar_continuous_reinforce_rbf.npy")
    mount_cont_ppo_nn = np.load("results/mountaincar_continuous_ppo_nn.npy")
    mount_cont_ppo_ten = np.load("results/mountaincar_continuous_ppo_ten.npy")
    mount_cont_ppo_rbf = np.load("results/mountaincar_continuous_ppo_rbf.npy")
    mount_cont_trpo_nn = np.load("results/mountaincar_continuous_trpo_nn.npy")
    mount_cont_trpo_ten = np.load("results/mountaincar_continuous_trpo_ten.npy")
    mount_cont_trpo_rbf = np.load("results/mountaincar_continuous_trpo_rbf.npy")

    mount_disc_rein_nn = np.load("results/mountaincar_discrete_reinforce_nn.npy")
    mount_disc_rein_ten = np.load("results/mountaincar_discrete_reinforce_ten.npy")
    mount_disc_rein_rbf = np.load("results/mountaincar_discrete_reinforce_rbf.npy")
    mount_disc_ppo_nn = np.load("results/mountaincar_discrete_ppo_nn.npy")
    mount_disc_ppo_ten = np.load("results/mountaincar_discrete_ppo_ten.npy")
    mount_disc_ppo_rbf = np.load("results/mountaincar_discrete_ppo_rbf.npy")
    mount_disc_trpo_nn = np.load("results/mountaincar_discrete_trpo_nn.npy")
    mount_disc_trpo_ten = np.load("results/mountaincar_discrete_trpo_ten.npy")
    mount_disc_trpo_rbf = np.load("results/mountaincar_discrete_trpo_rbf.npy")

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 3, figsize=(10, 5))  # Adjust figsize as needed

        # Pendulum Continuous

        ax = axs[0, 0]
        
        med = np.median(pend_cont_rein_rbf, axis=0)
        p25 = np.percentile(pend_cont_rein_rbf, q=25, axis=0)
        p75 = np.percentile(pend_cont_rein_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 603 pars.', alpha=.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_cont_rein_nn, axis=0)
        p25 = np.percentile(pend_cont_rein_nn, q=25, axis=0)
        p75 = np.percentile(pend_cont_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 642 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)    

        med = np.median(pend_cont_rein_ten, axis=0)
        p25 = np.percentile(pend_cont_rein_ten, q=25, axis=0)
        p75 = np.percentile(pend_cont_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 300 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 2_000)
        ax.set_xticks([0, 1_000, 2_000])
        ax.set_ylabel("Return - PendulumCont")
        ax.grid()
        
        #ax.legend()
        
        ####

        ax = axs[0, 1]
        
        med = np.median(pend_cont_ppo_rbf, axis=0)
        p25 = np.percentile(pend_cont_ppo_rbf, q=25, axis=0)
        p75 = np.percentile(pend_cont_ppo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 603 pars.', alpha=.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_cont_ppo_nn, axis=0)
        p25 = np.percentile(pend_cont_ppo_nn, q=25, axis=0)
        p75 = np.percentile(pend_cont_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 642 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(pend_cont_ppo_ten, axis=0)
        p25 = np.percentile(pend_cont_ppo_ten, q=25, axis=0)
        p75 = np.percentile(pend_cont_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 300 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 1_000)
        ax.set_xticks([0, 500, 1_000])
        ax.grid()
        
        #ax.legend()

        ###

        ax = axs[0, 2]
        
        med = np.median(pend_cont_trpo_rbf, axis=0)
        p25 = np.percentile(pend_cont_trpo_rbf, q=25, axis=0)
        p75 = np.percentile(pend_cont_trpo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 603 pars.', alpha=.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_cont_trpo_nn, axis=0)
        p25 = np.percentile(pend_cont_trpo_nn, q=25, axis=0)
        p75 = np.percentile(pend_cont_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 642 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(pend_cont_trpo_ten, axis=0)
        p25 = np.percentile(pend_cont_trpo_ten, q=25, axis=0)
        p75 = np.percentile(pend_cont_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 300 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
        ax.grid()
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        # Mountaincar Continuous
        ax = axs[1, 0]
        
        med = np.median(mount_cont_rein_rbf, axis=0)
        p25 = np.percentile(mount_cont_rein_rbf, q=25, axis=0)
        p75 = np.percentile(mount_cont_rein_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 43 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        med = np.median(mount_cont_rein_nn, axis=0)
        p25 = np.percentile(mount_cont_rein_nn, q=25, axis=0)
        p75 = np.percentile(mount_cont_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 62 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(mount_cont_rein_ten, axis=0)
        p25 = np.percentile(mount_cont_rein_ten, q=25, axis=0)
        p75 = np.percentile(mount_cont_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 40 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
        ax.set_ylabel("Return - MountaincarCont")
        ax.set_xlabel("Episodes - AC")
        ax.grid()
        
        #ax.legend()

        ###
        ax = axs[1, 1]
        
        med = np.median(mount_cont_ppo_rbf, axis=0)
        p25 = np.percentile(mount_cont_ppo_rbf, q=25, axis=0)
        p75 = np.percentile(mount_cont_ppo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 43 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_cont_ppo_nn, axis=0)
        p25 = np.percentile(mount_cont_ppo_nn, q=25, axis=0)
        p75 = np.percentile(mount_cont_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 62 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(mount_cont_ppo_ten, axis=0)
        p25 = np.percentile(mount_cont_ppo_ten, q=25, axis=0)
        p75 = np.percentile(mount_cont_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 40 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xticks([0, 250, 500])
        ax.set_xlim(0, 500)
        ax.set_xlabel("Episodes - PPO")
        ax.grid()
        
        #ax.legend()

        ###
        ax = axs[1, 2]
        
        med = np.median(mount_cont_trpo_rbf, axis=0)
        p25 = np.percentile(mount_cont_trpo_rbf, q=25, axis=0)
        p75 = np.percentile(mount_cont_trpo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 43 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_cont_trpo_nn, axis=0)
        p25 = np.percentile(mount_cont_trpo_nn, q=25, axis=0)
        p75 = np.percentile(mount_cont_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 62 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(mount_cont_trpo_ten, axis=0)
        p25 = np.percentile(mount_cont_trpo_ten, q=25, axis=0)
        p75 = np.percentile(mount_cont_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 40 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xticks([0, 100, 200])
        ax.set_xlim(0, 200)
        ax.set_xlabel("Episodes - TRPO")
        ax.grid()
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)


        plt.tight_layout()
        fig.savefig('figures/fig_2.jpg', dpi=300)
        plt.show()


    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 3, figsize=(10, 5))  # Adjust figsize as needed


        # Pendulum Discrete

        ax = axs[0, 0]
        
        med = np.median(pend_disc_rein_rbf, axis=0)
        p25 = np.percentile(pend_disc_rein_rbf, q=25, axis=0)
        p75 = np.percentile(pend_disc_rein_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 404 pars.', alpha=0.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        med = np.median(pend_disc_rein_nn, axis=0)
        p25 = np.percentile(pend_disc_rein_nn, q=25, axis=0)
        p75 = np.percentile(pend_disc_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 772 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_disc_rein_ten, axis=0)
        p25 = np.percentile(pend_disc_rein_ten, q=25, axis=0)
        p75 = np.percentile(pend_disc_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 630 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 2_000)
        ax.set_xticks([0, 1_000, 2_000])
        ax.set_ylabel("Return - PendulumDisc")
        ax.grid()
        
        #ax.legend()
        
        ####

        ax = axs[0, 1]
        
        med = np.median(pend_disc_ppo_rbf, axis=0)
        p25 = np.percentile(pend_disc_ppo_rbf, q=25, axis=0)
        p75 = np.percentile(pend_disc_ppo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 404 pars.', alpha=0.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_disc_ppo_nn, axis=0)
        p25 = np.percentile(pend_disc_ppo_nn, q=25, axis=0)
        p75 = np.percentile(pend_disc_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 772 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(pend_disc_ppo_ten, axis=0)
        p25 = np.percentile(pend_disc_ppo_ten, q=25, axis=0)
        p75 = np.percentile(pend_disc_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 630 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 2_000)
        ax.set_xticks([0, 1_000, 2_000])
        ax.grid()
        
        #ax.legend()
        
        ####

        ax = axs[0, 2]
        
        med = np.median(pend_disc_trpo_rbf, axis=0)
        p25 = np.percentile(pend_disc_trpo_rbf, q=25, axis=0)
        p75 = np.percentile(pend_disc_trpo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 404 pars.', alpha=0.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(pend_disc_trpo_nn, axis=0)
        p25 = np.percentile(pend_disc_trpo_nn, q=25, axis=0)
        p75 = np.percentile(pend_disc_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 772 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(pend_disc_trpo_ten, axis=0)
        p25 = np.percentile(pend_disc_trpo_ten, q=25, axis=0)
        p75 = np.percentile(pend_disc_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 630 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 4_000)
        ax.set_xticks([0, 2_000, 4_000])
        ax.grid()
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        # Mountaincar Discrete
        ax = axs[1, 0]
        
        med = np.median(mount_disc_rein_rbf, axis=0)
        p25 = np.percentile(mount_disc_rein_rbf, q=25, axis=0)
        p75 = np.percentile(mount_disc_rein_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 204 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_disc_rein_nn, axis=0)
        p25 = np.percentile(mount_disc_rein_nn, q=25, axis=0)
        p75 = np.percentile(mount_disc_rein_nn, q=75, axis=0)
        ax.plot(med, label='NN - 84 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(mount_disc_rein_ten, axis=0)
        p25 = np.percentile(mount_disc_rein_ten, q=25, axis=0)
        p75 = np.percentile(mount_disc_rein_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 43 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
        ax.set_xlabel("Episodes - AC")
        ax.set_ylabel("Return - MountaincarDisc")
        ax.grid()
        
        #ax.legend()

        ###
        ax = axs[1, 1]
        
        med = np.median(mount_disc_ppo_rbf, axis=0)
        p25 = np.percentile(mount_disc_ppo_rbf, q=25, axis=0)
        p75 = np.percentile(mount_disc_ppo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 204 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_disc_ppo_nn, axis=0)
        p25 = np.percentile(mount_disc_ppo_nn, q=25, axis=0)
        p75 = np.percentile(mount_disc_ppo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 84 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        med = np.median(mount_disc_ppo_ten, axis=0)
        p25 = np.percentile(mount_disc_ppo_ten, q=25, axis=0)
        p75 = np.percentile(mount_disc_ppo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 43 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
        ax.set_xlabel("Episodes - PPO")
        ax.grid()
        
        #ax.legend()
        
        ###
        ax = axs[1, 2]
        
        med = np.median(mount_disc_trpo_rbf, axis=0)
        p25 = np.percentile(mount_disc_trpo_rbf, q=25, axis=0)
        p75 = np.percentile(mount_disc_trpo_rbf, q=75, axis=0)
        ax.plot(med, label='RBF - 204 pars.', alpha=0.8)
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(mount_disc_trpo_nn, axis=0)
        p25 = np.percentile(mount_disc_trpo_nn, q=25, axis=0)
        p75 = np.percentile(mount_disc_trpo_nn, q=75, axis=0)
        ax.plot(med, label='NN - 84 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)
        
        med = np.median(mount_disc_trpo_ten, axis=0)
        p25 = np.percentile(mount_disc_trpo_ten, q=25, axis=0)
        p75 = np.percentile(mount_disc_trpo_ten, q=75, axis=0)
        ax.plot(med, label='TLR - 43 pars.')
        ax.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        ax.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 10_000)
        ax.set_xticks([0, 5_000, 10_000])
        ax.set_xlabel("Episodes - TRPO")
        ax.grid()
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout()
        fig.savefig('figures/fig_3.jpg', dpi=300)
        plt.show()
