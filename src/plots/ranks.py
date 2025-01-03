import numpy as  np
import matplotlib
import matplotlib.pyplot as plt


def plot_ranks():
    ranks_cont = np.load("results/ranks_pend_cont.npy")
    errors_cont = np.load("results/errors_pend_cont.npy")

    ranks_disc = np.load("results/ranks_pend_disc.npy")
    errors_disc = np.load("results/errors_pend_disc.npy")

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 16})
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[5, 8])
        axes = axes.flatten()
        
        axes[0].plot(ranks_cont, errors_cont, marker='o')
        axes[0].set_xlim(1, 10)
        axes[0].set_ylim(0, 100)
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("$\mathrm{NFE}$ (\%) - PendulumCont")
        axes[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[0].set_xticks([1, 5, 10])
        axes[0].grid()
        
        axes[1].plot(ranks_disc, errors_disc, marker='o')
        axes[1].set_xlim(1, 25)
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("$\mathrm{NFE}$ (\%) - PendulumDisc")
        axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[1].set_xticks([1, 15, 30])
        axes[1].grid()
        
        plt.tight_layout()
        fig.savefig('figures/fig_1.jpg', dpi=300)
