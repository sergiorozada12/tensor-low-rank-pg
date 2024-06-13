import numpy as  np
import matplotlib
import matplotlib.pyplot as plt

from src.utils import OOMFormatter


def plot_wireless():
    wireless_ten = np.load("results/wireless_ten.npy")
    wireless_rbf = np.load("results/wireless_rbf.npy")
    wireless_nn = np.load("results/wireless_nn_small.npy")
    wireless_nn_la = np.load("results/wireless_nn_big.npy")

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 16})
        fig = plt.figure(figsize=(8, 5))  # Adjust figsize as needed

        med = np.median(wireless_rbf, axis=0)
        p25 = np.percentile(wireless_rbf, q=25, axis=0)
        p75 = np.percentile(wireless_rbf, q=75, axis=0)
        plt.plot(med, label='RBF-PPO - 1,202 params.')
        plt.fill_between(range(p25.shape[0]), p25, p75, alpha=.2) 

        med = np.median(wireless_nn, axis=0)
        p25 = np.percentile(wireless_nn, q=25, axis=0)
        p75 = np.percentile(wireless_nn, q=75, axis=0)
        plt.plot(med, label='NN-PPO - 1,093 params.')
        plt.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(wireless_nn_la, axis=0)
        p25 = np.percentile(wireless_nn_la, q=25, axis=0)
        p75 = np.percentile(wireless_nn_la, q=75, axis=0)
        plt.plot(med, label='NN-PPO - 2,181 params.')
        plt.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        med = np.median(wireless_ten, axis=0)
        p25 = np.percentile(wireless_ten, q=25, axis=0)
        p75 = np.percentile(wireless_ten, q=75, axis=0)
        plt.plot(med, label='PTLRPO - 902 params.')
        plt.fill_between(range(p25.shape[0]), p25, p75, alpha=.2)

        plt.gca().yaxis.set_major_formatter(OOMFormatter(3, "%1.2f"))
        plt.gca().ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        plt.xticks([0, 2500, 5000])
        plt.xlim(0, 5_000)
        plt.ylabel("Return")
        plt.xlabel("Episodes")
        plt.grid()
        
        plt.legend()
        
        plt.tight_layout()
        fig.savefig('figures/fig_4.jpg', dpi=300)
