from src.experiments.ranks_exploration import run_ranks_exploration
from src.experiments.pendulum_continuous import run_pendulum_cont_experiments
from src.experiments.pendulum_discrete import run_pendulum_disc_experiments
from src.experiments.mountaincar_continuous import run_mountaincar_cont_experiments
from src.experiments.mountaincar_discrete import run_mountaincar_disc_experiments
from src.experiments.wireless import run_wireless_experiments

from src.plots.ranks import plot_ranks
from src.plots.control import plot_control
from src.plots.wireless import plot_wireless


N_EXP = 100
N_PROC = 50


if __name__ == "__main__":
    run_ranks_exploration()
    plot_ranks()

    run_pendulum_cont_experiments(N_EXP, N_PROC)
    run_pendulum_disc_experiments(N_EXP, N_PROC)
    run_mountaincar_cont_experiments(N_EXP, N_PROC)
    run_mountaincar_disc_experiments(N_EXP, N_PROC)
    plot_control()

    run_wireless_experiments(N_EXP, N_PROC)
    plot_wireless()
