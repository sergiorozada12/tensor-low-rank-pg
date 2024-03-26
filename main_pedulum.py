from src.experiments.pendulum_continuous run_exp_pendulum_cont
from src.experiments.pendulum_discrete run_exp_pendulum_disc
from src.experiments.mountaincar_continuous run_exp_mountaincar_cont
from src.experiments.mountaincar_discrete run_exp_mountaincar_disc
from src.experiments.wireless run_exp_wireless

from src.plots.control import plot_control
from src.plots.wireless import plot_wireless


if __name__ == "__main__":
    run_exp_pendulum_cont()
    run_exp_pendulum_disc()
    run_exp_mountaincar_cont()
    run_exp_mountaincar_disc()
    run_exp_wireless()

    plot_control()
    plot_wireless()
