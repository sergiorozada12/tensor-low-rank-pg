import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def load_arrays(path, names):
    base = Path(path)
    return [np.load(str(base / f"{stem}.npy")) for stem in names]


def plot_panel(ax, ranks, nfe, nme, envname, variant):
    ax.plot(ranks, nfe, marker="o", linestyle="-", label="NFE")
    ax.plot(ranks, nme, marker="s", linestyle="--", label="NME")
    ax.set_xlim(1, 15)
    ax.set_xticks([1, 8, 15])
    ax.set_ylim(0, 50)
    ax.set_xlabel("Rank")
    ax.set_ylabel(
        f"$\\mathrm{{Err.}}$ (\\%) - {envname}{variant}"
    )  # <-- keep commented
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, linestyle=":")


def plot_ranks():
    data_dir = "results/plot_ranks"

    # --- Pendulum
    (ranks_cont_pend,) = load_arrays(data_dir, ["ranks_pendulum_cont"])
    (nfe_cont_pend,) = load_arrays(data_dir, ["errors_nfe_pendulum_cont"])
    (nme_cont_pend,) = load_arrays(data_dir, ["errors_max_pendulum_cont"])
    (ranks_disc_pend,) = load_arrays(data_dir, ["ranks_pendulum_disc"])
    (nfe_disc_pend,) = load_arrays(data_dir, ["errors_nfe_pendulum_disc"])
    (nme_disc_pend,) = load_arrays(data_dir, ["errors_max_pendulum_disc"])

    # --- MountainCar
    (ranks_cont_mount,) = load_arrays(data_dir, ["ranks_mountaincar_cont"])
    (nfe_cont_mount,) = load_arrays(data_dir, ["errors_nfe_mountaincar_cont"])
    (nme_cont_mount,) = load_arrays(data_dir, ["errors_max_mountaincar_cont"])
    (ranks_disc_mount,) = load_arrays(data_dir, ["ranks_mountaincar_disc"])
    (nfe_disc_mount,) = load_arrays(data_dir, ["errors_nfe_mountaincar_disc"])
    (nme_disc_mount,) = load_arrays(data_dir, ["errors_max_mountaincar_disc"])

    # --- CartPole
    (ranks_cont_cart,) = load_arrays(data_dir, ["ranks_cartpole_cont"])
    (nfe_cont_cart,) = load_arrays(data_dir, ["errors_nfe_cartpole_cont"])
    (nme_cont_cart,) = load_arrays(data_dir, ["errors_max_cartpole_cont"])
    (ranks_disc_cart,) = load_arrays(data_dir, ["ranks_cartpole_disc"])
    (nfe_disc_cart,) = load_arrays(data_dir, ["errors_nfe_cartpole_disc"])
    (nme_disc_cart,) = load_arrays(data_dir, ["errors_max_cartpole_disc"])

    # --- Goddard
    (ranks_cont_goddard,) = load_arrays(data_dir, ["ranks_goddard_cont"])
    (nfe_cont_goddard,) = load_arrays(data_dir, ["errors_nfe_goddard_cont"])
    (nme_cont_goddard,) = load_arrays(data_dir, ["errors_max_goddard_cont"])
    (ranks_disc_goddard,) = load_arrays(data_dir, ["ranks_goddard_disc"])
    (nfe_disc_goddard,) = load_arrays(data_dir, ["errors_nfe_goddard_disc"])
    (nme_disc_goddard,) = load_arrays(data_dir, ["errors_max_goddard_disc"])

    with plt.style.context(["science"], ["ieee"]):
        matplotlib.rcParams.update({"font.size": 20})
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))

        configs = [
            (
                ranks_cont_pend,
                nfe_cont_pend,
                nme_cont_pend,
                axes[0, 0],
                "Pendulum",
                "Cont",
            ),
            (
                ranks_disc_pend,
                nfe_disc_pend,
                nme_disc_pend,
                axes[1, 0],
                "Pendulum",
                "Disc",
            ),
            (
                ranks_cont_mount,
                nfe_cont_mount,
                nme_cont_mount,
                axes[0, 1],
                "MountainCar",
                "Cont",
            ),
            (
                ranks_disc_mount,
                nfe_disc_mount,
                nme_disc_mount,
                axes[1, 1],
                "MountainCar",
                "Disc",
            ),
            (
                ranks_cont_cart,
                nfe_cont_cart,
                nme_cont_cart,
                axes[0, 2],
                "CartPole",
                "Cont",
            ),
            (
                ranks_disc_cart,
                nfe_disc_cart,
                nme_disc_cart,
                axes[1, 2],
                "CartPole",
                "Disc",
            ),
            (
                ranks_cont_goddard,
                nfe_cont_goddard,
                nme_cont_goddard,
                axes[0, 3],
                "Goddard",
                "Cont",
            ),
            (
                ranks_disc_goddard,
                nfe_disc_goddard,
                nme_disc_goddard,
                axes[1, 3],
                "Goddard",
                "Disc",
            ),
        ]

        for ranks, nfe, nme, ax, env, var in configs:
            plot_panel(ax, ranks, nfe, nme, env, var)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig("figures/fig_1.jpg", dpi=300)
