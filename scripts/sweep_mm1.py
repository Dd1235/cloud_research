import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim.mm1 import simulate_with_workers

import argparse

MU = 1.0
RHOS = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95]) # utilization levels, lam = rho*mu
SEEDS = range(5)


def main(n_workers: int = 1):
    med, lo, hi = [], [], []

    for rho in RHOS:
        means = np.array([simulate_with_workers(rho*MU,MU,n_workers=n_workers, seed=seed)[0] for seed in SEEDS])

        med.append(np.median(means))
        lo.append(means.min())
        hi.append(means.max())

        if n_workers == 1:
            analytic = 1 / (MU - rho * MU)
            print(
                f"rho={rho:.2f} "
                f"median={med[-1]:.2f} "
                f"range=[{lo[-1]:.2f}, {hi[-1]:.2f}] "
                f"analytic={analytic:.2f}"
            )
        else:
            print(
                f"offered_load={rho:.2f} "
                f"median={med[-1]:.2f} "
                f"range=[{lo[-1]:.2f}, {hi[-1]:.2f}]"
            )

    x = np.linspace(0.05, 0.97, 200)

    plt.figure(figsize=(6, 4))

    if n_workers == 1:
        plt.plot(
            x,
            1 / (MU - x * MU),
            "k--",
            label="analytic M/M/1: 1/(μ−λ)",
        )

    plt.errorbar(
        RHOS,
        med,
        yerr=[np.subtract(med, lo), np.subtract(hi, med)],
        fmt="o-",
        capsize=3,
        label="sim (5 seeds, median ± range)",
    )

    plt.xlabel("utilization ρ = λ/μ")
    plt.ylabel("mean sojourn time")
    plt.title(f"M/M/{n_workers}: the hockey stick")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("out", exist_ok=True)
    plt.savefig(f"out/mm{n_workers}_sweep.png", dpi=120)
    print(f"wrote out/mm{n_workers}_sweep.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep M/M/n utilization with round-robin routing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of identical FIFO workers (default: 1)",
    )
    args = parser.parse_args()

    main(n_workers=args.workers)