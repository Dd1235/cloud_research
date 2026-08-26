import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mm1 import simulate

MU = 1.0
RHOS = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95]) # utilization levels, lam = rho*mu
SEEDS = range(5)


def main():
    med, lo, hi = [], [], []

    for rho in RHOS:
        means = np.array([simulate(rho*MU,MU,seed=seed)[0] for seed in SEEDS])

        med.append(np.median(means))
        lo.append(means.min())
        hi.append(means.max())

        analytic = 1 / (MU - rho * MU)
        print(
            f"rho={rho:.2f} "
            f"median={med[-1]:.2f} "
            f"range=[{lo[-1]:.2f}, {hi[-1]:.2f}] "
            f"analytic={analytic:.2f}"
        )

    x = np.linspace(0.05, 0.97, 200)

    plt.figure(figsize=(6, 4))
    plt.plot(
        x,
        1 / (MU - x * MU),
        "k--",
        label="analytic 1/(μ−λ)",
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
    plt.title("M/M/1: the hockey stick")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("out", exist_ok=True)
    plt.savefig("out/mm1_sweep.png", dpi=120)
    print("wrote out/mm1_sweep.png")

if __name__ == "__main__":
    main()