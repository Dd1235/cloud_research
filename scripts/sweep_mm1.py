import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim.mm1 import simulate_with_workers

import argparse

C_PREFILL = 1e-3  # 1 ms per prompt token
C_DECODE = 1e-2   # 10 ms per output token

PROMPT_TOKENS = 500
OUTPUT_TOKENS = 100

SERVICE_TIME = (
    C_PREFILL * PROMPT_TOKENS
    + C_DECODE * OUTPUT_TOKENS
)

RHOS = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95]) # target per-worker utilization
SEEDS = range(5)


def main(n_workers: int = 1):
    med, lo, hi = [], [], []

    for rho in RHOS:

        rate = rho * n_workers / SERVICE_TIME

        means = np.array([
            simulate_with_workers(
                rate=rate,
                c_prefill=C_PREFILL,
                c_decode=C_DECODE,
                prompt_tokens=PROMPT_TOKENS,
                output_tokens=OUTPUT_TOKENS,
                n_workers=n_workers,
                seed=seed,
            )[0]
            for seed in SEEDS
        ])

        med.append(np.median(means))
        lo.append(means.min())
        hi.append(means.max())

        print(
            f"rho={rho:.2f} "
            f"rate={rate:.3f}/s "
            f"median_ttft={med[-1]:.3f} "
            f"range=[{lo[-1]:.3f}, {hi[-1]:.3f}]"
        )


    plt.figure(figsize=(6, 4))

    plt.errorbar(
        RHOS,
        med,
        yerr=[np.subtract(med, lo), np.subtract(hi, med)],
        fmt="o-",
        capsize=3,
        label="simulation (5 seeds, median ± range)",
    )

    plt.xlabel("target per-worker utilization ρ")
    plt.ylabel("mean TTFT (simulated time)")
    plt.title(f"Sequential LLM serving: {n_workers} worker(s)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("out", exist_ok=True)
    plt.savefig(f"out/mockreq_mm{n_workers}_sweep.png", dpi=120)
    print(f"wrote out/mockreq_mm{n_workers}_sweep.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep sequential LLM TTFT under round-robin routing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of identical FIFO workers (default: 1)",
    )
    args = parser.parse_args()

    main(n_workers=args.workers)