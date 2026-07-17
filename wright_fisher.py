"""Wright-Fisher simulation of allele-frequency dynamics.

A minimal, dependency-light implementation of the classic Wright-Fisher model
with selection, mutation, and random genetic drift. Each generation the allele
frequency is updated deterministically by selection and mutation, then a finite
sample of ``N`` diploid-equivalent gene copies is drawn (binomial sampling),
which introduces stochastic genetic drift.

Model (per generation, allele A at frequency p):
    1. Selection:  p_sel = p(1+s) / [p(1+s) + (1-p)]
    2. Mutation:   p_mut = p_sel(1-mu) + (1-p_sel)mu
    3. Drift:      count ~ Binomial(N, p_mut);  p' = count / N

Author: Seoho Shawn
License: MIT
"""

from __future__ import annotations

import argparse

import numpy as np


def wright_fisher(
    N: int,
    s: float,
    mu: float,
    generations: int,
    p0: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one allele-frequency trajectory under the Wright-Fisher model.

    Parameters
    ----------
    N : int
        Population size (number of gene copies sampled each generation).
    s : float
        Selection coefficient for allele A (s > 0 favours A).
    mu : float
        Per-generation mutation rate (symmetric between the two alleles).
    generations : int
        Number of generations to simulate.
    p0 : float, optional
        Initial frequency of allele A (default 0.5).
    rng : numpy.random.Generator, optional
        Random generator, for reproducibility. A default is created if omitted.

    Returns
    -------
    numpy.ndarray
        Array of length ``generations + 1`` with the frequency of A at each
        generation, starting from generation 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = p0
    trajectory = [p]
    for _ in range(generations):
        # 1. Selection
        p_sel = p * (1 + s) / (p * (1 + s) + (1 - p))
        # 2. Mutation (symmetric)
        p_mut = p_sel * (1 - mu) + (1 - p_sel) * mu
        # 3. Drift: finite-population binomial sampling
        count = rng.binomial(N, p_mut)
        p = count / N
        trajectory.append(p)
    return np.array(trajectory)


def run_replicates(
    N: int,
    s: float,
    mu: float,
    generations: int,
    replicates: int,
    p0: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Run several independent trajectories and stack them.

    Returns
    -------
    numpy.ndarray
        Shape ``(replicates, generations + 1)``.
    """
    rng = np.random.default_rng(seed)
    return np.vstack(
        [wright_fisher(N, s, mu, generations, p0, rng) for _ in range(replicates)]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wright-Fisher allele-frequency simulator "
        "(selection + mutation + drift)."
    )
    parser.add_argument("-N", type=int, default=100, help="population size")
    parser.add_argument("-s", type=float, default=0.01, help="selection coefficient")
    parser.add_argument("--mu", type=float, default=1e-4, help="mutation rate")
    parser.add_argument(
        "-g", "--generations", type=int, default=500, help="number of generations"
    )
    parser.add_argument("--p0", type=float, default=0.5, help="initial frequency of A")
    parser.add_argument(
        "-r", "--replicates", type=int, default=1, help="number of trajectories"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "-o", "--output", default=None, help="save figure to this path instead of showing"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    import matplotlib.pyplot as plt

    trajectories = run_replicates(
        args.N, args.s, args.mu, args.generations, args.replicates, args.p0, args.seed
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for traj in trajectories:
        ax.plot(traj, alpha=0.7, linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Frequency of allele A")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Wright-Fisher: N={args.N}, s={args.s}, mu={args.mu}, "
        f"{args.replicates} replicate(s)"
    )
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
