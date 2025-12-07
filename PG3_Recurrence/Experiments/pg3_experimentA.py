import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

try:
    import primesieve
    HAVE_PRIMESIEVE = True
except ImportError:
    HAVE_PRIMESIEVE = False
    from sympy import primerange


# ---------- Core utilities ----------

def generate_primes(n: int) -> List[int]:
    """
    Return the first n primes.
    Uses primesieve if available; falls back to sympy otherwise.
    """
    if HAVE_PRIMESIEVE:
        return list(primesieve.generate_n_primes(n))
    else:
        # crude but OK for moderate n
        # we overshoot an upper bound and then slice
        upper = int(n * (math.log(max(n, 2)) + math.log(math.log(max(n, 3))))) + 10
        return list(primerange(2, upper))[:n]


def gaps_from_primes(primes: List[int]) -> List[int]:
    return [primes[i+1] - primes[i] for i in range(len(primes) - 1)]


def chis_from_gaps(gaps: List[int]) -> np.ndarray:
    """
    Compute chi_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
    for n = 0 .. len(gaps)-3.
    """
    g = np.array(gaps, dtype=float)
    g_n   = g[:-2]
    g_np1 = g[1:-1]
    g_np2 = g[2:]

    denom = g_n + g_np1
    # avoid division by zero (should not happen for prime gaps, but be safe)
    mask = denom != 0
    chi = np.zeros_like(g_n, dtype=float)
    chi[mask] = (g_np2[mask] - g_n[mask]) / denom[mask]
    return chi


def action_from_chis(chi: np.ndarray) -> float:
    """
    S = sum chi_n^2
    """
    return float(np.sum(chi * chi))


@dataclass
class ActionExperimentResult:
    N: int
    num_perms: int
    S_true: float
    S_perm_mean: float
    S_perm_std: float
    percentile_true: float


# ---------- Experiment A: Action scaling ----------

def experiment_A_action_scaling(
    N: int,
    num_perms: int = 200,
    rng_seed: int = 12345
) -> ActionExperimentResult:
    """
    Experiment A: For first N primes, compute S_true and compare to
    num_perms random permutations of the same gaps.

    Returns summary statistics as an ActionExperimentResult.
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # We need N+2 primes to have gaps up to g_N+1 and chi up to index N-1.
    primes = generate_primes(N + 2)
    gaps = gaps_from_primes(primes)

    # restrict to first N gaps (or N-1, but we'll be consistent)
    gaps = gaps[:N]

    chi_true = chis_from_gaps(gaps)
    S_true = action_from_chis(chi_true)

    S_perm_values = []
    gaps_array = np.array(gaps, dtype=int)

    for k in range(num_perms):
        perm = np.random.permutation(gaps_array)
        chi_perm = chis_from_gaps(perm.tolist())
        S_perm = action_from_chis(chi_perm)
        S_perm_values.append(S_perm)

    S_perm_arr = np.array(S_perm_values, dtype=float)
    S_perm_mean = float(S_perm_arr.mean())
    S_perm_std = float(S_perm_arr.std(ddof=1))

    # percentile of S_true among S_perm_values
    # (fraction of permutations with S <= S_true)
    percentile_true = float(np.mean(S_perm_arr <= S_true) * 100.0)

    return ActionExperimentResult(
        N=N,
        num_perms=num_perms,
        S_true=S_true,
        S_perm_mean=S_perm_mean,
        S_perm_std=S_perm_std,
        percentile_true=percentile_true,
    )


def run_experiment_A_batch(
    Ns: List[int],
    num_perms: int = 200,
    rng_seed: int = 12345,
    csv_path: str = "PG3_action_scaling_results.csv"
):
    """
    Run Experiment A for several N values and save results to CSV.
    """
    import csv

    results: List[ActionExperimentResult] = []
    for N in Ns:
        print(f"Running Experiment A for N = {N} (num_perms = {num_perms})...")
        res = experiment_A_action_scaling(N, num_perms=num_perms, rng_seed=rng_seed)
        print(
            f"  N={res.N}, S_true={res.S_true:.4e}, "
            f"S_perm_mean={res.S_perm_mean:.4e}, "
            f"S_perm_std={res.S_perm_std:.4e}, "
            f"percentile={res.percentile_true:.2f}%"
        )
        results.append(res)
        rng_seed += 1  # change seed for each N

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "num_perms",
            "S_true",
            "S_perm_mean",
            "S_perm_std",
            "percentile_true"
        ])
        for r in results:
            writer.writerow([
                r.N,
                r.num_perms,
                f"{r.S_true:.10e}",
                f"{r.S_perm_mean:.10e}",
                f"{r.S_perm_std:.10e}",
                f"{r.percentile_true:.4f}",
            ])

    print(f"\nSaved Experiment A results to {csv_path}")


# ---------- Main CLI entry ----------

if __name__ == "__main__":
    # Example: run action scaling for a few N values.
    Ns = [50_000, 100_000, 250_000, 500_000, 1_000_000]  # you can add 250000, 500000 later
    run_experiment_A_batch(Ns, num_perms=200)
