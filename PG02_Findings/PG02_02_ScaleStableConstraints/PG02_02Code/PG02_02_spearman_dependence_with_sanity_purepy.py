#!/usr/bin/env python3
"""
PG02_02_spearman_dependence_with_sanity_purepy.py

Lagged dependence test using Spearman rank correlation.
Lag k = 1 is explicitly excluded (trivial local anticorrelation).

Pure Python, PG02-grade:
- robust sanity check
- permutation null
- plot saving
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


# ======================================================
# Pure Python prime generation
# ======================================================

def primes_up_to(n: int) -> np.ndarray:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = b"\x00" * ((n - p * p) // p + 1)
    return np.fromiter((i for i, v in enumerate(sieve) if v), dtype=np.int64)


def prime_gaps(pmax: int) -> np.ndarray:
    return np.diff(primes_up_to(pmax))


# ======================================================
# Spearman machinery
# ======================================================

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def spearman_curve(gaps: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute Spearman rho for lags k = 2..max_lag.
    Lag-1 excluded by design.
    """
    rho = np.zeros(max_lag - 1)
    for k in range(2, max_lag + 1):
        rho[k - 2] = spearman_corr(gaps[:-k], gaps[k:])
    return rho


# ======================================================
# Sanity check (permutation vs permutation)
# ======================================================

def sanity_check(gaps, max_lag, seed, trials=20, ztol=2.5):
    rng = np.random.default_rng(seed)

    # Reference permutation
    g0 = gaps.copy()
    rng.shuffle(g0)
    r0 = spearman_curve(g0, max_lag)

    # Permutation ensemble
    curves = np.zeros((trials, max_lag - 1))
    for t in range(trials):
        g = gaps.copy()
        rng.shuffle(g)
        curves[t] = spearman_curve(g, max_lag)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1)
    std[std == 0] = 1e-12

    z = (r0 - mean) / std
    zmax = np.max(np.abs(z))

    print("=== SANITY CHECK ===")
    print(f"Max |z| vs permutation null: {zmax:.2f}")

    if zmax > ztol:
        raise RuntimeError("Sanity check FAILED")

    print("Sanity check PASSED.\n")


# ======================================================
# Plotting
# ======================================================

def plot_results(rho_prime, null_mean, null_std, out_prefix):
    k = np.arange(2, len(rho_prime) + 2)

    # rho vs lag
    plt.figure()
    plt.plot(k, rho_prime, label="Primes", linewidth=2)
    plt.fill_between(
        k,
        null_mean - null_std,
        null_mean + null_std,
        alpha=0.3,
        label="Null ± 1σ",
    )
    plt.xlabel("Lag k")
    plt.ylabel("Spearman ρ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rho_vs_lag.png", dpi=200)
    plt.close()

    # z-score plot
    z = (rho_prime - null_mean) / null_std
    plt.figure()
    plt.plot(k, z, linewidth=2)
    plt.axhline(2, linestyle="--", alpha=0.5)
    plt.axhline(-2, linestyle="--", alpha=0.5)
    plt.xlabel("Lag k")
    plt.ylabel("Z-score")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rho_zscore_vs_lag.png", dpi=200)
    plt.close()


# ======================================================
# Main
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, required=True)
    ap.add_argument("--max-lag", type=int, default=150)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-prefix", type=str, default="PG02_02_Spearman")
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} (pure Python sieve)...")
    gaps = prime_gaps(args.pmax)
    print(f"Number of gaps: {len(gaps)}")

    # ---- Mandatory sanity check ----
    sanity_check(
        gaps=gaps,
        max_lag=args.max_lag,
        seed=args.seed,
    )

    # ---- Real test ----
    rho_prime = spearman_curve(gaps, args.max_lag)

    rng = np.random.default_rng(args.seed)
    curves = np.zeros((args.trials, args.max_lag - 1))
    for t in range(args.trials):
        g = gaps.copy()
        rng.shuffle(g)
        curves[t] = spearman_curve(g, args.max_lag)

    null_mean = curves.mean(axis=0)
    null_std = curves.std(axis=0, ddof=1)
    null_std[null_std == 0] = 1e-12

    plot_results(rho_prime, null_mean, null_std, args.out_prefix)

    print("Test completed.")
    print("Saved plots:")
    print(f"  {args.out_prefix}_rho_vs_lag.png")
    print(f"  {args.out_prefix}_rho_zscore_vs_lag.png")


if __name__ == "__main__":
    main()
