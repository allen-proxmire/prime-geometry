#!/usr/bin/env python3
"""
PG02_02_spearman_dependence_blocknull_purepy.py

Spearman lag-dependence test for prime gaps:
- lag k >= 2 only
- permutation null
- block-permutation null
- mandatory sanity check
- plot saving

PG02-grade, pure Python (numpy + matplotlib + scipy).
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
    Spearman rho for lags k = 2..max_lag (lag-1 excluded).
    """
    rho = np.zeros(max_lag - 1)
    for k in range(2, max_lag + 1):
        rho[k - 2] = spearman_corr(gaps[:-k], gaps[k:])
    return rho


# ======================================================
# Null models
# ======================================================

def permute_gaps(gaps: np.ndarray, rng) -> np.ndarray:
    g = gaps.copy()
    rng.shuffle(g)
    return g


def block_permute_gaps(gaps: np.ndarray, block: int, rng) -> np.ndarray:
    n = len(gaps)
    m = n // block
    blocks = gaps[: m * block].reshape(m, block)
    order = rng.permutation(m)
    out = blocks[order].reshape(-1)
    if m * block < n:
        out = np.concatenate([out, gaps[m * block :]])
    return out


# ======================================================
# Sanity check
# ======================================================

def sanity_check(gaps, max_lag, seed, trials=20, ztol=2.5):
    rng = np.random.default_rng(seed)

    g0 = permute_gaps(gaps, rng)
    r0 = spearman_curve(g0, max_lag)

    curves = np.zeros((trials, max_lag - 1))
    for t in range(trials):
        g = permute_gaps(gaps, rng)
        curves[t] = spearman_curve(g, max_lag)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1)
    std[std == 0] = 1e-12

    zmax = np.max(np.abs((r0 - mean) / std))

    print("=== SANITY CHECK ===")
    print(f"Max |z| vs permutation null: {zmax:.2f}")

    if zmax > ztol:
        raise RuntimeError("Sanity check FAILED")

    print("Sanity check PASSED.\n")


# ======================================================
# Plotting
# ======================================================

def plot_results(
    rho_prime,
    perm_mean,
    perm_std,
    block_mean,
    block_std,
    out_prefix,
):
    k = np.arange(2, len(rho_prime) + 2)

    # rho vs lag
    plt.figure()
    plt.plot(k, rho_prime, label="Primes", linewidth=2)
    plt.fill_between(
        k,
        perm_mean - perm_std,
        perm_mean + perm_std,
        alpha=0.25,
        label="Permutation null ±1σ",
    )
    plt.fill_between(
        k,
        block_mean - block_std,
        block_mean + block_std,
        alpha=0.35,
        label="Block-permutation null ±1σ",
    )
    plt.xlabel("Lag k")
    plt.ylabel("Spearman ρ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rho_vs_lag.png", dpi=200)
    plt.close()

    # z-score vs block null
    z = (rho_prime - block_mean) / block_std
    plt.figure()
    plt.plot(k, z, linewidth=2)
    plt.axhline(2, linestyle="--", alpha=0.5)
    plt.axhline(-2, linestyle="--", alpha=0.5)
    plt.xlabel("Lag k")
    plt.ylabel("Z-score (vs block null)")
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
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-prefix", type=str, default="PG02_02_Spearman")
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} ...")
    gaps = prime_gaps(args.pmax)
    print(f"Number of gaps: {len(gaps)}")

    # ---- Sanity check ----
    sanity_check(gaps, args.max_lag, args.seed)

    # ---- Prime curve ----
    rho_prime = spearman_curve(gaps, args.max_lag)

    rng = np.random.default_rng(args.seed)

    # ---- Permutation null ----
    perm_curves = np.zeros((args.trials, args.max_lag - 1))
    for t in range(args.trials):
        g = permute_gaps(gaps, rng)
        perm_curves[t] = spearman_curve(g, args.max_lag)

    perm_mean = perm_curves.mean(axis=0)
    perm_std = perm_curves.std(axis=0, ddof=1)
    perm_std[perm_std == 0] = 1e-12

    # ---- Block-permutation null ----
    block_curves = np.zeros((args.trials, args.max_lag - 1))
    for t in range(args.trials):
        g = block_permute_gaps(gaps, args.block, rng)
        block_curves[t] = spearman_curve(g, args.max_lag)

    block_mean = block_curves.mean(axis=0)
    block_std = block_curves.std(axis=0, ddof=1)
    block_std[block_std == 0] = 1e-12

    plot_results(
        rho_prime,
        perm_mean,
        perm_std,
        block_mean,
        block_std,
        args.out_prefix,
    )

    print("Test completed.")
    print("Saved plots:")
    print(f"  {args.out_prefix}_rho_vs_lag.png")
    print(f"  {args.out_prefix}_rho_zscore_vs_lag.png")


if __name__ == "__main__":
    main()
