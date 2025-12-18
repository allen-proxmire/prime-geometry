#!/usr/bin/env python3
"""
PG02_02_mutual_information_with_sanity_purepy.py

PG02 Mutual Information test with:
- Pure Python prime generation
- Correct permutation-based sanity check
- Automatic plot saving

NO external dependencies beyond numpy + matplotlib.
"""

import argparse
import math
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


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
    primes = primes_up_to(pmax)
    return np.diff(primes)


# ======================================================
# Mutual Information machinery
# ======================================================

def quantile_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        edges = np.linspace(x.min(), x.max(), bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def discretize(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, edges[1:-1]).astype(np.int32)


def mutual_information_discrete(xb, yb, bins: int) -> float:
    n = xb.size
    joint = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(joint, (xb, yb), 1)

    px = joint.sum(axis=1) / n
    py = joint.sum(axis=0) / n
    pxy = joint / n

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return mi


def mi_curve(gaps: np.ndarray, max_lag: int, bins: int, edges: np.ndarray) -> np.ndarray:
    xb = discretize(gaps, edges)
    mi = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        mi[k - 1] = mutual_information_discrete(xb[:-k], xb[k:], bins)
    return mi


# ======================================================
# Correct sanity check (permutation vs permutation)
# ======================================================

def sanity_check(
    gaps: np.ndarray,
    max_lag: int,
    bins: int,
    edges: np.ndarray,
    seed: int,
    trials: int = 20,
    ztol: float = 2.5,
):
    rng = np.random.default_rng(seed)

    # Reference permutation
    g0 = gaps.copy()
    rng.shuffle(g0)
    mi0 = mi_curve(g0, max_lag, bins, edges)

    # Permutation ensemble
    curves = np.zeros((trials, max_lag))
    for t in range(trials):
        g = gaps.copy()
        rng.shuffle(g)
        curves[t] = mi_curve(g, max_lag, bins, edges)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1)
    std[std == 0] = 1e-12

    z = (mi0 - mean) / std
    zmax = np.max(np.abs(z))

    print("=== SANITY CHECK ===")
    print(f"Max |z| vs permutation null: {zmax:.2f}")

    if zmax > ztol:
        raise RuntimeError(
            "Sanity check FAILED: permutation distinguishable from permutation null."
        )

    print("Sanity check PASSED.\n")


# ======================================================
# Plotting
# ======================================================

def plot_results(mi_prime, null_mean, null_std, out_prefix):
    k = np.arange(1, len(mi_prime) + 1)

    plt.figure()
    plt.plot(k, mi_prime, label="Primes", linewidth=2)
    plt.fill_between(k, null_mean - null_std, null_mean + null_std, alpha=0.3)
    plt.xlabel("Lag k")
    plt.ylabel("Mutual Information (nats)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_MI_vs_lag.png", dpi=200)
    plt.close()

    z = (mi_prime - null_mean) / null_std
    plt.figure()
    plt.plot(k, z, linewidth=2)
    plt.axhline(2, linestyle="--", alpha=0.5)
    plt.axhline(-2, linestyle="--", alpha=0.5)
    plt.xlabel("Lag k")
    plt.ylabel("Z-score")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_MI_zscore_vs_lag.png", dpi=200)
    plt.close()


# ======================================================
# Main
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, required=True)
    ap.add_argument("--max-lag", type=int, default=150)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-prefix", type=str, default="PG02_02_MI")
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} (pure Python sieve)...")
    gaps = prime_gaps(args.pmax)
    print(f"Number of gaps: {len(gaps)}")

    edges = quantile_bin_edges(gaps, args.bins)

    # ---- Mandatory sanity check ----
    sanity_check(
        gaps=gaps,
        max_lag=args.max_lag,
        bins=args.bins,
        edges=edges,
        seed=args.seed,
    )

    # ---- Real test ----
    mi_prime = mi_curve(gaps, args.max_lag, args.bins, edges)

    rng = np.random.default_rng(args.seed)
    null_curves = np.zeros((args.trials, args.max_lag))
    for t in range(args.trials):
        g = gaps.copy()
        rng.shuffle(g)
        null_curves[t] = mi_curve(g, args.max_lag, args.bins, edges)

    null_mean = null_curves.mean(axis=0)
    null_std = null_curves.std(axis=0, ddof=1)
    null_std[null_std == 0] = 1e-12

    plot_results(mi_prime, null_mean, null_std, args.out_prefix)

    print("Test completed.")
    print("Saved plots:")
    print(f"  {args.out_prefix}_MI_vs_lag.png")
    print(f"  {args.out_prefix}_MI_zscore_vs_lag.png")


if __name__ == "__main__":
    main()
