#!/usr/bin/env python3
"""
PG03_03_state_conditioned_ordering_purepy.py

Test whether gap ordering coherence depends on local curvature magnitude.
Splits indices by |chi_n| quantiles and measures Spearman ordering coherence
within each subset vs a permutation null.
"""

import argparse
import numpy as np
from scipy.stats import rankdata


# ======================================================
# Prime generation (pure Python)
# ======================================================

def primes_up_to(n):
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = b"\x00" * ((n - p * p) // p + 1)
    return np.fromiter((i for i, v in enumerate(sieve) if v), dtype=np.int64)


def prime_gaps(pmax):
    return np.diff(primes_up_to(pmax)).astype(np.float64)


# ======================================================
# Curvature
# ======================================================

def curvature_from_gaps(g):
    denom = g[:-2] + g[1:-1]
    denom = np.where(denom == 0, 1e-12, denom)
    return (g[2:] - g[:-2]) / denom


# ======================================================
# Spearman machinery
# ======================================================

def spearman_corr(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return 0.0 if denom == 0 else float(np.sum(rx * ry) / denom)


def spearman_curve(series, max_lag):
    rho = np.zeros(max_lag - 1)
    for k in range(2, max_lag + 1):
        rho[k - 2] = spearman_corr(series[:-k], series[k:])
    return rho


# ======================================================
# Main
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, required=True)
    ap.add_argument("--max-lag", type=int, default=150)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} ...")
    gaps = prime_gaps(args.pmax)
    chi = curvature_from_gaps(gaps)
    s = np.abs(chi)

    # Align gaps with curvature index
    gaps_c = gaps[1:-1]

    # Quantile thresholds
    q_low, q_high = np.quantile(s, [0.3, 0.7])

    bins = {
        "LOW |chi|": s <= q_low,
        "MID |chi|": (s > q_low) & (s < q_high),
        "HIGH |chi|": s >= q_high,
    }

    rng = np.random.default_rng(args.seed)

    print("\n=== STATE-CONDITIONED ORDERING COHERENCE ===")
    print("Bin        | frac_sig | area_pos")
    print("----------------------------------")

    for label, mask in bins.items():
        series = gaps_c[mask]

        if len(series) < args.max_lag + 10:
            print(f"{label:10s} | insufficient data")
            continue

        rho = spearman_curve(series, args.max_lag)

        null = np.zeros((args.trials, args.max_lag - 1))
        for t in range(args.trials):
            perm = rng.permutation(series)
            null[t] = spearman_curve(perm, args.max_lag)

        mean = null.mean(axis=0)
        std = null.std(axis=0, ddof=1)
        std[std == 0] = 1e-12

        z = (rho - mean) / std

        frac_sig = float(np.mean(z > 2.0))
        area_pos = float(np.sum(np.maximum(z - 2.0, 0.0)))

        print(f"{label:10s} |   {frac_sig:6.3f} |  {area_pos:7.2f}")


if __name__ == "__main__":
    main()
