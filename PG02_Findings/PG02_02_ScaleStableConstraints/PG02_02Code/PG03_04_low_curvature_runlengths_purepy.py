#!/usr/bin/env python3
"""
PG03_04_low_curvature_runlengths_purepy.py

Test whether low-curvature regions form unusually long contiguous runs.
Compares run-length statistics of |chi_n| <= threshold against permutation null.
"""

import argparse
import numpy as np


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
# Run-length analysis
# ======================================================

def run_lengths(binary):
    runs = []
    cur = 0
    for v in binary:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


# ======================================================
# Main
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, required=True)
    ap.add_argument("--quantile", type=float, default=0.30)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} ...")
    gaps = prime_gaps(args.pmax)
    chi = curvature_from_gaps(gaps)
    s = np.abs(chi)

    thresh = np.quantile(s, args.quantile)
    indicator = s <= thresh

    obs_runs = run_lengths(indicator)
    obs_max = max(obs_runs)
    obs_mean = np.mean(obs_runs)

    rng = np.random.default_rng(args.seed)

    null_max = []
    null_mean = []

    for _ in range(args.trials):
        perm = rng.permutation(indicator)
        runs = run_lengths(perm)
        null_max.append(max(runs))
        null_mean.append(np.mean(runs))

    null_max = np.array(null_max)
    null_mean = np.array(null_mean)

    print("\n=== LOW-CURVATURE RUN-LENGTH TEST ===")
    print(f"Quantile threshold: {args.quantile}")
    print(f"Number of points:   {len(indicator)}")
    print("")
    print("Observed:")
    print(f"  max run length  = {obs_max}")
    print(f"  mean run length = {obs_mean:.2f}")
    print("")
    print("Permutation null:")
    print(f"  max run length  = {null_max.mean():.2f} ± {null_max.std():.2f}")
    print(f"  mean run length = {null_mean.mean():.2f} ± {null_mean.std():.2f}")
    print("")
    print("Empirical p-values:")
    print(f"  P(null max ≥ obs max)  = {np.mean(null_max >= obs_max):.4f}")
    print(f"  P(null mean ≥ obs mean)= {np.mean(null_mean >= obs_mean):.4f}")


if __name__ == "__main__":
    main()
