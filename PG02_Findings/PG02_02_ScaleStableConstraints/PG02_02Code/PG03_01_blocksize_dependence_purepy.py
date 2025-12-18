#!/usr/bin/env python3
"""
PG03_01_blocksize_dependence_purepy.py

Study how ordering coherence depends on block size B
using Spearman gap ordering vs block-permutation nulls.

Outputs:
- CSV of coherence metrics vs block size
"""

import argparse
import numpy as np
from scipy.stats import rankdata


# ======================================================
# Prime generation (pure Python)
# ======================================================

def primes_up_to(n: int) -> np.ndarray:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = b"\x00" * ((n - p * p) // p + 1)
    return np.fromiter((i for i, v in enumerate(sieve) if v), dtype=np.int64)


def prime_gaps(pmax: int) -> np.ndarray:
    return np.diff(primes_up_to(pmax)).astype(np.float64)


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


def spearman_curve(gaps, max_lag):
    rho = np.zeros(max_lag - 1)
    for k in range(2, max_lag + 1):
        rho[k - 2] = spearman_corr(gaps[:-k], gaps[k:])
    return rho


# ======================================================
# Block permutation
# ======================================================

def block_permute(gaps, block, rng):
    n = len(gaps)
    m = n // block
    blocks = gaps[:m * block].reshape(m, block)
    out = blocks[rng.permutation(m)].reshape(-1)
    if m * block < n:
        out = np.concatenate([out, gaps[m * block :]])
    return out


# ======================================================
# Coherence metrics
# ======================================================

def longest_run(mask):
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def coherence_metrics(z, threshold=2.0):
    sig = z > threshold
    frac = float(np.mean(sig))
    run = int(longest_run(sig))
    area = float(np.sum(np.maximum(z - threshold, 0.0)))
    k_last = int(np.where(sig)[0][-1] + 2) if np.any(sig) else None
    return frac, run, area, k_last


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

    block_sizes = [4, 8, 16, 32, 64, 128, 256]

    print(f"Generating primes ≤ {args.pmax} ...")
    gaps = prime_gaps(args.pmax)
    print(f"Number of gaps: {len(gaps)}")

    rho_prime = spearman_curve(gaps, args.max_lag)
    rng = np.random.default_rng(args.seed)

    print("\n=== BLOCK SIZE DEPENDENCE ===")
    print("B | frac_sig | run_max | area_pos | k_last")
    print("------------------------------------------")

    results = []

    for B in block_sizes:
        null_curves = np.zeros((args.trials, args.max_lag - 1))
        for t in range(args.trials):
            g = block_permute(gaps, B, rng)
            null_curves[t] = spearman_curve(g, args.max_lag)

        mean = null_curves.mean(axis=0)
        std = null_curves.std(axis=0, ddof=1)
        std[std == 0] = 1e-12

        z = (rho_prime - mean) / std
        frac, run, area, k_last = coherence_metrics(z)

        results.append((B, frac, run, area, k_last))
        print(f"{B:3d} | {frac:8.3f} | {run:7d} | {area:8.2f} | {k_last}")

    # Save CSV
    out = "PG03_01_blocksize_dependence.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("block,frac_sig,run_max,area_pos,k_last\n")
        for r in results:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
