#!/usr/bin/env python3
"""
PG02_02_spearman_blocknull_coherence_purepy.py

Spearman lag-dependence (k >= 2) for prime gaps with:
- pure-Python prime generation
- permutation sanity check (quantile-based, multiple-comparison safe)
- block-permutation null
- coherence metrics
- plot + CSV output

PG02-grade empirical experiment.
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
# Sanity check (quantile-based)
# ======================================================

def sanity_check(gaps, max_lag, seed, trials=20):
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

    z = (r0 - mean) / std
    z_abs = np.abs(z)

    z_mean = float(np.mean(z_abs))
    z_q99 = float(np.quantile(z_abs, 0.99))

    print("=== SANITY CHECK ===")
    print(f"Mean |z| vs permutation null: {z_mean:.2f}")
    print(f"99th pct |z| vs permutation null: {z_q99:.2f}")

    if z_q99 > 4.0:
        raise RuntimeError("Sanity check FAILED")

    print("Sanity check PASSED.\n")


# ======================================================
# Coherence metrics
# ======================================================

def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def coherence_metrics(z: np.ndarray, threshold: float):
    sig = z > threshold
    frac_sig = float(np.mean(sig))
    run_max = int(longest_true_run(sig))
    area_pos = float(np.sum(np.maximum(z - threshold, 0.0)))

    if np.any(sig):
        idx_last = int(np.where(sig)[0][-1])
        k_last = idx_last + 2
    else:
        k_last = None

    return k_last, frac_sig, run_max, area_pos


# ======================================================
# Plotting + CSV
# ======================================================

def plot_results(rho_prime, block_mean, block_std, out_prefix):
    k = np.arange(2, len(rho_prime) + 2)

    plt.figure()
    plt.plot(k, rho_prime, label="Primes", linewidth=2)
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

    return z


def write_csv(path, rho_prime, block_mean, block_std, z):
    k = np.arange(2, len(rho_prime) + 2)
    with open(path, "w", encoding="utf-8") as f:
        f.write("k,rho_prime,block_mean,block_std,z\n")
        for i in range(len(rho_prime)):
            f.write(f"{k[i]},{rho_prime[i]},{block_mean[i]},{block_std[i]},{z[i]}\n")


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
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--zthr", type=float, default=2.0)
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} ...")
    gaps = prime_gaps(args.pmax)
    print(f"Number of gaps: {len(gaps)}")
    print(f"Settings: max_lag={args.max_lag}, trials={args.trials}, block={args.block}")

    sanity_check(gaps, args.max_lag, args.seed)

    rho_prime = spearman_curve(gaps, args.max_lag)

    rng = np.random.default_rng(args.seed)
    block_curves = np.zeros((args.trials, args.max_lag - 1))
    for t in range(args.trials):
        g = block_permute_gaps(gaps, args.block, rng)
        block_curves[t] = spearman_curve(g, args.max_lag)

    block_mean = block_curves.mean(axis=0)
    block_std = block_curves.std(axis=0, ddof=1)
    block_std[block_std == 0] = 1e-12

    z = plot_results(rho_prime, block_mean, block_std, args.out_prefix)

    k_last, frac_sig, run_max, area_pos = coherence_metrics(z, args.zthr)

    print("=== COHERENCE SUMMARY (vs block null) ===")
    print(f"Threshold zthr = {args.zthr:.2f}")
    print(f"Last lag with z > zthr: {k_last}")
    print(f"Fraction of lags with z > zthr: {frac_sig:.3f}")
    print(f"Longest consecutive run with z > zthr: {run_max} lags")
    print(f"Area above threshold Σ max(z - zthr, 0): {area_pos:.2f}")

    csv_path = args.out_csv or f"{args.out_prefix}_rho_z.csv"
    write_csv(csv_path, rho_prime, block_mean, block_std, z)

    print("\nSaved outputs:")
    print(f"  {args.out_prefix}_rho_vs_lag.png")
    print(f"  {args.out_prefix}_rho_zscore_vs_lag.png")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
