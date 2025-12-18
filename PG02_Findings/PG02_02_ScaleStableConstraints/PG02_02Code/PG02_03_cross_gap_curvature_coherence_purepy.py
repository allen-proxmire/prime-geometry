#!/usr/bin/env python3
"""
PG02_03_cross_gap_curvature_coherence_purepy.py

Test B (PG02): Cross-compare the lagwise ordering dependence of the prime gap sequence
with the lagwise ordering dependence of the curvature sequence chi_n.

Definitions:
  gaps: g_n = p_{n+1} - p_n
  curvature: chi_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
Lags:
  For gaps: Spearman rho_g(k) between g_n and g_{n+k}, for k=2..K (lag-1 excluded).
  For chi:  Spearman rho_chi(k) between chi_n and chi_{n+k}, for k=2..K.

Null:
  Block-permutation of the gap sequence (preserving local structure within blocks),
  recomputing both rho_g(k) and rho_chi(k), giving Zg(k), Zchi(k).

Outputs:
  - Z overlay plot (Zg(k) and Zchi(k) vs k)
  - scatter plot Zg vs Zchi
  - CSV: k, Zg, Zchi, rho_g, rho_chi
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
    return np.diff(primes_up_to(pmax)).astype(np.float64)


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


def spearman_curve(series: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Spearman rho(k) for lags k = 2..max_lag (lag-1 excluded).
    Returns array of length (max_lag-1) aligned to k=2..max_lag.
    """
    out = np.zeros(max_lag - 1, dtype=np.float64)
    for k in range(2, max_lag + 1):
        out[k - 2] = spearman_corr(series[:-k], series[k:])
    return out


# ======================================================
# Curvature sequence
# ======================================================

def curvature_from_gaps(g: np.ndarray) -> np.ndarray:
    """
    chi_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
    Length = len(g) - 2
    """
    denom = g[:-2] + g[1:-1]
    denom = np.where(denom == 0, 1e-12, denom)
    chi = (g[2:] - g[:-2]) / denom
    return chi.astype(np.float64)


# ======================================================
# Block-permutation null
# ======================================================

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
# Utility: Z-score with safe std
# ======================================================

def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    s = std.copy()
    s[s == 0] = 1e-12
    return (x - mean) / s


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
    ap.add_argument("--out-prefix", type=str, default="PG02_03_cross")
    args = ap.parse_args()

    print(f"Generating primes ≤ {args.pmax} ...")
    gaps = prime_gaps(args.pmax)
    print(f"Number of gaps: {len(gaps)}")
    print(f"Settings: max_lag={args.max_lag}, trials={args.trials}, block={args.block}, seed={args.seed}")

    # Prime curves
    rho_g = spearman_curve(gaps, args.max_lag)
    chi = curvature_from_gaps(gaps)
    rho_chi = spearman_curve(chi, args.max_lag)

    # Null ensemble (block permute gaps, recompute both)
    rng = np.random.default_rng(args.seed)
    null_g = np.zeros((args.trials, args.max_lag - 1), dtype=np.float64)
    null_chi = np.zeros((args.trials, args.max_lag - 1), dtype=np.float64)

    for t in range(args.trials):
        g_perm = block_permute_gaps(gaps, args.block, rng)
        null_g[t] = spearman_curve(g_perm, args.max_lag)
        chi_perm = curvature_from_gaps(g_perm)
        null_chi[t] = spearman_curve(chi_perm, args.max_lag)

    mean_g = null_g.mean(axis=0)
    std_g = null_g.std(axis=0, ddof=1)
    mean_chi = null_chi.mean(axis=0)
    std_chi = null_chi.std(axis=0, ddof=1)

    Zg = zscore(rho_g, mean_g, std_g)
    Zchi = zscore(rho_chi, mean_chi, std_chi)

    # Correlation across k between the Z-score curves
    # (ignore any NaNs; shouldn't happen, but be safe)
    mask = np.isfinite(Zg) & np.isfinite(Zchi)
    Zg_m = Zg[mask]
    Zchi_m = Zchi[mask]

    # Pearson correlation
    pearson = float(np.corrcoef(Zg_m, Zchi_m)[0, 1]) if len(Zg_m) > 2 else float("nan")
    # Spearman correlation (over k), i.e., rank corr of the Z-vectors
    spearman = spearman_corr(Zg_m, Zchi_m) if len(Zg_m) > 2 else float("nan")

    print("\n=== CROSS-STATISTIC SUMMARY ===")
    print(f"Pearson corr across k:  {pearson:.3f}")
    print(f"Spearman corr across k: {spearman:.3f}")

    # Plots
    k = np.arange(2, args.max_lag + 1)

    plt.figure()
    plt.plot(k, Zg, linewidth=2, label="Z_g(k) (gap ordering vs block null)")
    plt.plot(k, Zchi, linewidth=2, label="Z_χ(k) (curvature ordering vs block null)")
    plt.axhline(2, linestyle="--", alpha=0.5)
    plt.axhline(-2, linestyle="--", alpha=0.5)
    plt.xlabel("Lag k")
    plt.ylabel("Z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_Z_overlay.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(Zg, Zchi, s=18)
    plt.xlabel("Z_g(k)")
    plt.ylabel("Z_χ(k)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_Z_scatter.png", dpi=200)
    plt.close()

    # CSV
    csv_path = f"{args.out_prefix}_Z_cross.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("k,rho_g,rho_chi,Zg,Zchi\n")
        for i in range(len(k)):
            f.write(f"{k[i]},{rho_g[i]},{rho_chi[i]},{Zg[i]},{Zchi[i]}\n")

    print("\nSaved outputs:")
    print(f"  {args.out_prefix}_Z_overlay.png")
    print(f"  {args.out_prefix}_Z_scatter.png")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
