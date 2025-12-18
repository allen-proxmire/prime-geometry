#!/usr/bin/env python3
"""
PG02_02_mutual_information_plots.py

Mutual-information dependence test for prime gap ordering,
with automatic plot generation and saving.

PG02-style: constraint discovery, not dynamics.

Saves:
  - MI_vs_lag.png (and .pdf if requested)
  - MI_zscore_vs_lag.png (and .pdf if requested)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data acquisition
# -----------------------------

def gaps_from_file(path: str, max_n: Optional[int] = None) -> np.ndarray:
    gaps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            gaps.append(int(s))
            if max_n is not None and len(gaps) >= max_n:
                break
    return np.array(gaps, dtype=np.int64)


def gaps_from_primesieve(pmax: int) -> np.ndarray:
    try:
        import primesieve  # type: ignore
    except ImportError as e:
        raise ImportError(
            "primesieve not installed. Install with: pip install primesieve\n"
            "Or provide --gaps-file instead."
        ) from e

    primes = np.array(primesieve.primes(pmax), dtype=np.int64)
    return np.diff(primes)


# -----------------------------
# MI machinery
# -----------------------------

def quantile_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        edges = np.linspace(np.min(x), np.max(x), bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def discretize(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, edges[1:-1]).astype(np.int32)


def mutual_information_discrete(xb: np.ndarray, yb: np.ndarray, bins: int) -> float:
    n = xb.size
    joint = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(joint, (xb, yb), 1)

    px = joint.sum(axis=1) / n
    py = joint.sum(axis=0) / n
    pxy = joint / n

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return mi


def mi_curve(gaps: np.ndarray, max_lag: int, bins: int, edges: np.ndarray) -> np.ndarray:
    xb = discretize(gaps, edges)
    mi = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        mi[k - 1] = mutual_information_discrete(
            xb[:-k], xb[k:], bins=bins
        )
    return mi


# -----------------------------
# Null models
# -----------------------------

def permute_gaps(gaps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = gaps.copy()
    rng.shuffle(g)
    return g


def block_permute_gaps(gaps: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(gaps)
    m = n // block
    blocks = gaps[: m * block].reshape(m, block)
    order = rng.permutation(m)
    out = blocks[order].reshape(-1)
    if m * block < n:
        out = np.concatenate([out, gaps[m * block :]])
    return out


@dataclass
class NullStats:
    mean: np.ndarray
    std: np.ndarray


def null_stats(
    gaps: np.ndarray,
    max_lag: int,
    bins: int,
    edges: np.ndarray,
    trials: int,
    null_type: str,
    block: int,
    seed: int,
) -> NullStats:
    rng = np.random.default_rng(seed)
    curves = np.zeros((trials, max_lag))
    for t in range(trials):
        if null_type == "perm":
            g = permute_gaps(gaps, rng)
        else:
            g = block_permute_gaps(gaps, block, rng)
        curves[t] = mi_curve(g, max_lag, bins, edges)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1)
    std[std == 0] = 1e-12
    return NullStats(mean, std)


# -----------------------------
# Plotting
# -----------------------------

def plot_mi(
    mi_prime: np.ndarray,
    null: NullStats,
    out_prefix: str,
    save_pdf: bool,
):
    k = np.arange(1, len(mi_prime) + 1)

    # MI vs lag
    plt.figure()
    plt.plot(k, mi_prime, label="Primes", linewidth=2)
    plt.fill_between(
        k,
        null.mean - null.std,
        null.mean + null.std,
        alpha=0.3,
        label="Null ± 1σ",
    )
    plt.xlabel("Lag k")
    plt.ylabel("Mutual Information (nats)")
    plt.title("Mutual Information Between Prime Gaps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_MI_vs_lag.png", dpi=200)
    if save_pdf:
        plt.savefig(f"{out_prefix}_MI_vs_lag.pdf")
    plt.close()

    # Z-score plot
    z = (mi_prime - null.mean) / null.std
    plt.figure()
    plt.plot(k, z, linewidth=2)
    plt.axhline(2.0, linestyle="--", alpha=0.5)
    plt.axhline(-2.0, linestyle="--", alpha=0.5)
    plt.xlabel("Lag k")
    plt.ylabel("Z-score")
    plt.title("Mutual Information Z-score vs Null")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_MI_zscore_vs_lag.png", dpi=200)
    if save_pdf:
        plt.savefig(f"{out_prefix}_MI_zscore_vs_lag.pdf")
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pmax", type=int)
    src.add_argument("--gaps-file", type=str)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--max-lag", type=int, default=200)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--null", choices=["perm", "block"], default="perm")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-prefix", type=str, default="PG02_02_MI")
    ap.add_argument("--save-pdf", action="store_true")
    args = ap.parse_args()

    if args.gaps_file:
        gaps = gaps_from_file(args.gaps_file, args.max_n)
    else:
        gaps = gaps_from_primesieve(args.pmax)
        if args.max_n:
            gaps = gaps[: args.max_n]

    edges = quantile_bin_edges(gaps, args.bins)

    mi_prime = mi_curve(gaps, args.max_lag, args.bins, edges)
    null = null_stats(
        gaps,
        args.max_lag,
        args.bins,
        edges,
        args.trials,
        args.null,
        args.block,
        args.seed,
    )

    plot_mi(mi_prime, null, args.out_prefix, args.save_pdf)

    print("Saved plots:")
    print(f"  {args.out_prefix}_MI_vs_lag.png")
    print(f"  {args.out_prefix}_MI_zscore_vs_lag.png")


if __name__ == "__main__":
    main()
