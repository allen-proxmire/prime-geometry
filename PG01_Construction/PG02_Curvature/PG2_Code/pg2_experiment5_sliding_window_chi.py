# ============================================================
# File: pg2_experiment5_sliding_window_chi.py
# Prime Geometry II — Experiment 5:
# Sliding-window mean of curvature χ_n for primes vs random permutations.
#
# Requirements: sympy, tqdm, matplotlib
#   pip install sympy tqdm matplotlib
# ============================================================

import sympy as sp
import random
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 50000         # bump to 100000 later if you want
NUM_PERMUTATIONS = 50      # number of random permutations to average

WINDOW_SIZES = [500, 2000, 5000]   # sliding window sizes (in χ_n indices)

# ============================================================
# 1. Generate primes
# ============================================================

print(f"Generating first {NUM_PRIMES} primes...")
primes = list(sp.primerange(2, sp.prime(NUM_PRIMES) + 1))
if len(primes) < NUM_PRIMES:
    raise ValueError("Prime generation failed.")

# ============================================================
# 2. Compute gaps
# ============================================================

gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
M = len(gaps)

# ============================================================
# Helper: compute chi_n from a gap list
# ============================================================

def compute_chi(gap_list):
    chi_values = []
    m = len(gap_list)
    for n in range(m - 2):
        g0 = gap_list[n]
        g1 = gap_list[n+1]
        g2 = gap_list[n+2]
        denom = g0 + g1
        if denom == 0:
            continue
        chi = (g2 - g0) / denom
        chi_values.append(chi)
    return chi_values

# ============================================================
# Helper: sliding window means using prefix sums
# ============================================================

def sliding_window_means(values, window_size):
    """
    Return list of means of consecutive windows of length W:
        mean(values[i:i+W]) for i = 0 .. len(values)-W
    """
    n = len(values)
    if window_size > n:
        raise ValueError("Window size larger than sequence length.")
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    means = []
    for i in range(n - window_size + 1):
        s = prefix[i + window_size] - prefix[i]
        means.append(s / window_size)
    return means

# ============================================================
# 3. Compute χ_n for true primes
# ============================================================

print("Computing χ_n for true primes...")
chi_true = compute_chi(gaps)
N_chi = len(chi_true)

print("\n=== TRUE PRIME χ_n (GLOBAL) ===")
print(f"Number of χ_n values: {N_chi}")
print(f"Mean χ:   {statistics.mean(chi_true)}")
print(f"Var(χ):   {statistics.pvariance(chi_true)}")
print(f"Max |χ|:  {max(abs(c) for c in chi_true)}")

# ============================================================
# 4. Prepare random permutations and average windowed χ
# ============================================================

random_window_means_sum = {W: None for W in WINDOW_SIZES}

print(f"\nGenerating {NUM_PERMUTATIONS} random permutations for χ baselines...")

for _ in tqdm(range(NUM_PERMUTATIONS)):
    perm = gaps.copy()
    random.shuffle(perm)
    chi_rand = compute_chi(perm)

    for W in WINDOW_SIZES:
        if W > len(chi_rand):
            continue
        means_rand = sliding_window_means(chi_rand, W)
        if random_window_means_sum[W] is None:
            random_window_means_sum[W] = [0.0] * len(means_rand)
        acc = random_window_means_sum[W]
        for i, val in enumerate(means_rand):
            acc[i] += val

# turn sums into averages
random_window_means_avg = {}
for W, sums in random_window_means_sum.items():
    if sums is None:
        continue
    random_window_means_avg[W] = [s / NUM_PERMUTATIONS for s in sums]

# ============================================================
# 5. Sliding-window means for the true prime χ_n
# ============================================================

true_window_means = {}
for W in WINDOW_SIZES:
    if W > len(chi_true):
        continue
    true_window_means[W] = sliding_window_means(chi_true, W)

# ============================================================
# 6. Plot for each window size
# ============================================================

for W in WINDOW_SIZES:
    if W not in true_window_means or W not in random_window_means_avg:
        continue

    prime_means = true_window_means[W]
    rand_means = random_window_means_avg[W]

    x_vals = list(range(len(prime_means)))

    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, prime_means, label=f"Primes (window mean χ, W={W})", linewidth=1.0)
    plt.plot(x_vals, rand_means, label=f"Random avg (window mean χ, W={W})", linewidth=1.0, alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.title(f"Sliding-Window Mean χ_n: W={W}, N_primes={NUM_PRIMES}, perms={NUM_PERMUTATIONS}")
    plt.xlabel("Window start index (n)")
    plt.ylabel("Mean χ in window")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nDone with sliding-window χ experiment.")
