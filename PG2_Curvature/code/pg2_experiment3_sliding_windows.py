# ============================================================
# File: pg2_experiment3_sliding_windows.py
# Prime Geometry II — Experiment 3:
# Sliding-window action for primes vs random permutations.
#
# This script:
#   - Generates the first NUM_PRIMES primes
#   - Computes gaps, χ_n, and L_n = χ_n^2
#   - For each window size W in WINDOW_SIZES:
#       * Computes windowed mean action for the true prime sequence
#       * Computes windowed mean action for random permutations
#         (averaged across NUM_PERMUTATIONS shuffles)
#   - Plots prime vs random windowed action curves.
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

NUM_PRIMES = 50000         # You can raise to 100000 or 200000 if patient
NUM_PERMUTATIONS = 50      # Number of random permutations to average over

WINDOW_SIZES = [1000, 5000]  # Sliding window sizes (in indices of L_n)

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
# Helper: compute chi_n and L_n from gap list
# ============================================================

def compute_chi_and_L(gap_list):
    chi_values = []
    L_values = []
    # χ_n uses g_n, g_{n+1}, g_{n+2}  → indices 0 .. M-3
    for n in range(M - 2):
        g0 = gap_list[n]
        g1 = gap_list[n+1]
        g2 = gap_list[n+2]
        denom = g0 + g1
        if denom == 0:
            continue
        chi = (g2 - g0) / denom
        chi_values.append(chi)
        L_values.append(chi * chi)
    return chi_values, L_values

# ============================================================
# Helper: sliding window mean using prefix sums
# ============================================================

def sliding_window_means(values, window_size):
    """
    Given a list 'values' and window_size W, return a list of
    windowed means:
        mean(values[i:i+W]) for i = 0 .. len(values)-W
    Computed in O(n) time using prefix sums.
    """
    n = len(values)
    if window_size > n:
        raise ValueError("Window size larger than length of values.")
    # prefix sums
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    means = []
    for i in range(n - window_size + 1):
        window_sum = prefix[i + window_size] - prefix[i]
        means.append(window_sum / window_size)
    return means

# ============================================================
# 3. Compute χ_n and L_n for true primes
# ============================================================

print("Computing χ_n and L_n for true prime sequence...")
chi_true, L_true = compute_chi_and_L(gaps)
N_L = len(L_true)

print("\n=== TRUE PRIME SEQUENCE (GLOBAL) ===")
print(f"Number of L_n values: {N_L}")
print(f"Mean L:   {statistics.mean(L_true)}")
print(f"Var(chi): {statistics.pvariance(chi_true)}")
print(f"Max |chi|: {max(abs(c) for c in chi_true)}")

# ============================================================
# 4. Prepare random permutations and average windowed means
# ============================================================

# For each window size, we'll accumulate a running sum of windowed means
# across permutations, then divide at the end to get an average baseline.

random_window_means_sum = {W: None for W in WINDOW_SIZES}

print(f"\nGenerating {NUM_PERMUTATIONS} random permutations for sliding-window baselines...")

for _ in tqdm(range(NUM_PERMUTATIONS)):
    perm = gaps.copy()
    random.shuffle(perm)
    _, L_rand = compute_chi_and_L(perm)

    for W in WINDOW_SIZES:
        if W > len(L_rand):
            continue
        means_rand = sliding_window_means(L_rand, W)
        if random_window_means_sum[W] is None:
            random_window_means_sum[W] = [0.0] * len(means_rand)
        # accumulate
        acc = random_window_means_sum[W]
        for i, val in enumerate(means_rand):
            acc[i] += val

# Convert sums to averages
random_window_means_avg = {}
for W, sums in random_window_means_sum.items():
    if sums is None:
        continue
    random_window_means_avg[W] = [s / NUM_PERMUTATIONS for s in sums]

# ============================================================
# 5. Compute sliding window means for the true prime sequence
# ============================================================

true_window_means = {}
for W in WINDOW_SIZES:
    if W > len(L_true):
        continue
    true_window_means[W] = sliding_window_means(L_true, W)

# ============================================================
# 6. Plot results for each window size
# ============================================================

for W in WINDOW_SIZES:
    if W not in true_window_means or W not in random_window_means_avg:
        continue

    prime_means = true_window_means[W]
    rand_means = random_window_means_avg[W]

    # x-axis: index of window center (or starting index)
    # Here we use starting index; we could shift by +W/2 if we want centers.
    x_vals = list(range(len(prime_means)))

    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, prime_means, label=f"Primes (window mean L, W={W})", linewidth=1.0)
    plt.plot(x_vals, rand_means, label=f"Random avg (W={W})", linewidth=1.0, alpha=0.8)
    plt.title(f"Sliding-Window Mean Action: W={W}, N_primes={NUM_PRIMES}, perms={NUM_PERMUTATIONS}")
    plt.xlabel("Window start index (n)")
    plt.ylabel("Mean L in window")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nDone with sliding-window experiment.")
