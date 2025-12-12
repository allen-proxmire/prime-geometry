# ============================================================
# File: pg2_experiment6_cramer_comparison.py
# Prime Geometry II — Experiment 6:
# Cramér-model comparison for sliding-window mean χ_n.
#
# Sequences compared:
#   - True primes
#   - Cramér pseudo-primes (probability 1 / log n)
#   - Random permutations of the true prime gaps
#
# Requirements: sympy, tqdm, matplotlib
#   pip install sympy tqdm matplotlib
# ============================================================

import sympy as sp
import random
import math
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 50000         # true primes count
NUM_PERMUTATIONS = 50      # permutations for random-gap baseline

WINDOW_SIZES = [1000, 5000]   # window sizes in χ_n indices

# ============================================================
# Helpers
# ============================================================

def compute_chi(gap_list):
    """Compute χ_n from a list of gaps."""
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

def sliding_window_means(values, window_size):
    """Mean of values[i:i+W] for all valid windows."""
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
# 1. True primes and their χ_n
# ============================================================

print(f"Generating first {NUM_PRIMES} true primes...")
primes = list(sp.primerange(2, sp.prime(NUM_PRIMES) + 1))
if len(primes) < NUM_PRIMES:
    raise ValueError("Prime generation failed.")

gaps_true = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
chi_true = compute_chi(gaps_true)

print("\n=== TRUE PRIMES χ_n ===")
print(f"Count: {len(chi_true)}")
print(f"Mean χ: {statistics.mean(chi_true)}")
print(f"Var χ:  {statistics.pvariance(chi_true)}")
print(f"Max |χ|: {max(abs(c) for c in chi_true)}")

# ============================================================
# 2. Cramér-model pseudo-primes and their χ_n
# ============================================================

max_n = primes[-1]  # same numerical range as true primes

print(f"\nGenerating Cramér pseudo-primes up to n = {max_n}...")
cramer_primes = []
for n in range(2, max_n + 1):
    # avoid log(1) etc; probability ~ 1 / log n
    p = 1.0 / math.log(n) if n > 2 else 1.0
    if random.random() < p:
        cramer_primes.append(n)

if len(cramer_primes) < 10:
    raise RuntimeError("Cramér model produced too few pseudo-primes; try increasing max_n or adjusting model.")

gaps_cramer = [cramer_primes[i+1] - cramer_primes[i] for i in range(len(cramer_primes) - 1)]
chi_cramer = compute_chi(gaps_cramer)

print("\n=== CRAMÉR χ_n ===")
print(f"Count: {len(chi_cramer)}")
print(f"Mean χ: {statistics.mean(chi_cramer)}")
print(f"Var χ:  {statistics.pvariance(chi_cramer)}")
print(f"Max |χ|: {max(abs(c) for c in chi_cramer)}")

# ============================================================
# 3. Random permutations of true gaps → χ_n baseline
# ============================================================

print(f"\nGenerating {NUM_PERMUTATIONS} random permutations for baseline χ...")
random_window_means_sum = {W: None for W in WINDOW_SIZES}

# We'll also compute a combined χ list for global stats
chi_random_all = []

for _ in tqdm(range(NUM_PERMUTATIONS)):
    perm = gaps_true.copy()
    random.shuffle(perm)
    chi_rand = compute_chi(perm)
    chi_random_all.extend(chi_rand)

    for W in WINDOW_SIZES:
        if W > len(chi_rand):
            continue
        means_rand = sliding_window_means(chi_rand, W)
        if random_window_means_sum[W] is None:
            random_window_means_sum[W] = [0.0] * len(means_rand)
        acc = random_window_means_sum[W]
        for i, val in enumerate(means_rand):
            acc[i] += val

chi_rand_mean = statistics.mean(chi_random_all)
chi_rand_var = statistics.pvariance(chi_random_all)
chi_rand_max = max(abs(c) for c in chi_random_all)

print("\n=== RANDOM-PERM χ_n (global, combined) ===")
print(f"Count: {len(chi_random_all)}")
print(f"Mean χ: {chi_rand_mean}")
print(f"Var χ:  {chi_rand_var}")
print(f"Max |χ|: {chi_rand_max}")

# Turn sums into averages for baseline curves
random_window_means_avg = {}
for W, sums in random_window_means_sum.items():
    if sums is None:
        continue
    random_window_means_avg[W] = [s / NUM_PERMUTATIONS for s in sums]

# ============================================================
# 4. Sliding-window means for true primes & Cramér χ_n
# ============================================================

true_window_means = {}
cramer_window_means = {}

for W in WINDOW_SIZES:
    if W <= len(chi_true):
        true_window_means[W] = sliding_window_means(chi_true, W)
    if W <= len(chi_cramer):
        cramer_window_means[W] = sliding_window_means(chi_cramer, W)

# ============================================================
# 5. Plot comparisons for each window size
# ============================================================

for W in WINDOW_SIZES:
    if (W not in true_window_means or
        W not in cramer_window_means or
        W not in random_window_means_avg):
        continue

    prime_means = true_window_means[W]
    cramer_means = cramer_window_means[W]
    rand_means = random_window_means_avg[W]

    # Align lengths by truncating to the shortest
    L = min(len(prime_means), len(cramer_means), len(rand_means))
    prime_means = prime_means[:L]
    cramer_means = cramer_means[:L]
    rand_means = rand_means[:L]

    x_vals = list(range(L))

    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, prime_means, label=f"Primes (mean χ, W={W})", linewidth=1.0)
    plt.plot(x_vals, cramer_means, label=f"Cramér (mean χ, W={W})", linewidth=1.0, alpha=0.8)
    plt.plot(x_vals, rand_means, label=f"Random perms (mean χ, W={W})", linewidth=1.0, alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.title(f"Sliding-Window Mean χ_n Comparison (W={W}, N_primes={NUM_PRIMES})")
    plt.xlabel("Window start index (n)")
    plt.ylabel("Mean χ in window")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nDone with Cramér comparison experiment.")
