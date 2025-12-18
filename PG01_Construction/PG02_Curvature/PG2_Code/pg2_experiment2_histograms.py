# ============================================================
# File: pg2_experiment2_histograms.py
# Prime Geometry II — Experiment 2:
# Histogram comparison of curvature χ_n for primes vs random
#
# Requirements: matplotlib, sympy, tqdm
# Install if needed:
#   pip install matplotlib tqdm sympy
# ============================================================

import sympy as sp
import random
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 50000         # You can raise this to 100k or 200k
NUM_PERMUTATIONS = 50      # Use 50 for visualization; can go higher

# Histogram parameters
NUM_BINS = 200

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

gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
M = len(gaps)

# ============================================================
# Helper: compute chi_n from a gap list
# ============================================================

def compute_chi(gap_list):
    chi_values = []
    for n in range(M - 2):
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
# 3. Compute chi for true primes
# ============================================================

print("Computing χ_n for true primes...")
chi_true = compute_chi(gaps)

var_true = statistics.pvariance(chi_true)
skew_true = statistics.mean(chi_true)  # Should be ~0
kurt_true = statistics.pvariance([(x - statistics.mean(chi_true))**2 for x in chi_true])

print("\n=== TRUE PRIMES χ_n STATS ===")
print(f"Count: {len(chi_true)}")
print(f"Variance: {var_true}")
print(f"Mean: {statistics.mean(chi_true)}")
print(f"Max |χ|: {max(abs(c) for c in chi_true)}")

# ============================================================
# 4. Compute chi for random permutations
# ============================================================

print(f"\nGenerating {NUM_PERMUTATIONS} random permutations for χ distributions...")

chi_random_all = []

for _ in tqdm(range(NUM_PERMUTATIONS)):
    perm = gaps.copy()
    random.shuffle(perm)
    chi_r = compute_chi(perm)
    chi_random_all.extend(chi_r)

var_rand = statistics.pvariance(chi_random_all)
mean_rand = statistics.mean(chi_random_all)
max_abs_rand = max(abs(c) for c in chi_random_all)

print("\n=== RANDOM χ_n STATS (combined) ===")
print(f"Count: {len(chi_random_all)}")
print(f"Variance: {var_rand}")
print(f"Mean: {mean_rand}")
print(f"Max |χ|: {max_abs_rand}")

# ============================================================
# 5. Plot histograms
# ============================================================

plt.figure(figsize=(12, 6))
plt.hist(chi_true, bins=NUM_BINS, alpha=0.6, density=True, label="Primes χ_n")
plt.hist(chi_random_all, bins=NUM_BINS, alpha=0.4, density=True, label="Random χ_n")
plt.title(f"Histogram Comparison of χ_n (N={NUM_PRIMES}, Perms={NUM_PERMUTATIONS})")
plt.xlabel("χ_n")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ============================================================
# 6. Tail statistics
# ============================================================

TAIL_THRESHOLD = 5  # values with |chi| > 5

tails_true = sum(1 for x in chi_true if abs(x) > TAIL_THRESHOLD)
tails_rand = sum(1 for x in chi_random_all if abs(x) > TAIL_THRESHOLD)

print("\n=== TAIL STATISTICS (|χ| > 5) ===")
print(f"Primes tail count:  {tails_true}")
print(f"Random tail count:  {tails_rand}")
print(f"Tail ratio (rand/prime): {tails_rand / max(1, tails_true):.2f}x")

print("\nDone.")