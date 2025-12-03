# ============================================================
# File: pg2_experiment4_autocorrelation.py
# Prime Geometry II — Experiment 4:
# Autocorrelation of curvature χ_n for primes vs random permutations.
#
# Requirements: sympy, tqdm, matplotlib
#   pip install sympy tqdm matplotlib
# ============================================================

import sympy as sp
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 50000         # you can bump this to 100000 later
NUM_PERMUTATIONS = 20      # number of random permutations for baseline
MAX_LAG = 1000             # maximum lag for autocorrelation

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
# Helper: autocorrelation function
# ============================================================

def autocorrelation(values, max_lag):
    """
    Compute normalized autocorrelation for lags 0..max_lag.
    ac[0] = 1 by definition. Mean is subtracted before computing.
    """
    n = len(values)
    if n == 0:
        return [0.0] * (max_lag + 1)

    mean = sum(values) / n
    centered = [v - mean for v in values]

    # variance (denominator)
    var = sum(v * v for v in centered) / n
    if var == 0:
        return [0.0] * (max_lag + 1)
    denom = var * n  # since numerators will use sum(...) over n-l terms

    ac = []
    for lag in range(max_lag + 1):
        s = 0.0
        limit = n - lag
        for i in range(limit):
            s += centered[i] * centered[i + lag]
        ac.append(s / denom)
    return ac

# ============================================================
# 3. χ_n for true primes + autocorrelation
# ============================================================

print("Computing χ_n for true primes...")
chi_true = compute_chi(gaps)
print(f"Number of χ_n values: {len(chi_true)}")

print(f"Computing autocorrelation up to lag {MAX_LAG} for primes...")
ac_true = autocorrelation(chi_true, MAX_LAG)

# ============================================================
# 4. χ_n autocorrelation for random permutations (baseline)
# ============================================================

print(f"\nComputing autocorrelation for {NUM_PERMUTATIONS} random permutations...")
ac_random_sum = [0.0] * (MAX_LAG + 1)

for _ in tqdm(range(NUM_PERMUTATIONS)):
    perm = gaps.copy()
    random.shuffle(perm)
    chi_r = compute_chi(perm)
    ac_r = autocorrelation(chi_r, MAX_LAG)
    for i in range(MAX_LAG + 1):
        ac_random_sum[i] += ac_r[i]

ac_random_avg = [s / NUM_PERMUTATIONS for s in ac_random_sum]

# ============================================================
# 5. Print first few lags for quick comparison
# ============================================================

print("\n=== Autocorrelation at small lags (prime vs random avg) ===")
print("lag\tac_prime\tac_random")
for lag in range(1, 21):
    print(f"{lag}\t{ac_true[lag]:.5f}\t\t{ac_random_avg[lag]:.5f}")

# ============================================================
# 6. Plot full autocorrelation curves
# ============================================================

lags = list(range(MAX_LAG + 1))

plt.figure(figsize=(12, 6))
plt.plot(lags, ac_true, label="Primes χ_n autocorr", linewidth=1.0)
plt.plot(lags, ac_random_avg, label="Random χ_n autocorr (avg)", linewidth=1.0, alpha=0.8)
plt.title(f"Autocorrelation of χ_n (N_primes={NUM_PRIMES}, perms={NUM_PERMUTATIONS})")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDone with autocorrelation experiment.")
