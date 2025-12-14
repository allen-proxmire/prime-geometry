"""
Prime Geometry — Gap Difference Cancellation Test

Goal:
Test whether first differences of prime gaps
    Δg_n = g_{n+1} - g_n
exhibit rapid cancellation (short memory)
compared to a permutation null.

This is a simple arithmetic confirmation of
short-range structure detected via PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
MAX_N = 200_000   # number of gaps to examine

np.random.seed(SEED)

# -----------------------------
# GENERATE PRIMES
# -----------------------------
print("Generating primes...")
primes = np.array(list(primerange(2, P_MAX)), dtype=np.int64)

# Prime gaps
gaps = np.diff(primes)

# Truncate for clarity / speed
gaps = gaps[:MAX_N + 1]

# -----------------------------
# PERMUTATION NULL
# -----------------------------
gaps_null = gaps.copy()
np.random.shuffle(gaps_null)

# -----------------------------
# GAP DIFFERENCES
# -----------------------------
delta_g = np.diff(gaps)
delta_g_null = np.diff(gaps_null)

# -----------------------------
# CUMULATIVE SUMS
# -----------------------------
cum_prime = np.cumsum(delta_g)
cum_null = np.cumsum(delta_g_null)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(cum_prime, label="Primes", linewidth=2)
plt.plot(cum_null, label="Permutation null", alpha=0.7)
plt.xlabel("Index n")
plt.ylabel("Cumulative sum of (g_{n+1} - g_n)")
plt.title("Cumulative Sum of Prime Gap Differences")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------
print("\n=== CANCELLATION SUMMARY ===")
print(f"Primes: final sum          = {cum_prime[-1]}")
print(f"Null:   final sum          = {cum_null[-1]}")

print(f"Primes: max |sum|          = {np.max(np.abs(cum_prime))}")
print(f"Null:   max |sum|          = {np.max(np.abs(cum_null))}")

print(f"Primes: std of cumulative  = {np.std(cum_prime):.2f}")
print(f"Null:   std of cumulative  = {np.std(cum_null):.2f}")
