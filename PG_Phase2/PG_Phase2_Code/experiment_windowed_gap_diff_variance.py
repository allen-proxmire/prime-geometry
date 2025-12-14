"""
Prime Geometry — Windowed Gap-Difference Variance Test

Goal:
Measure how fast local cancellation of gap differences occurs.

We compute, for window size W:
    S_n(W) = sum_{k=n}^{n+W-1} (g_{k+1} - g_k)

and compare the variance of S_n(W) for:
- primes
- a permutation null

Short memory => variance saturates quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
MAX_N = 200_000      # number of gaps
MAX_W = 30           # window sizes to test

np.random.seed(SEED)

# -----------------------------
# GENERATE PRIMES
# -----------------------------
print("Generating primes...")
primes = np.array(list(primerange(2, P_MAX)), dtype=np.int64)
gaps = np.diff(primes)
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
# WINDOWED SUM VARIANCE
# -----------------------------
def windowed_variance(delta, max_w):
    vars_ = []
    for W in range(1, max_w + 1):
        sums = np.array([
            np.sum(delta[i:i+W])
            for i in range(len(delta) - W)
        ])
        vars_.append(np.var(sums))
    return np.array(vars_)

print("Computing windowed variances...")
var_prime = windowed_variance(delta_g, MAX_W)
var_null = windowed_variance(delta_g_null, MAX_W)

# -----------------------------
# PLOT
# -----------------------------
Ws = np.arange(1, MAX_W + 1)

plt.figure(figsize=(8,5))
plt.plot(Ws, var_prime, marker='o', linewidth=2, label="Primes")
plt.plot(Ws, var_null, marker='o', linestyle='--', label="Permutation null")
plt.xlabel("Window size W")
plt.ylabel("Variance of windowed sum")
plt.title("Windowed Gap-Difference Variance")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# PRINT SUMMARY
# -----------------------------
print("\n=== VARIANCE SUMMARY ===")
for W in [1, 2, 3, 5, 10, 20, 30]:
    if W <= MAX_W:
        print(f"W={W:2d} | primes={var_prime[W-1]:.2f} | null={var_null[W-1]:.2f}")
