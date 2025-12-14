"""
Prime Geometry — Local PCA Thickness Test

Goal:
Measure how the PCA thickness ratio (λ3 / λ1) of
renormalized prime-gap triples varies locally
and compare to a permutation null.

This tests whether low-dimensional structure
persists across the prime sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
WINDOW_TRIPLES = 20_000   # number of triples per window
STEP = 5_000              # step size between windows

np.random.seed(SEED)

# -----------------------------
# GENERATE PRIMES
# -----------------------------
print("Generating primes...")
primes = np.array(list(primerange(2, P_MAX)), dtype=np.int64)
gaps = np.diff(primes)

# -----------------------------
# PERMUTATION NULL
# -----------------------------
print("Generating permutation null...")
gaps_null = gaps.copy()
np.random.shuffle(gaps_null)

primes_null = np.empty_like(primes)
primes_null[0] = primes[0]
for i in range(len(gaps_null)):
    primes_null[i + 1] = primes_null[i] + gaps_null[i]

# -----------------------------
# RENORMALIZED GAPS
# -----------------------------
tilde_g = gaps / np.log(primes[:-1])
tilde_g_null = gaps_null / np.log(primes_null[:-1])

# -----------------------------
# BUILD TRIPLES
# -----------------------------
def build_triples(arr):
    return np.column_stack([arr[:-2], arr[1:-1], arr[2:]])

triples_prime = build_triples(tilde_g)
triples_null = build_triples(tilde_g_null)

# -----------------------------
# PCA FUNCTION
# -----------------------------
def thickness_ratio(X):
    Xc = X - np.mean(X, axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    return eigvals[-1] / eigvals[0]

# -----------------------------
# LOCAL PCA
# -----------------------------
ratios_prime = []
ratios_null = []
indices = []

print("Running local PCA...")

for start in range(0, len(triples_prime) - WINDOW_TRIPLES, STEP):
    end = start + WINDOW_TRIPLES
    
    rp = thickness_ratio(triples_prime[start:end])
    rn = thickness_ratio(triples_null[start:end])
    
    ratios_prime.append(rp)
    ratios_null.append(rn)
    indices.append(start)

ratios_prime = np.array(ratios_prime)
ratios_null = np.array(ratios_null)

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(indices, ratios_prime, label="Primes", linewidth=2)
plt.plot(indices, ratios_null, label="Permutation null", alpha=0.7)
plt.xlabel("Triple index")
plt.ylabel(r"Thickness ratio $\lambda_3 / \lambda_1$")
plt.title("Local PCA Thickness of Renormalized Gap Triples")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# SUMMARY STATS
# -----------------------------
print("\n=== LOCAL THICKNESS SUMMARY ===")
print(f"Primes: mean={ratios_prime.mean():.4f}, std={ratios_prime.std():.4f}")
print(f"Nulls:  mean={ratios_null.mean():.4f}, std={ratios_null.std():.4f}")
