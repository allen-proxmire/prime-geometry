"""
Prime Geometry — Lagged Triple PCA Test

Goal:
Measure how PCA thickness of renormalized gap triples
depends on lag k:
    (g_n, g_{n+k}, g_{n+2k}) / log p

This probes the correlation length of prime-gap structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
MAX_LAG = 20          # test k = 1 ... MAX_LAG
MAX_TRIPLES = 200_000 # subsample for speed/stability

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
# PCA THICKNESS
# -----------------------------
def thickness_ratio(X):
    Xc = X - np.mean(X, axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    return eigvals[-1] / eigvals[0]

# -----------------------------
# LAG SCAN
# -----------------------------
ratios_prime = []
ratios_null = []
lags = []

print("Running lagged PCA...")

for k in range(1, MAX_LAG + 1):
    max_n = len(tilde_g) - 2*k
    idx = np.arange(max_n)
    
    if len(idx) > MAX_TRIPLES:
        idx = np.random.choice(idx, MAX_TRIPLES, replace=False)
    
    triples_p = np.column_stack([
        tilde_g[idx],
        tilde_g[idx + k],
        tilde_g[idx + 2*k]
    ])
    
    triples_n = np.column_stack([
        tilde_g_null[idx],
        tilde_g_null[idx + k],
        tilde_g_null[idx + 2*k]
    ])
    
    rp = thickness_ratio(triples_p)
    rn = thickness_ratio(triples_n)
    
    ratios_prime.append(rp)
    ratios_null.append(rn)
    lags.append(k)
    
    print(f"k={k:2d} | primes={rp:.4f} | null={rn:.4f}")

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(lags, ratios_prime, marker='o', label="Primes")
plt.plot(lags, ratios_null, marker='o', label="Permutation null")
plt.xlabel("Lag k")
plt.ylabel(r"Thickness ratio $\lambda_3 / \lambda_1$")
plt.title("Lag Dependence of Triple PCA Thickness")
plt.legend()
plt.tight_layout()
plt.show()
