"""
Prime Geometry — Block Permutation Null Test

Goal:
Test whether lagged triple PCA structure survives
block-permutation null models that preserve
short-range correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
BLOCK_SIZE = 5          # number of gaps per block
MAX_LAG = 10
MAX_TRIPLES = 200_000

np.random.seed(SEED)

# -----------------------------
# GENERATE PRIMES
# -----------------------------
print("Generating primes...")
primes = np.array(list(primerange(2, P_MAX)), dtype=np.int64)
gaps = np.diff(primes)

# -----------------------------
# BLOCK PERMUTATION NULL
# -----------------------------
def block_permute(arr, block_size):
    blocks = [arr[i:i+block_size]
              for i in range(0, len(arr), block_size)]
    np.random.shuffle(blocks)
    return np.concatenate(blocks)[:len(arr)]

gaps_block = block_permute(gaps, BLOCK_SIZE)

# -----------------------------
# RECONSTRUCT PSEUDO-PRIMES
# -----------------------------
def reconstruct_primes(gaps, start):
    p = np.empty(len(gaps) + 1, dtype=np.int64)
    p[0] = start
    for i in range(len(gaps)):
        p[i+1] = p[i] + gaps[i]
    return p

primes_block = reconstruct_primes(gaps_block, primes[0])

# -----------------------------
# RENORMALIZED GAPS
# -----------------------------
tilde_g = gaps / np.log(primes[:-1])
tilde_g_block = gaps_block / np.log(primes_block[:-1])

# -----------------------------
# PCA THICKNESS
# -----------------------------
def thickness_ratio(X):
    Xc = X - np.mean(X, axis=0)
    cov = np.cov(Xc, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.sort(eig)[::-1]
    return eig[-1] / eig[0]

# -----------------------------
# LAG SCAN
# -----------------------------
ratios_prime = []
ratios_block = []
lags = []

print("Running lagged PCA with block null...")

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
    
    triples_b = np.column_stack([
        tilde_g_block[idx],
        tilde_g_block[idx + k],
        tilde_g_block[idx + 2*k]
    ])
    
    rp = thickness_ratio(triples_p)
    rb = thickness_ratio(triples_b)
    
    ratios_prime.append(rp)
    ratios_block.append(rb)
    lags.append(k)
    
    print(f"k={k:2d} | primes={rp:.4f} | block-null={rb:.4f}")

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(lags, ratios_prime, marker='o', label="Primes")
plt.plot(lags, ratios_block, marker='o', label="Block-permutation null")
plt.xlabel("Lag k")
plt.ylabel(r"Thickness ratio $\lambda_3 / \lambda_1$")
plt.title("Lagged PCA: Primes vs Block-Permutation Null")
plt.legend()
plt.tight_layout()
plt.show()
