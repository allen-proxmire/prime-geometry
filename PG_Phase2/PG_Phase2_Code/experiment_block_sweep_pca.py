"""
Prime Geometry — Block-Size Sweep for Lagged PCA

Goal:
Test how lagged triple PCA thickness depends on
block-permutation null size B.

Block sizes tested: B = 3, 5, 7, 10
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
BLOCK_SIZES = [3, 5, 7, 10]
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
# RENORMALIZED GAPS
# -----------------------------
tilde_g = gaps / np.log(primes[:-1])

# -----------------------------
# BLOCK PERMUTATION
# -----------------------------
def block_permute(arr, block_size):
    blocks = [arr[i:i+block_size]
              for i in range(0, len(arr), block_size)]
    np.random.shuffle(blocks)
    return np.concatenate(blocks)[:len(arr)]

def reconstruct_primes(gaps, start):
    p = np.empty(len(gaps) + 1, dtype=np.int64)
    p[0] = start
    for i in range(len(gaps)):
        p[i+1] = p[i] + gaps[i]
    return p

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
# PRIME BASELINE
# -----------------------------
def lagged_ratios(arr):
    ratios = []
    for k in range(1, MAX_LAG + 1):
        max_n = len(arr) - 2*k
        idx = np.arange(max_n)
        if len(idx) > MAX_TRIPLES:
            idx = np.random.choice(idx, MAX_TRIPLES, replace=False)
        triples = np.column_stack([
            arr[idx],
            arr[idx + k],
            arr[idx + 2*k]
        ])
        ratios.append(thickness_ratio(triples))
    return np.array(ratios)

print("Computing prime baseline...")
ratios_prime = lagged_ratios(tilde_g)

# -----------------------------
# BLOCK SWEEP
# -----------------------------
results = {}

for B in BLOCK_SIZES:
    print(f"\nBlock size B = {B}")
    gaps_block = block_permute(gaps, B)
    primes_block = reconstruct_primes(gaps_block, primes[0])
    tilde_g_block = gaps_block / np.log(primes_block[:-1])
    results[B] = lagged_ratios(tilde_g_block)

# -----------------------------
# PLOT
# -----------------------------
lags = np.arange(1, MAX_LAG + 1)

plt.figure(figsize=(9,6))
plt.plot(lags, ratios_prime, marker='o', linewidth=3,
         label="Primes")

for B in BLOCK_SIZES:
    plt.plot(lags, results[B], marker='o', linestyle='--',
             label=f"Block-null B={B}")

plt.xlabel("Lag k")
plt.ylabel(r"Thickness ratio $\lambda_3 / \lambda_1$")
plt.title("Lagged PCA Thickness vs Block Size")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# SUMMARY TABLE
# -----------------------------
print("\n=== SUMMARY (k=1 values) ===")
print("B     thickness")
print("-------------------")
print(f"Primes  {ratios_prime[0]:.4f}")
for B in BLOCK_SIZES:
    print(f"{B:<5d}  {results[B][0]:.4f}")
