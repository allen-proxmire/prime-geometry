"""
Prime Geometry — Triple PCA Test (Paper 2, Step 1)

Goal:
Test whether triples of renormalized prime gaps
    (g_n/log p_n, g_{n+1}/log p_{n+1}, g_{n+2}/log p_{n+2})
occupy a lower-dimensional region than a permutation null.

This script:
1. Generates primes and gaps up to 1e7
2. Constructs permutation null
3. Builds renormalized gap triples
4. Performs PCA on primes and nulls
5. Reports eigenvalues and thickness ratios
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000
SEED = 42
np.random.seed(SEED)

# -----------------------------
# STEP 1 — GENERATE PRIMES
# -----------------------------
print("Generating primes...")
primes = np.array(list(primerange(2, P_MAX)), dtype=np.int64)
gaps = np.diff(primes)

# -----------------------------
# STEP 1.2 — PERMUTATION NULL
# -----------------------------
print("Generating permutation null...")
gaps_null = gaps.copy()
np.random.shuffle(gaps_null)

primes_null = np.empty_like(primes)
primes_null[0] = primes[0]
for i in range(len(gaps_null)):
    primes_null[i + 1] = primes_null[i] + gaps_null[i]

# -----------------------------
# STEP 2 — RENORMALIZED GAPS
# -----------------------------
tilde_g = gaps / np.log(primes[:-1])
tilde_g_null = gaps_null / np.log(primes_null[:-1])

# -----------------------------
# STEP 3 — BUILD TRIPLES
# -----------------------------
def build_triples(arr):
    return np.column_stack([arr[:-2], arr[1:-1], arr[2:]])

triples_prime = build_triples(tilde_g)
triples_null = build_triples(tilde_g_null)

print(f"Prime triples: {triples_prime.shape}")
print(f"Null triples:  {triples_null.shape}")

# -----------------------------
# STEP 4 — PCA FUNCTION
# -----------------------------
def pca_eigenvalues(X):
    Xc = X - np.mean(X, axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return np.sort(eigvals)[::-1]  # descending

eig_prime = pca_eigenvalues(triples_prime)
eig_null = pca_eigenvalues(triples_null)

# -----------------------------
# RESULTS
# -----------------------------
print("\n=== PCA EIGENVALUES ===")
print("Primes:", eig_prime)
print("Nulls: ", eig_null)

ratio_prime = eig_prime[-1] / eig_prime[0]
ratio_null = eig_null[-1] / eig_null[0]

print("\n=== THICKNESS RATIO λ3 / λ1 ===")
print(f"Primes: {ratio_prime:.6f}")
print(f"Nulls:  {ratio_null:.6f}")

# -----------------------------
# OPTIONAL VISUALIZATION
# -----------------------------
plt.figure()
plt.bar([0, 1, 2], eig_prime, alpha=0.7, label="Primes")
plt.bar([0, 1, 2], eig_null, alpha=0.7, label="Nulls")
plt.xticks([0, 1, 2], [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"])
plt.ylabel("Eigenvalue")
plt.title("PCA Eigenvalues of Renormalized Gap Triples")
plt.legend()
plt.tight_layout()
plt.show()

