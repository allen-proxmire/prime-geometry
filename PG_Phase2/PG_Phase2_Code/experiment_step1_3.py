"""
Prime Geometry — Minimal Experiment (Steps 1–3)

Goal:
Test whether renormalized prime gaps
    g_n / log(p_n)
exhibit scale-stable behavior distinct from a permutation null.

This script:
1. Generates primes up to 1e7
2. Constructs prime gaps and a permutation null
3. Computes renormalized gaps
4. Runs scale comparison + null comparison
5. Produces 3 diagnostic plots + KS statistics

Nothing here assumes dynamics or interpretation.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import primerange
from scipy.stats import ks_2samp

# -----------------------------
# PARAMETERS
# -----------------------------
P_MAX = 10_000_000      # upper prime bound
SEED = 42               # reproducibility
WINDOW = 50_000         # sliding window size
BINS = np.linspace(0, 5, 100)

np.random.seed(SEED)

# -----------------------------
# STEP 1 — GENERATE PRIMES
# -----------------------------
print("Generating primes up to", P_MAX)
primes = np.array(list(primerange(2, P_MAX)), dtype=np.int64)

gaps = np.diff(primes)

print(f"Number of primes: {len(primes)}")
print(f"Number of gaps:   {len(gaps)}")

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
# STEP 2 — CORE OBSERVABLE
# -----------------------------
print("Computing renormalized gaps...")

tilde_g = gaps / np.log(primes[:-1])
tilde_g_null = gaps_null / np.log(primes_null[:-1])

# -----------------------------
# STEP 3.1 — SCALE SPLITS
# -----------------------------
mask_1 = (primes[:-1] >= 1e5) & (primes[:-1] < 1e6)
mask_2 = (primes[:-1] >= 1e6) & (primes[:-1] < 1e7)

tg_1 = tilde_g[mask_1]
tg_2 = tilde_g[mask_2]
tg_null_2 = tilde_g_null[mask_2]

print(f"Samples in 1e5–1e6: {len(tg_1)}")
print(f"Samples in 1e6–1e7: {len(tg_2)}")

# -----------------------------
# STEP 3.2 — PLOTS
# -----------------------------

# (a) Scale comparison histogram
plt.figure()
plt.hist(tg_1, bins=BINS, density=True, alpha=0.5, label="1e5–1e6")
plt.hist(tg_2, bins=BINS, density=True, alpha=0.5, label="1e6–1e7")
plt.xlabel(r"$\tilde g_n = g_n / \log p_n$")
plt.ylabel("Density")
plt.title("Scale comparison of renormalized gaps")
plt.legend()
plt.tight_layout()
plt.show()

# (b) Prime vs null histogram
plt.figure()
plt.hist(tg_2, bins=BINS, density=True, alpha=0.5, label="Primes")
plt.hist(tg_null_2, bins=BINS, density=True, alpha=0.5, label="Permutation null")
plt.xlabel(r"$\tilde g_n$")
plt.ylabel("Density")
plt.title("Primes vs permutation null")
plt.legend()
plt.tight_layout()
plt.show()

# (c) Sliding window mean
means = [
    np.mean(tilde_g[i:i + WINDOW])
    for i in range(0, len(tilde_g) - WINDOW, WINDOW)
]

plt.figure()
plt.plot(means)
plt.xlabel("Window index")
plt.ylabel(r"Mean of $\tilde g_n$")
plt.title("Sliding-window mean of renormalized gaps")
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 3.3 — KS STATISTICS
# -----------------------------
ks_scale = ks_2samp(tg_1, tg_2)
ks_null = ks_2samp(tg_2, tg_null_2)

print("\n=== KS TEST RESULTS ===")
print("Scale comparison (1e5–1e6 vs 1e6–1e7):")
print(ks_scale)

print("\nPrimes vs permutation null:")
print(ks_null)

print("\nExperiment complete.")
