# ============================================================
# File: pg2_experiment1_curvature_action.py
# Prime Geometry II — Experiment 1:
# Curvature χ_n, Lagrangian ℒ_n, total action S
# for primes vs random gap permutations
# ============================================================

import sympy as sp
import random
import statistics
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 200000      
NUM_PERMUTATIONS = 200   

# ============================================================
# 1. Generate primes
# ============================================================

print(f"Generating first {NUM_PRIMES} primes...")
primes = list(sp.primerange(2, sp.prime(NUM_PRIMES) + 1))

# Sanity check
if len(primes) < NUM_PRIMES:
    raise ValueError("Prime generation failed.")

# ============================================================
# 2. Compute gaps
# ============================================================

gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
M = len(gaps)

# ============================================================
# Helper: compute chi, Lagrangian, and action S
# ============================================================

def compute_chi_and_action(gap_list):
    chi_values = []
    L_values = []

    # chi_n uses g_n, g_{n+1}, g_{n+2} → index up to M-3
    for n in range(M - 2):
        g_n = gap_list[n]
        g1 = gap_list[n+1]
        g2 = gap_list[n+2]

        denom = g_n + g1
        if denom == 0:
            continue  # shouldn't happen for primes

        chi = (g2 - g_n) / denom
        L = chi * chi

        chi_values.append(chi)
        L_values.append(L)

    S = sum(L_values)
    mean_L = S / len(L_values)
    return chi_values, L_values, S, mean_L

# ============================================================
# 3. Compute for true primes
# ============================================================

print("Computing χ_n and action S for true prime sequence...")
chi_true, L_true, S_true, meanL_true = compute_chi_and_action(gaps)

print("\n=== TRUE PRIME SEQUENCE RESULTS ===")
print(f"S_true (total action): {S_true}")
print(f"mean(L_true): {meanL_true}")
print(f"Variance of chi: {statistics.pvariance(chi_true)}")
print(f"Max |chi|: {max(abs(c) for c in chi_true)}")
print(f"Number of chi values: {len(chi_true)}")

# ============================================================
# 4. Random permutations comparison
# ============================================================

print(f"\nGenerating {NUM_PERMUTATIONS} random permutations...")
S_random = []
meanL_random = []
var_chi_random = []
maxabs_chi_random = []

for _ in tqdm(range(NUM_PERMUTATIONS)):
    perm = gaps.copy()
    random.shuffle(perm)

    chi_r, L_r, S_r, meanL_r = compute_chi_and_action(perm)
    S_random.append(S_r)
    meanL_random.append(meanL_r)
    var_chi_random.append(statistics.pvariance(chi_r))
    maxabs_chi_random.append(max(abs(c) for c in chi_r))

# ============================================================
# 5. Summary
# ============================================================

print("\n=== RANDOM PERMUTATIONS SUMMARY ===")
print(f"Mean S_random:       {statistics.mean(S_random)}")
print(f"Median S_random:     {statistics.median(S_random)}")
print(f"Min S_random:        {min(S_random)}")
print(f"Max S_random:        {max(S_random)}")

print(f"\nMean mean(L):        {statistics.mean(meanL_random)}")
print(f"Mean var(chi):       {statistics.mean(var_chi_random)}")
print(f"Mean max|chi|:       {statistics.mean(maxabs_chi_random)}")

# Percentile of S_true inside S_random
sorted_S = sorted(S_random)
rank = sum(1 for x in sorted_S if x < S_true)
percentile = 100 * rank / len(sorted_S)

print(f"\n=== COMPARISON ===")
print(f"S_true percentile among random permutations: {percentile:.2f}%")
print("(Lower percentile = smoother / lower curvature than random)")
