# ============================================================
# File: pg2_make_action_figure.py
# Purpose: Generate pg2_action.png (Action comparison figure)
# ============================================================

import sympy as sp
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
NUM_PRIMES = 50000
NUM_PERMS = 200   # number of random permutations

# -----------------------------
# 1. Generate primes and gaps
# -----------------------------
print(f"Generating first {NUM_PRIMES} primes...")
primes = list(sp.primerange(2, sp.prime(NUM_PRIMES) + 1))

gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

# -----------------------------
# 2. Curvature function
# -----------------------------
def compute_action(gap_list):
    """Compute S = sum chi_n^2 for a given gap ordering."""
    S = 0.0
    m = len(gap_list)
    for n in range(m - 2):
        g0 = gap_list[n]
        g1 = gap_list[n+1]
        g2 = gap_list[n+2]
        denom = g0 + g1
        if denom == 0:
            continue
        chi = (g2 - g0) / denom
        S += chi * chi
    return S

# -----------------------------
# 3. Compute true action
# -----------------------------
print("Computing true action...")
S_true = compute_action(gaps)

# -----------------------------
# 4. Compute action distribution
# -----------------------------
print(f"Generating {NUM_PERMS} random permutations...")
S_random = []

for _ in range(NUM_PERMS):
    perm = gaps.copy()
    random.shuffle(perm)
    S_random.append(compute_action(perm))

S_random = np.array(S_random)

# -----------------------------
# 5. Plot figure
# -----------------------------
plt.figure(figsize=(10,6))

plt.hist(S_random, bins=30, color='gray', alpha=0.7, label="Random permutations")
plt.axvline(S_true, color='red', linewidth=2,
            label=f"True primes (S = {S_true:.1f})")

plt.xlabel("Total action S")
plt.ylabel("Frequency")
plt.title("Action Comparison: True Primes vs Random Permutations")
plt.legend()

plt.tight_layout()
plt.savefig("pg2_action.png", dpi=300)
print("Saved pg2_action.png")
