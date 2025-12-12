# ============================================================
# File: pg2_experiment8_return_map.py
# Prime Geometry II — Experiment 8:
# Return-map portraits for gaps and curvature:
#   (g_n, g_{n+1}) and (chi_n, chi_{n+1})
#
# Sequences compared:
#   - True primes
#   - Random permutations of the true gaps
#   - Cramér pseudo-primes
#
# Requirements: sympy, matplotlib
#   pip install sympy matplotlib
# ============================================================

import sympy as sp
import random
import math
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 50000      # number of true primes
MAX_POINTS_PLOT = 20000 # max points per scatter (for visual clarity)

# ============================================================
# Helpers
# ============================================================

def compute_gaps(primes):
    return [primes[i+1] - primes[i] for i in range(len(primes) - 1)]

def compute_chi(gaps):
    chi_values = []
    m = len(gaps)
    for n in range(m - 2):
        g0 = gaps[n]
        g1 = gaps[n+1]
        g2 = gaps[n+2]
        denom = g0 + g1
        if denom == 0:
            continue
        chi = (g2 - g0) / denom
        chi_values.append(chi)
    return chi_values

def make_pairs(values):
    """Return list of (v_n, v_{n+1}) pairs."""
    return list(zip(values[:-1], values[1:]))

def subsample_pairs(pairs, max_points):
    """Subsample a list of pairs for plotting."""
    if len(pairs) <= max_points:
        return pairs
    step = len(pairs) // max_points
    return pairs[::step]

# ============================================================
# 1. True primes
# ============================================================

print(f"Generating first {NUM_PRIMES} true primes...")
primes_true = list(sp.primerange(2, sp.prime(NUM_PRIMES) + 1))
if len(primes_true) < NUM_PRIMES:
    raise ValueError("Prime generation failed.")

gaps_true = compute_gaps(primes_true)
chi_true = compute_chi(gaps_true)

print(f"True primes: {len(primes_true)} primes, {len(gaps_true)} gaps, {len(chi_true)} chi-values.")

# ============================================================
# 2. Random permutation of true gaps
# ============================================================

print("Generating one random permutation of true gaps...")
gaps_perm = gaps_true.copy()
random.shuffle(gaps_perm)
chi_perm = compute_chi(gaps_perm)

# ============================================================
# 3. Cramér pseudo-primes
# ============================================================

max_n = primes_true[-1]
print(f"Generating Cramér pseudo-primes up to n = {max_n}...")
cramer_primes = []
for n in range(2, max_n + 1):
    p = 1.0 / math.log(n) if n > 2 else 1.0
    if random.random() < p:
        cramer_primes.append(n)

if len(cramer_primes) < 10:
    raise RuntimeError("Cramér model produced too few pseudo-primes; adjust range.")

gaps_cramer = compute_gaps(cramer_primes)
chi_cramer = compute_chi(gaps_cramer)

print(f"Cramér: {len(cramer_primes)} pseudo-primes, {len(gaps_cramer)} gaps, {len(chi_cramer)} chi-values.")

# ============================================================
# 4. Build return-map pairs
# ============================================================

gap_pairs_true   = subsample_pairs(make_pairs(gaps_true),   MAX_POINTS_PLOT)
gap_pairs_perm   = subsample_pairs(make_pairs(gaps_perm),   MAX_POINTS_PLOT)
gap_pairs_cramer = subsample_pairs(make_pairs(gaps_cramer), MAX_POINTS_PLOT)

chi_pairs_true   = subsample_pairs(make_pairs(chi_true),   MAX_POINTS_PLOT)
chi_pairs_perm   = subsample_pairs(make_pairs(chi_perm),   MAX_POINTS_PLOT)
chi_pairs_cramer = subsample_pairs(make_pairs(chi_cramer), MAX_POINTS_PLOT)

# ============================================================
# 5. Plot gap return-map: (g_n, g_{n+1})
# ============================================================

plt.figure(figsize=(12, 6))
if gap_pairs_true:
    x_t, y_t = zip(*gap_pairs_true)
    plt.scatter(x_t, y_t, s=5, alpha=0.4, label="Primes gaps")
if gap_pairs_perm:
    x_r, y_r = zip(*gap_pairs_perm)
    plt.scatter(x_r, y_r, s=5, alpha=0.4, label="Random perm gaps")
if gap_pairs_cramer:
    x_c, y_c = zip(*gap_pairs_cramer)
    plt.scatter(x_c, y_c, s=5, alpha=0.4, label="Cramér gaps")

plt.xlabel("g_n")
plt.ylabel("g_{n+1}")
plt.title(f"Gap Return Map (g_n, g_{{n+1}}), N_primes={NUM_PRIMES}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 6. Plot curvature return-map: (chi_n, chi_{n+1})
# ============================================================

plt.figure(figsize=(12, 6))
if chi_pairs_true:
    x_t, y_t = zip(*chi_pairs_true)
    plt.scatter(x_t, y_t, s=5, alpha=0.4, label="Primes chi")
if chi_pairs_perm:
    x_r, y_r = zip(*chi_pairs_perm)
    plt.scatter(x_r, y_r, s=5, alpha=0.4, label="Random perm chi")
if chi_pairs_cramer:
    x_c, y_c = zip(*chi_pairs_cramer)
    plt.scatter(x_c, y_c, s=5, alpha=0.4, label="Cramér chi")

plt.xlabel("χ_n")
plt.ylabel("χ_{n+1}")
plt.title(f"Curvature Return Map (χ_n, χ_{{n+1}}), N_primes={NUM_PRIMES}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDone with return-map experiment.")
