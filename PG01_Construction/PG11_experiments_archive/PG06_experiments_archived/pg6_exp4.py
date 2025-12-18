import sympy as sp
import random
import csv
from statistics import mean, stdev

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
MAX_PRIME = 2_000_000       # deep
TAU_VALUES = [0.05, 0.10, 0.20]
K_VALUES   = [0.25, 0.50, 1.00]
N_PERMS = 200
CSV_FILENAME = "PG6_Exp4_Predictive.csv"

# ----------------------------------------------------
# Step 1: primes + gaps
# ----------------------------------------------------
print(f"Generating primes up to {MAX_PRIME:,} ...")
primes = list(sp.primerange(2, MAX_PRIME))
print(f"  Found {len(primes):,} primes.")

gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
print(f"  Computed {len(gaps):,} gaps.\n")

# ----------------------------------------------------
# Step 2: curvature
# ----------------------------------------------------
def curvature_from_gaps(g):
    chi = []
    for n in range(len(g)-2):
        g0, g1, g2 = g[n], g[n+1], g[n+2]
        denom = g0 + g1
        if denom == 0:
            chi.append(0.0)
        else:
            chi.append((g2 - g0) / denom)
    return chi

print("Computing curvature for true sequence...")
chi_true = curvature_from_gaps(gaps)

# ----------------------------------------------------
# Step 3: heuristic hit function
# ----------------------------------------------------
def hit_rate(gaps, chi, tau, k):
    hits = 0
    total = 0
    for n in range(len(chi)):
        if abs(chi[n]) < tau:
            g0 = gaps[n]
            g2 = gaps[n+2]
            if g0 > 0:
                total += 1
                if abs(g2 - g0) / g0 <= k:
                    hits += 1
    if total == 0:
        return float("nan")
    return hits / total

# ----------------------------------------------------
# Step 4: compute true hit rates
# ----------------------------------------------------
print("Computing true predictive hit rates...\n")
true_rates = {}
for tau in TAU_VALUES:
    for k in K_VALUES:
        true_rates[(tau,k)] = hit_rate(gaps, chi_true, tau, k)

# ----------------------------------------------------
# Step 5: permutation sampling
# ----------------------------------------------------
def permutation_samples(gaps, tau_vals, k_vals, n_perms):
    rng = random.Random(24680)
    base = gaps[:]
    results = { (tau,k): [] for tau in tau_vals for k in k_vals }

    for p in range(n_perms):
        rng.shuffle(base)
        chi = curvature_from_gaps(base)

        for tau in tau_vals:
            for k in k_vals:
                results[(tau,k)].append(hit_rate(base, chi, tau, k))

        if (p+1) % 20 == 0:
            print(f"  Completed permutation {p+1}/{n_perms}")

    return results

print("Running permutation sampling...")
perm_results = permutation_samples(gaps, TAU_VALUES, K_VALUES, N_PERMS)
print("Permutation sampling complete.\n")

# ----------------------------------------------------
# Step 6: write CSV
# ----------------------------------------------------
print(f"Saving results to {CSV_FILENAME}...")

with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tau", "k", "hit_true", "perm_mean", "perm_sd"])

    for tau in TAU_VALUES:
        for k in K_VALUES:
            hit_t = true_rates[(tau,k)]
            vals = perm_results[(tau,k)]
            mu = mean(vals)
            sd = stdev(vals)
            writer.writerow([tau, k, hit_t, mu, sd])

print("Done.")
