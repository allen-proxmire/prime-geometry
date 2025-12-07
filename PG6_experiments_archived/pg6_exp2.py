import sympy as sp
import random
import csv
from statistics import mean, stdev

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
MAX_PRIME = 2_000_000     # deep run (you can change to 1_000_000)
TAU_VALUES = [0.05, 0.10, 0.20]
N_VALUES   = [50_000, 75_000, 100_000]   # sample sizes
N_PERMS    = 200          # permutations per (tau, N)
CSV_FILENAME = "PG6_Exp2_Concentration.csv"

# ----------------------------------------------------
# Step 1: Compute primes and gaps
# ----------------------------------------------------
print(f"Generating primes up to {MAX_PRIME:,} ...")
primes = list(sp.primerange(2, MAX_PRIME))
print(f"  Found {len(primes):,} primes.")

gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
print(f"  Computed {len(gaps):,} gaps.\n")

# ----------------------------------------------------
# Step 2: Compute curvature chi_n for true sequence
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

print("Computing curvature for true primes...")
chi_true = curvature_from_gaps(gaps)

# ----------------------------------------------------
# Step 3: Fraction of small-curvature sites
# ----------------------------------------------------
def f_small(chi, tau, N):
    N = min(N, len(chi))
    return sum(1 for c in chi[:N] if abs(c) < tau) / N


# ----------------------------------------------------
# Step 4: Run permutation sampling
# ----------------------------------------------------
def permutation_samples(g, tau_vals, N_vals, n_perms):
    rng = random.Random(123456)
    base = g[:]
    results = { (tau, N): [] for tau in tau_vals for N in N_vals }

    for p in range(n_perms):
        rng.shuffle(base)
        chi_pi = curvature_from_gaps(base)

        for tau in tau_vals:
            for N in N_vals:
                results[(tau, N)].append(f_small(chi_pi, tau, N))

        if (p+1) % 20 == 0:
            print(f"  Completed permutation {p+1}/{n_perms}")

    return results


print("Running permutation sampling...")
perm_results = permutation_samples(gaps, TAU_VALUES, N_VALUES, N_PERMS)
print("Permutation sampling complete.\n")

# ----------------------------------------------------
# Step 5: Write CSV results
# ----------------------------------------------------
print(f"Saving results to {CSV_FILENAME} ...")
with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tau", "N", "f_true", "perm_mean", "perm_sd"])

    for tau in TAU_VALUES:
        for N in N_VALUES:
            f_true = f_small(chi_true, tau, N)
            vals = perm_results[(tau, N)]
            mu   = mean(vals)
            sd   = stdev(vals)
            writer.writerow([tau, N, f_true, mu, sd])

print("Done.")
