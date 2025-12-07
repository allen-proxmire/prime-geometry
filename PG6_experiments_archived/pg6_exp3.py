import sympy as sp
import random
import csv
from statistics import mean, stdev

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
MAX_PRIME = 2_000_000     # deep run
N_VALUES = [50_000, 75_000, 100_000, 125_000]  # sample window sizes
N_PERMS = 200
CSV_FILENAME = "PG6_Exp3_LowAction.csv"

# ----------------------------------------------------
# Step 1: Compute primes and gaps
# ----------------------------------------------------
print(f"Generating primes up to {MAX_PRIME:,} ...")
primes = list(sp.primerange(2, MAX_PRIME))
print(f"  Found {len(primes):,} primes.")

gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
print(f"  Computed {len(gaps):,} gaps.\n")

# ----------------------------------------------------
# Step 2: Compute curvature χ_n for the true sequence
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
# Step 3: Cumulative action S(N)
# ----------------------------------------------------
def S_value(chi, N):
    return sum(c*c for c in chi[:N])

# ----------------------------------------------------
# Step 4: Permutation sampling
# ----------------------------------------------------
def permutation_samples(g, N_values, n_perms):
    rng = random.Random(987654)
    base = g[:]
    results = {N: [] for N in N_values}

    for p in range(n_perms):
        rng.shuffle(base)
        chi_pi = curvature_from_gaps(base)
        for N in N_values:
            results[N].append(S_value(chi_pi, N))

        if (p+1) % 20 == 0:
            print(f"  Completed permutation {p+1}/{n_perms}")

    return results

print("Running permutation sampling...")
perm_results = permutation_samples(gaps, N_VALUES, N_PERMS)
print("Permutation sampling complete.\n")

# ----------------------------------------------------
# Step 5: Write CSV output
# ----------------------------------------------------
print(f"Saving results to {CSV_FILENAME} ...")

with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "S_true", "perm_mean", "perm_sd", "ratio"])

    for N in N_VALUES:
        S_true_N = S_value(chi_true, N)
        vals = perm_results[N]
        mu = mean(vals)
        sd = stdev(vals)
        ratio = S_true_N / mu
        writer.writerow([N, S_true_N, mu, sd, ratio])

print("Done.")
