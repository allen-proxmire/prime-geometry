import sympy as sp
import csv

# ----- PARAMETERS -----
MAX_PRIME = 200000   # Increase later if you want
STRIDE = 500         # How often to sample M(N)
CSV_FILENAME = "PG6_Exp1_Output.csv"

# ----- GET PRIMES AND GAPS -----
primes = list(sp.primerange(2, MAX_PRIME))
gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

# ----- COMPUTE CURVATURE -----
chi = []
weights = []
for n in range(len(gaps)-2):
    g0, g1, g2 = gaps[n], gaps[n+1], gaps[n+2]
    denom = g0 + g1
    weights.append(denom)
    if denom == 0:
        chi.append(0)
    else:
        chi.append((g2 - g0) / denom)

# ----- WEIGHTED MEAN M(N) -----
M = []
num = 0.0
den = 0.0
for w, c in zip(weights, chi):
    num += w * c
    den += w
    M.append(num / den)

# ----- WRITE CSV FILE -----
with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "M(N)"])
    for i in range(0, len(M), STRIDE):
        writer.writerow([i+1, M[i]])

# ----- ALSO PRINT TO SCREEN -----
print(f"Saved CSV to: {CSV_FILENAME}")
