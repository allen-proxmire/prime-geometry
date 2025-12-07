import matplotlib.pyplot as plt
import csv

CSV_FILENAME = "PG6_Exp3_LowAction.csv"
PNG_FILENAME = "PG6_Fig3_LowAction.png"

N = []
S_true = []
perm_mean = []
perm_sd = []
ratio = []

with open(CSV_FILENAME) as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        N.append(int(row[0]))
        S_true.append(float(row[1]))
        perm_mean.append(float(row[2]))
        perm_sd.append(float(row[3]))
        ratio.append(float(row[4]))

plt.figure(figsize=(10,6))

plt.plot(N, S_true, "o-", linewidth=2, label="True primes")
plt.plot(N, perm_mean, "o--", linewidth=2, label="Permutation mean")

plt.xlabel("N")
plt.ylabel("Cumulative action $S(N)$")
plt.title("Low-Action Structure: True Primes vs Permutations")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PNG_FILENAME, dpi=300)

print(f"Saved figure: {PNG_FILENAME}")
