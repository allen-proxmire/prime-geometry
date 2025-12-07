import matplotlib.pyplot as plt
import csv

CSV_FILENAME = "PG6_Exp2_Concentration.csv"
PNG_FILENAME = "PG6_Fig2_Concentration.png"

tau = []
N = []
f_true = []
perm_mean = []
perm_sd = []

with open(CSV_FILENAME) as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        tau.append(float(row[0]))
        N.append(int(row[1]))
        f_true.append(float(row[2]))
        perm_mean.append(float(row[3]))
        perm_sd.append(float(row[4]))

plt.figure(figsize=(10,6))

# Plot each N as separate clusters
unique_N = sorted(set(N))

for N_val in unique_N:
    tau_vals = [tau[i] for i in range(len(tau)) if N[i] == N_val]
    f_vals   = [f_true[i] for i in range(len(f_true)) if N[i] == N_val]
    p_vals   = [perm_mean[i] for i in range(len(perm_mean)) if N[i] == N_val]
    sd_vals  = [perm_sd[i] for i in range(len(perm_sd)) if N[i] == N_val]

    plt.errorbar(
        tau_vals, p_vals,
        yerr=sd_vals,
        fmt='o--', capsize=4, label=f'Permutations (N={N_val})'
    )
    plt.plot(tau_vals, f_vals, 'o-', linewidth=2, label=f'True primes (N={N_val})')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$f_{\mathrm{true}}(\tau,N)$')
plt.title('Curvature Concentration: True Primes vs Permutation Models')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PNG_FILENAME, dpi=300)

print(f"Saved PG6 figure: {PNG_FILENAME}")
