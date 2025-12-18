import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Utilities (same as before) ----------

os.makedirs("figures_overview", exist_ok=True)

def primes_up_to(N):
    sieve = np.ones(N+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(N**0.5)+1):
        if sieve[p]:
            sieve[p*p:N+1:p] = False
    return np.flatnonzero(sieve)

def prime_gaps(primes):
    return np.diff(primes)

def curvature(g):
    # χ_n = (g_{n+2} - g_n)/(g_n + g_{n+1})
    return (g[2:] - g[:-2]) / (g[:-2] + g[1:-1])

# ---------- Data ----------

pr = primes_up_to(2_000_000)
g = prime_gaps(pr)
chi = curvature(g)

# chi has length M-3 where M = len(pr)
n_chi = len(chi)

# Choose matching slices so everything has length n_chi
# Use interior segments aligned roughly with where χ_n is defined
pr_for = pr[2:2 + n_chi]      # length n_chi
g_for = g[1:1 + n_chi]        # length n_chi

tilde_g = g_for / np.log(pr_for)
tilde_chi = chi * np.log(pr_for)

x = np.arange(n_chi)

# ---------- Plot ----------

plt.figure(figsize=(10, 5))
plt.plot(x, tilde_g, label=r'$\tilde g_n$', alpha=0.7)
plt.plot(x, tilde_chi, label=r'$\tilde \chi_n$', alpha=0.7)
plt.title("Renormalized Signals Stabilize")
plt.xlabel("Index $n$")
plt.legend()
plt.tight_layout()

plt.savefig("figures_overview/PG_overview_Fig5_renormalized_signals.png", dpi=300)
plt.close()
