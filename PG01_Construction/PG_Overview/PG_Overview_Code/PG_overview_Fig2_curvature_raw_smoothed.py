import numpy as np
import matplotlib.pyplot as plt
import os

# Make output folder
os.makedirs("figures_overview", exist_ok=True)

# Simple Sieve of Eratosthenes
def primes_up_to(N):
    sieve = np.ones(N+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(N**0.5)+1):
        if sieve[p]:
            sieve[p*p:N+1:p] = False
    return np.flatnonzero(sieve)

# Compute gaps
def prime_gaps(primes):
    return np.diff(primes)

# Normalized curvature χ_n
def curvature(g):
    # g = array of gaps
    # χ_n = (g_{n+2} - g_n)/(g_n + g_{n+1})
    return (g[2:] - g[:-2]) / (g[:-2] + g[1:-1])

# Smoothed curvature
def smooth(x, W=500):
    return np.convolve(x, np.ones(W)/W, mode='valid')

pr = primes_up_to(2_000_000)
g = prime_gaps(pr)
chi = curvature(g)

chi_sm = smooth(chi, W=1000)
x = np.arange(len(chi))
xs = np.arange(len(chi_sm))

plt.figure(figsize=(10,5))
plt.plot(x, chi, color='gray', alpha=0.5, label='Raw Curvature $\chi_n$')
plt.plot(xs, chi_sm, color='blue', linewidth=2, label='Smoothed $\chi^{(1000)}_n$')

plt.title("Curvature Signal: Raw and Smoothed")
plt.xlabel("Index $n$")
plt.ylabel(r"$\chi_n$")
plt.legend()
plt.tight_layout()
plt.savefig("figures_overview/PG_overview_Fig2_curvature_raw_smoothed.png", dpi=300)
plt.close()
