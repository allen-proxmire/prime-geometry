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

# Drift approximation
delta_alpha = (g[1:] - g[:-1]) / (2 * pr[:-2])  # align arrays carefully

x = np.arange(len(delta_alpha))

plt.figure(figsize=(10,5))
plt.plot(x, delta_alpha, color='green', alpha=0.7)
plt.title("Angle Drift Approximation $\Delta\\alpha_n$")
plt.xlabel("Index $n$")
plt.ylabel(r"$\Delta\alpha_n$")
plt.tight_layout()
plt.savefig("figures_overview/PG_overview_Fig3_angle_drift.png", dpi=300)
plt.close()
