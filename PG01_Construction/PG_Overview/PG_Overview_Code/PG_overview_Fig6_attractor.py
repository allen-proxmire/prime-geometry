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

pr = primes_up_to(1_000_000)
g = prime_gaps(pr)
chi = curvature(g)

# Align lengths
g0 = g[:-2]
g1 = g[1:-1]
chiA = chi

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g0, g1, chiA, s=1, alpha=0.4)

ax.set_xlabel("$g_n$")
ax.set_ylabel("$g_{n+1}$")
ax.set_zlabel("$\chi_n$")
ax.set_title("PGME / PGEE Attractor Geometry")

plt.tight_layout()
plt.savefig("figures_overview/PG_overview_Fig6_attractor.png", dpi=300)
plt.close()
