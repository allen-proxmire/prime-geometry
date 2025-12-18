import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from math import log
from mpl_toolkits.mplot3d import Axes3D

# -----------------------
# Generate primes
# -----------------------
N = 50000     # number of primes to use (adjust for speed)
primes = list(sp.primerange(2, 700000))[:N]
g = np.diff(primes)            # gaps g_n
p = np.array(primes[:-1])      # p_n aligned with g_n

# -----------------------
# Compute derived quantities
# -----------------------
# Curvature:
# chi_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
chi = (g[2:] - g[:-2]) / (g[1:-1] + g[:-2])

# Angle drift:
# Δα_n ≈ (g_{n+1} - g_n) / (2p_n)
dalpha = (g[1:] - g[:-1]) / (2 * p[:-1])

# Third-order smoothness:
# Θ_n = g_{n+3} - 3g_{n+2} + 3g_{n+1} - g_n
Theta = g[3:] - 3*g[2:-1] + 3*g[1:-2] - g[:-3]

# Renormalized
tilde_g = g / np.log(p)
tilde_chi = chi * np.log(p[1:-1])

# -----------------------
# FIGURE 1: Attractor plot
# -----------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g[:-2], g[1:-1], chi, s=2, alpha=0.5)
ax.set_xlabel("g_n")
ax.set_ylabel("g_{n+1}")
ax.set_zlabel("chi_n")
plt.title("PGME Attractor: (g_n, g_{n+1}, chi_n)")
plt.savefig("PG13_Fig1_Attractor.png", dpi=300)
plt.close()

# -----------------------
# FIGURE 2: Curvature coherence
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(chi, linewidth=0.5)
plt.title("Curvature χₙ Showing Coherence Phases")
plt.xlabel("n")
plt.ylabel("χ_n")
plt.savefig("PG13_Fig2_Coherence.png", dpi=300)
plt.close()

# -----------------------
# FIGURE 3: Angle drift
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(dalpha, linewidth=0.5)
plt.title("Angle Drift Δαₙ")
plt.xlabel("n")
plt.ylabel("Δα_n")
plt.savefig("PG13_Fig3_AngleDrift.png", dpi=300)
plt.close()

# -----------------------
# FIGURE 4: Renormalized gaps
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(tilde_g, linewidth=0.5)
plt.title("Renormalized Gaps 𝑔̃ₙ = gₙ / log(pₙ)")
plt.xlabel("n")
plt.ylabel("g̃_n")
plt.savefig("PG13_Fig4_RenormalizedGaps.png", dpi=300)
plt.close()

# -----------------------
# FIGURE 5: Renormalized curvature
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(tilde_chi, linewidth=0.5)
plt.title("Renormalized Curvature χ̃ₙ = (log pₙ) * χₙ")
plt.xlabel("n")
plt.ylabel("χ̃_n")
plt.savefig("PG13_Fig5_RenormalizedCurv.png", dpi=300)
plt.close()

# -----------------------
# FIGURE 6: Third-order smoothness
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(Theta, linewidth=0.5)
plt.title("Third-Order Smoothness Θₙ")
plt.xlabel("n")
plt.ylabel("Θ_n")
plt.savefig("PG13_Fig6_Theta.png", dpi=300)
plt.close()

# -----------------------
# Optional FIGURE 7: Forward simulation
# -----------------------
def simulate_pgme(g0, g1, chi0, steps=1000, C=0.0):
    """Toy PGME simulator: not exact, just demonstrates dynamics."""
    g_sim = [g0, g1]
    chi_sim = [chi0]

    for k in range(steps):
        g_n, g_n1 = g_sim[-2], g_sim[-1]
        chi_n = chi_sim[-1]

        # Simple curvature recurrence (approx PGME)
        g_n2 = g_n + (g_n + g_n1) * chi_n

        # update curvature
        chi_next = (g_n2 - g_n) / (g_n + g_n1 + 1e-9)

        g_sim.append(g_n2)
        chi_sim.append(chi_next)

    return np.array(g_sim), np.array(chi_sim)

g_sim, chi_sim = simulate_pgme(g[0], g[1], chi[0], steps=1000)

plt.figure(figsize=(10,4))
plt.plot(g_sim, label="Simulated", linewidth=1)
plt.plot(g[:1002], label="Actual", linewidth=1)
plt.legend()
plt.title("PGME Forward Simulation vs Real Primes (short range)")
plt.xlabel("n")
plt.ylabel("gap")
plt.savefig("PG13_Fig7_Simulation.png", dpi=300)
plt.close()
