import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# PARAMETERS
# ===============================
N_PRIMES = 300000
SAVE_DPI = 300

# ===============================
# LOAD VERIFIED PRIME LIST
# ===============================
print("Loading primes...")
p = np.loadtxt("primes1e6.txt", dtype=np.int64)
p = p[:N_PRIMES + 5]
print("Loaded", len(p), "primes.")

# ===============================
# BASIC ARRAYS
# ===============================
g = p[1:] - p[:-1]
alpha = np.arctan(p[:-1] / p[1:])
chi = (g[2:] - g[:-2]) / (g[:-2] + g[1:-1])
Theta = g[3:] - 3*g[2:-1] + 3*g[1:-2] - g[:-3]
adev = np.abs(alpha - np.pi/4)

# ===============================
# CUMULATIVE CURVATURE IMBALANCE
# ===============================
imb = chi[:-1] * (g[1:-2] + g[2:-1])
C = np.cumsum(imb)

# ===============================
# STABILITY BOUND
# ===============================
p_for_bound = p[1:1 + len(chi)]
bound = 0.5 * np.cumsum(np.abs(chi) / p_for_bound)

# ===============================
# FIGURE 1 — Stability Inequality
# ===============================
plt.figure(figsize=(12,5))
plt.plot(adev[2:2+len(bound)], label="|alpha - pi/4|")
plt.plot(bound, label="Bound", alpha=0.7)
plt.yscale("log")
plt.title("PG7 Fig 1 — Stability Bound")
plt.legend()
plt.tight_layout()
plt.savefig("PG7_Fig1_stability_bound.png", dpi=SAVE_DPI)
plt.close()

# ===============================
# FIGURE 2 — Curvature Imbalance
# ===============================
plt.figure(figsize=(12,5))
plt.plot(C)
plt.axhline(0, color="black", linestyle="--")
plt.title("PG7 Fig 2 — Curvature Imbalance")
plt.tight_layout()
plt.savefig("PG7_Fig2_curvature_balance.png", dpi=SAVE_DPI)
plt.close()

# ===============================
# FIGURE 3 — Theta Distribution
# ===============================
perm = np.random.permutation(g[:-3])
Theta_perm = perm[3:] - 3*perm[2:-1] + 3*perm[1:-2] - perm[:-3]

plt.figure(figsize=(10,5))
plt.hist(Theta, bins=200, density=True, alpha=0.7, label="True primes")
plt.hist(Theta_perm, bins=200, density=True, alpha=0.4, label="Permutation")
plt.title("PG7 Fig 3 — Theta Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("PG7_Fig3_theta_distribution.png", dpi=SAVE_DPI)
plt.close()

# ===============================
# FIGURE 4 — Smoothed Theta
# ===============================
def smooth(x, W):
    return np.convolve(x, np.ones(W)/W, mode="valid")

plt.figure(figsize=(12,6))
for W in [100, 250, 500]:
    plt.plot(smooth(Theta, W), label=f"W={W}")

plt.title("PG7 Fig 4 — Smoothed Theta")
plt.legend()
plt.tight_layout()
plt.savefig("PG7_Fig4_theta_smoothed.png", dpi=SAVE_DPI)
plt.close()

# ===============================
# FIGURE 5 — Attractor
# ===============================
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

gn = g[:-3]
gn1 = g[1:-2]
chin = chi[:-1]

ax.scatter(gn, gn1, chin, s=1, alpha=0.25)
ax.set_xlabel("g_n")
ax.set_ylabel("g_{n+1}")
ax.set_zlabel("chi_n")
ax.set_title("PG7 Fig 5 — Attractor")

plt.tight_layout()
plt.savefig("PG7_Fig5_attractor_3d.png", dpi=SAVE_DPI)
plt.close()

print("All PG7 figures created successfully!")
