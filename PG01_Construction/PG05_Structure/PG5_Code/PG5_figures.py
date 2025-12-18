import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from math import atan

# ================================================================
# 1. PURE PYTHON PRIME GENERATOR
# ================================================================

def simple_primes_upto(n):
    """Fast sieve of Eratosthenes returning primes up to n."""
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i : n+1 : i] = False
    return np.flatnonzero(sieve)

# Adjust to your machine — 2,000,000 runs comfortably on most PCs
N = 2_000_000

print("Generating primes...")
p = simple_primes_upto(N)
g = np.diff(p)
print(f"Generated {len(p)} primes.")

# ================================================================
# 2. CURVATURE & RELATED SIGNALS
# ================================================================

# χ_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
chi = (g[2:] - g[:-2]) / (g[:-2] + g[1:-1])

def smooth(x, W):
    return np.convolve(x, np.ones(W)/W, mode='valid')

chi_500  = smooth(chi, 500)
chi_2000 = smooth(chi, 2000)
chi_5000 = smooth(chi, 5000)

# Prime Triangle angles αₙ = arctan(p_n / p_{n+1})
alpha = np.arctan(p[:-1] / p[1:])
delta_alpha = np.diff(alpha)

# Align Δαₙ to match χ_n indexing
delta_alpha = delta_alpha[2:-1]

# Padding in case smoothing reduces lengths
min_len = min(len(chi_2000), len(delta_alpha))

# ================================================================
# 3. POWER SPECTRA
# ================================================================

def power_spectrum(x):
    freqs, psd = welch(x, nperseg=4096, scaling='density')
    return freqs, psd

# Spectrum of χ_n
freqs, psd_true = power_spectrum(chi)

# Permutation baseline
num_perm = 20
psd_perm_list = []

print("Computing permutation reference spectra...")
for _ in range(num_perm):
    perm = np.random.permutation(chi)
    _, psd_p = power_spectrum(perm)
    psd_perm_list.append(psd_p)

psd_perm_mean = np.mean(psd_perm_list, axis=0)

# ================================================================
# FIGURE 1 — Power Spectrum of χₙ
# ================================================================

plt.figure(figsize=(9,6))
plt.loglog(freqs, psd_true, label='True primes', linewidth=2)
plt.loglog(freqs, psd_perm_mean, linestyle='--', label='Permutation mean')
plt.title("Power Spectrum of Curvature χₙ")
plt.xlabel("Frequency")
plt.ylabel("Power Spectral Density")
plt.legend()
plt.tight_layout()
plt.savefig("PG5_Fig1_chi_spectrum.png", dpi=300)
plt.close()

# ================================================================
# FIGURE 2 — Power Spectrum of Smoothed χₙ (W=2000)
# ================================================================

freqs2, psd_smooth = power_spectrum(chi_2000)

plt.figure(figsize=(9,6))
plt.loglog(freqs2, psd_smooth, linewidth=2, label="χₙ smoothed (W=2000)")
plt.title("Power Spectrum of Smoothed Curvature χₙ^{(2000)}")
plt.xlabel("Frequency")
plt.ylabel("Power Spectral Density")
plt.legend()
plt.tight_layout()
plt.savefig("PG5_Fig2_chi2000_spectrum.png", dpi=300)
plt.close()

# ================================================================
# FIGURE 3 — Angle Drift Δαₙ
# ================================================================

plt.figure(figsize=(10,5))
plt.plot(delta_alpha, linewidth=0.7)
plt.title("Angle Drift Δαₙ = αₙ₊₁ - αₙ")
plt.xlabel("n")
plt.ylabel("Δαₙ (radians)")
plt.tight_layout()
plt.savefig("PG5_Fig3_angle_drift.png", dpi=300)
plt.close()

# ================================================================
# FIGURE 4 — χₙ^{(2000)} vs Δαₙ (normalized)
# ================================================================

def normalize(x):
    return x / np.max(np.abs(x))

plt.figure(figsize=(10,6))
plt.plot(normalize(chi_2000[:min_len]), label="Normalized χₙ^{(2000)}", linewidth=1.2)
plt.plot(normalize(delta_alpha[:min_len]), label="Normalized Δαₙ", linewidth=1.0)
plt.title("Comparison of Smoothed Curvature and Angle Drift")
plt.xlabel("n")
plt.legend()
plt.tight_layout()
plt.savefig("PG5_Fig4_chi2000_vs_deltaalpha.png", dpi=300)
plt.close()

# ================================================================
# 5. HIGHER-ORDER SIGNALS: Aₙ and K₂(n)
# ================================================================

# A_n = (g_{n+2} - 2*g_{n+1} + g_n) / (g_n + g_{n+1})
A = (g[2:] - 2*g[1:-1] + g[:-2]) / (g[:-2] + g[1:-1])

# E_n ≈ sqrt(2)/2 * (g_n + g_{n+1})
E = (np.sqrt(2)/2) * (g[:-1] + g[1:])
K2 = E[2:] - 2*E[1:-1] + E[:-2]

# Permutation reference for A_n
print("Computing permutation reference for Aₙ…")
A_perm = []
for _ in range(num_perm):
    A_perm.append(np.random.permutation(A))
A_perm = np.concatenate(A_perm)

# ================================================================
# FIGURE 5 — Histogram of Aₙ vs permutation model
# ================================================================

plt.figure(figsize=(9,6))
plt.hist(A, bins=200, density=True, alpha=0.6, label="True primes")
plt.hist(A_perm, bins=200, density=True, alpha=0.4, label="Permutation")
plt.xlim([-1, 1])
plt.title("Distribution of Gap-Acceleration Ratio Aₙ")
plt.xlabel("Aₙ")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("PG5_Fig5_A_hist.png", dpi=300)
plt.close()

# ================================================================
# FIGURE 6 — Histogram of K₂(n)
# ================================================================

plt.figure(figsize=(9,6))
plt.hist(K2, bins=200, density=True, alpha=0.65)
plt.title("Distribution of Second-Order Curvature K₂(n)")
plt.xlabel("K₂(n)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("PG5_Fig6_K2_hist.png", dpi=300)
plt.close()

print("All PG5 figures successfully generated!")
