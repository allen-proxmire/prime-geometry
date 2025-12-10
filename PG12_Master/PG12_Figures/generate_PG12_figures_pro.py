#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import isqrt
from scipy.stats import gaussian_kde
import os

# ============================================================
# PRIME GENERATOR
# ============================================================

def generate_primes_upto(n):
    sqrt_n = isqrt(n)
    sieve = np.ones(sqrt_n + 1, dtype=bool)
    sieve[:2] = False

    for i in range(2, isqrt(sqrt_n) + 1):
        if sieve[i]:
            sieve[i*i : sqrt_n+1 : i] = False

    small = np.nonzero(sieve)[0]
    primes = small.tolist()
    block = 100_000

    for low in range(sqrt_n + 1, n + 1, block):
        high = min(low + block - 1, n)
        segment = np.ones(high - low + 1, dtype=bool)

        for p in small:
            start = max(p*p, ((low + p - 1)//p) * p)
            segment[start - low : high - low + 1 : p] = False

        for i, isprime in enumerate(segment):
            if isprime:
                primes.append(low + i)

    return primes

# ============================================================
# LOAD PRIMES
# ============================================================

MAX_PRIME = 3_000_000
print("Generating primes…")
p = np.array(generate_primes_upto(MAX_PRIME), dtype=np.int64)
print("Generated", len(p), "primes.")

g = p[1:] - p[:-1]

# ============================================================
# RENORMALIZED VARIABLES
# ============================================================

p_mid = p[1:-1]
g1 = g[:-1]
g2 = g[1:]

logp = np.log(p_mid)

chi = (g2 - g1) / (g1 + g2)
g_tilde = g1 / logp
dalpha_tilde = (g2 - g1) / logp

chi_norm = (chi - chi.min()) / (chi.max() - chi.min() + 1e-12)

# ============================================================
# FIGURE 1 — KDE SMOOTHED RENORMALIZATION COLLAPSE
# ============================================================

plt.figure(figsize=(10,5))

# histogram
plt.hist(chi, bins=120, density=True, alpha=0.35, color="tab:blue", label="raw χₙ")

# KDE smoothing
xs = np.linspace(chi.min(), chi.max(), 800)
kde = gaussian_kde(chi)
plt.plot(xs, kde(xs), color="tab:blue", lw=1.5)

# normalized version
kde2 = gaussian_kde(chi_norm)
plt.plot(xs, kde2((xs - xs.min()) / (xs.max()-xs.min()+1e-9)),
         color="tab:orange", lw=1.5, label="renormalized χ̃ₙ")

plt.yscale("log")
plt.xlabel("χ")
plt.ylabel("density")
plt.title("PG12 Fig 1 — Renormalization Collapse")
plt.legend()
plt.savefig("PG12_Fig1_RenormCollapse.png", dpi=300)
plt.close()


# ============================================================
# FIGURE 2 — SUBSAMPLED, SMOOTHED ATTRACTOR
# ============================================================

plt.figure(figsize=(8,6))

# subsample for clarity
N = len(g_tilde)
idx = np.linspace(0, N-2, 60000).astype(int)

plt.scatter(
    g_tilde[idx],
    g_tilde[idx + 1],
    s=0.5,
    c=chi_norm[idx],
    cmap="viridis",
    alpha=0.6,
    edgecolors="none"
)

plt.xlabel("g̃ₙ")
plt.ylabel("g̃ₙ₊₁")
plt.title("PG12 Fig 2 — Renormalized Attractor")
plt.colorbar(label="normalized χₙ")
plt.savefig("PG12_Fig2_Attractor.png", dpi=300)
plt.close()


# ============================================================
# FIGURE 3 — SMOOTHED COHERENCE–PHASE
# ============================================================

plt.figure(figsize=(12,4))

Nshow = min(4000, len(chi))
window = 40  # smoothing window

# smooth χ
chi_smooth = np.convolve(chi[:Nshow], np.ones(window)/window, mode="same")
alpha_smooth = np.convolve(dalpha_tilde[:Nshow], np.ones(window)/window, mode="same")

plt.plot(chi_smooth, lw=1.0, label="smoothed χₙ", color="tab:blue")
plt.plot(alpha_smooth * 20, lw=1.0, label="smoothed Δα̃ₙ ×20", color="tab:orange")

plt.title("PG12 Fig 3 — Coherence–Phase Alignment")
plt.legend()
plt.savefig("PG12_Fig3_CoherencePhase.png", dpi=300)
plt.close()


# ============================================================
# FIGURE 4 — KDE-SMOOTHED CURVATURE SUPPRESSION
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(chi, bins=120, density=True, alpha=0.3, color="tab:blue")

xs = np.linspace(-1, 1, 800)
kde = gaussian_kde(chi)
plt.plot(xs, kde(xs), color="tab:blue", lw=2)

plt.xlabel("χₙ")
plt.ylabel("density")
plt.title("PG12 Fig 4 — Curvature Suppression")
plt.savefig("PG12_Fig4_CurvatureSuppression.png", dpi=300)
plt.close()


# ============================================================
# FIGURE 5 — CLEANED SPECTRUM
# ============================================================

fft_vals = np.abs(np.fft.rfft(chi - np.mean(chi)))
freqs = np.fft.rfftfreq(len(chi))

plt.figure(figsize=(12,4))
plt.semilogy(freqs[:5000], fft_vals[:5000], color="tab:blue", lw=0.8)
plt.xlabel("frequency")
plt.ylabel("|FFT|")
plt.title("PG12 Fig 5 — Curvature Spectrum")
plt.savefig("PG12_Fig5_Spectrum.png", dpi=300)
plt.close()


# ============================================================
# FIGURE 6 — PLACEHOLDER RESIDUAL (WILL BE REAL IN STAGE D)
# ============================================================

residual = chi - (g2 - g1) / (g1 + g2 + 1e-12)

plt.figure(figsize=(12,4))
plt.plot(residual[:6000], lw=0.6, color="tab:green")
plt.title("PG12 Fig 6 — Residual Structure (placeholder)")
plt.xlabel("n")
plt.ylabel("residual")
plt.savefig("PG12_Fig6_Residuals.png", dpi=300)
plt.close()


print("\nPG12 refined figures created.")
