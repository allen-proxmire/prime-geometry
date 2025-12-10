#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import isqrt
import os

# ============================================================
#  PRIME GENERATOR
# ============================================================

def generate_primes_upto(n):
    """Segmented sieve generating primes up to n safely."""
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
#  LOAD PRIMES + DEBUG
# ============================================================

MAX_PRIME = 3_000_000
print("Generating primes…")
p = np.array(generate_primes_upto(MAX_PRIME), dtype=np.int64)
print("Generated", len(p), "primes.")

print("\n=== DEBUG ===")
print("First 30 primes:", p[:30])

g = p[1:] - p[:-1]
print("First 30 gaps:", g[:30])

# ============================================================
#  GEOMETRIC QUANTITIES
# ============================================================

p_mid = p[1:-1]
g1 = g[:-1]
g2 = g[1:]

logp = np.log(p_mid)

chi = (g2 - g1) / (g1 + g2)
g_tilde = g1 / logp
dalpha_tilde = (g2 - g1) / logp

# normalized for colormap only
chi_norm = (chi - chi.min()) / (chi.max() - chi.min() + 1e-12)

# ============================================================
#  FIGURE 1 — Renormalization Collapse
# ============================================================

plt.figure(figsize=(10,5))
plt.hist(chi, bins=80, alpha=0.6, density=True, label="raw χₙ", color="tab:blue")
plt.hist(chi_norm, bins=80, alpha=0.5, density=True, label="χ̃ₙ (scaled)", color="tab:orange")
plt.yscale("log")
plt.xlabel("χ")
plt.ylabel("density")
plt.title("PG12 Fig 1 — Renormalization Collapse")
plt.legend()
plt.savefig("PG12_Fig1_RenormCollapse.png", dpi=300)
plt.close()


# ============================================================
#  FIGURE 2 — Renormalized Attractor
# ============================================================

plt.figure(figsize=(8,6))
plt.scatter(g_tilde[:-1], g_tilde[1:], s=1,
            c=chi_norm[:-1], cmap="viridis")
plt.xlabel("g̃ₙ")
plt.ylabel("g̃ₙ₊₁")
plt.title("PG12 Fig 2 — Renormalized Attractor")
plt.colorbar(label="normalized χₙ")
plt.savefig("PG12_Fig2_Attractor.png", dpi=300)
plt.close()


# ============================================================
#  FIGURE 3 — Coherence-Phase Alignment
# ============================================================

plt.figure(figsize=(12,4))
N = min(4000, len(chi))
plt.plot(chi[:N], lw=0.7, label="χₙ", color="tab:blue")
plt.plot(dalpha_tilde[:N] * 20, lw=0.7, label="Δα̃ₙ ×20", color="tab:orange")
plt.title("PG12 Fig 3 — Coherence-Phase Alignment")
plt.legend()
plt.savefig("PG12_Fig3_CoherencePhase.png", dpi=300)
plt.close()


# ============================================================
#  FIGURE 4 — Curvature Suppression
# ============================================================

plt.figure(figsize=(8,5))
plt.hist(chi, bins=80, density=True, alpha=0.8, color="tab:blue")
plt.xlabel("χₙ")
plt.ylabel("density")
plt.title("PG12 Fig 4 — Curvature Suppression")
plt.savefig("PG12_Fig4_CurvatureSuppression.png", dpi=300)
plt.close()


# ============================================================
#  FIGURE 5 — Curvature Spectrum
# ============================================================

fft_vals = np.abs(np.fft.rfft(chi - np.mean(chi)))
freqs = np.fft.rfftfreq(len(chi))

plt.figure(figsize=(12,4))
plt.semilogy(freqs[:4000], fft_vals[:4000], color="tab:blue")
plt.xlabel("frequency")
plt.ylabel("|FFT|")
plt.title("PG12 Fig 5 — Curvature Spectrum")
plt.savefig("PG12_Fig5_Spectrum.png", dpi=300)
plt.close()


# ============================================================
#  FIGURE 6 — Residual Structure
# ============================================================

residual = chi - (g2 - g1) / (g1 + g2 + 1e-12)

plt.figure(figsize=(12,4))
plt.plot(residual[:6000], lw=0.7, color="tab:green")
plt.title("PG12 Fig 6 — Residual Structure")
plt.xlabel("n")
plt.ylabel("residual")
plt.savefig("PG12_Fig6_Residuals.png", dpi=300)
plt.close()


print("\nAll PG12 figures generated successfully.")
