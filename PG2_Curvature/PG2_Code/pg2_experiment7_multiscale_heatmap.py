# ============================================================
# File: pg2_experiment7_multiscale_heatmap.py
# Prime Geometry II — Experiment 7:
# Multi-scale heatmap of curvature χ_n and L_n = χ_n^2
# across different window sizes.
#
# Requirements: sympy, numpy, matplotlib
#   pip install sympy numpy matplotlib
# ============================================================

import sympy as sp
import numpy as np
import statistics
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

NUM_PRIMES = 50000   # can bump to 100000 if you're patient

# Window sizes (in χ/L indices) – choose a nice spread of scales
WINDOW_SIZES = [200, 500, 1000, 2000, 5000]

# ============================================================
# Helpers
# ============================================================

def compute_chi_and_L(gap_list):
    """
    From a list of gaps g_n, compute:
      χ_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
      L_n = χ_n^2
    for n = 0 .. len(gaps)-3.
    """
    chi_values = []
    L_values = []
    m = len(gap_list)
    for n in range(m - 2):
        g0 = gap_list[n]
        g1 = gap_list[n+1]
        g2 = gap_list[n+2]
        denom = g0 + g1
        if denom == 0:
            continue
        chi = (g2 - g0) / denom
        chi_values.append(chi)
        L_values.append(chi * chi)
    return chi_values, L_values

def sliding_window_means(values, window_size):
    """
    Return list of means of consecutive windows of length W:
      mean(values[i:i+W]) for i = 0 .. len(values)-W
    """
    n = len(values)
    if window_size > n:
        raise ValueError("Window size larger than sequence length.")
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    means = []
    for i in range(n - window_size + 1):
        s = prefix[i + window_size] - prefix[i]
        means.append(s / window_size)
    return means

# ============================================================
# 1. Generate primes and gaps
# ============================================================

print(f"Generating first {NUM_PRIMES} primes...")
primes = list(sp.primerange(2, sp.prime(NUM_PRIMES) + 1))
if len(primes) < NUM_PRIMES:
    raise ValueError("Prime generation failed.")

gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]

# ============================================================
# 2. Compute χ_n and L_n
# ============================================================

print("Computing χ_n and L_n...")
chi, L = compute_chi_and_L(gaps)

print("\n=== GLOBAL STATS ===")
print(f"Number of χ_n values: {len(chi)}")
print(f"Mean χ:   {statistics.mean(chi)}")
print(f"Var χ:    {statistics.pvariance(chi)}")
print(f"Max |χ|:  {max(abs(c) for c in chi)}")
print(f"Mean L:   {statistics.mean(L)}")

# ============================================================
# 3. Sliding-window means at multiple scales
# ============================================================

print("\nComputing sliding-window means at multiple scales...")

window_means_L = {}
window_means_chi = {}

for W in WINDOW_SIZES:
    if W > len(L):
        continue
    print(f"  Window size W = {W}...")
    window_means_L[W] = sliding_window_means(L, W)
    window_means_chi[W] = sliding_window_means(chi, W)

# To make a rectangular matrix for imshow, we truncate all
# window-mean series to the same minimum length.
min_len = min(len(v) for v in window_means_L.values())

Ws_sorted = sorted(window_means_L.keys())

heat_L = np.array([window_means_L[W][:min_len] for W in Ws_sorted])
heat_chi = np.array([window_means_chi[W][:min_len] for W in Ws_sorted])

# x-axis indices (window start)
x_vals = np.arange(min_len)

# y-axis is window size; we'll map row index -> WINDOW_SIZES
y_vals = np.array(Ws_sorted)

# ============================================================
# 4. Plot heatmap for L (curvature magnitude / action density)
# ============================================================

plt.figure(figsize=(12, 6))
extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
plt.imshow(
    heat_L,
    aspect='auto',
    origin='lower',
    extent=extent,
    cmap='viridis'
)
plt.colorbar(label="Mean L in window")
plt.title(f"Multi-Scale Heatmap of Mean L (N_primes={NUM_PRIMES})")
plt.xlabel("Window start index n")
plt.ylabel("Window size W (rows from bottom to top)")
plt.yticks(y_vals)  # label rows by actual window sizes
plt.tight_layout()
plt.show()

# ============================================================
# 5. Plot heatmap for χ (signed curvature direction)
# ============================================================

plt.figure(figsize=(12, 6))
# Use a diverging colormap to show sign of χ
plt.imshow(
    heat_chi,
    aspect='auto',
    origin='lower',
    extent=extent,
    cmap='coolwarm'
)
plt.colorbar(label="Mean χ in window")
plt.title(f"Multi-Scale Heatmap of Mean χ (N_primes={NUM_PRIMES})")
plt.xlabel("Window start index n")
plt.ylabel("Window size W (rows from bottom to top)")
plt.yticks(y_vals)
plt.tight_layout()
plt.show()

print("\nDone with multi-scale heatmap experiment.")
