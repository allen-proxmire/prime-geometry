import os
import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

# ---------------------------------------
# CONFIG
# ---------------------------------------
LIMIT = 5_000_000        # primes up to this value
OUT_DIR = "PG6_Figures"  # output folder for PNGs

plt.rcParams["figure.dpi"] = 160
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 10

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------
# PRIME GENERATION (SymPy)
# ---------------------------------------
print(f"Generating primes up to {LIMIT} using sympy...")
p = np.array(list(primerange(1, LIMIT + 1)), dtype=np.int64)
print(f"Generated {len(p)} primes.")

# ---------------------------------------
# BASIC ARRAYS: gaps, curvature, angles
# ---------------------------------------
g = np.diff(p)  # gaps
N = len(g)

# curvature chi_n = (g_{n+2} - g_n)/(g_n + g_{n+1})
chi = (g[2:] - g[:-2]) / (g[1:-1] + g[:-2])  # length N-2

# Prime Triangle angles: alpha_n = arctan(p_n / p_{n+1})
alpha = np.arctan(p[:-1] / p[1:].astype(np.float64))  # length N
delta_alpha = np.diff(alpha)  # length N-1

print("Arrays:")
print(f"  g:           {g.shape}")
print(f"  chi:         {chi.shape}")
print(f"  alpha:       {alpha.shape}")
print(f"  delta_alpha: {delta_alpha.shape}")

# ---------------------------------------
# CONTINUATION MODELS
# ---------------------------------------

g_pred_chi = np.full_like(g, np.nan, dtype=float)
g_pred_joint = np.full_like(g, np.nan, dtype=float)

W = 50  # smoothing window for curvature estimate

print("Building continuation predictions...")
for n in range(W, len(chi)):
    chi_hat = chi[n-W+1:n+1].mean()

    g_pred = g[n] + chi_hat * (g[n] + g[n+1])
    idx = n + 2
    if idx < len(g):
        g_pred_chi[idx] = g_pred

        delta_g = g[n+1] - g[n]
        g_pred_joint[idx] = g[n] + 0.5 * delta_g + chi_hat * (g[n] + g[n+1])

mask = ~np.isnan(g_pred_chi)
valid_idx = np.where(mask)[0]

errors_chi = g_pred_chi[mask] - g[mask]
errors_joint = g_pred_joint[mask] - g[mask]

print(f"Valid prediction points: {len(valid_idx)}")

def safe_window_indices(indices, offset=1000, length=300):
    if len(indices) == 0:
        raise ValueError("No valid indices for plotting.")
    start_pos = offset if len(indices) > offset + length else 0
    start = indices[start_pos]
    end = min(start + length, indices[-1] + 1)
    return start, end

# ========================================================
# FIGURE 1
# ========================================================
print("Making Figure 1...")
start, end = safe_window_indices(valid_idx, offset=2000, length=300)
L = end - start
x = np.arange(L)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(x, g[start:end], label="True gaps", lw=1.5)
ax.plot(x, g_pred_chi[start:end], label="Predicted (curvature-only)", lw=1.2)

ax.set_title("PG6 Fig 1 — True vs Predicted Gaps (Curvature-Only)")
ax.set_xlabel("Index offset")
ax.set_ylabel("Gap size")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig1_true_vs_curvature_prediction.png"))
plt.close()

# ========================================================
# FIGURE 2
# ========================================================
print("Making Figure 2...")
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(np.abs(errors_chi), label="Curvature-only error", lw=0.7)
ax.plot(np.abs(errors_joint), label="Curvature + angle error", lw=0.7)

ax.set_title("PG6 Fig 2 — Absolute Error Accumulation")
ax.set_xlabel("Prediction index (valid region)")
ax.set_ylabel("|error|")
ax.set_yscale("log")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig2_error_accumulation.png"))
plt.close()

# ========================================================
# FIGURE 3
# ========================================================
print("Making Figure 3...")
sign_err = np.sign(errors_chi)

fig, ax = plt.subplots(figsize=(7, 3))
window = min(5000, len(sign_err))
ax.plot(sign_err[:window], lw=0.5)
ax.set_title("PG6 Fig 3 — Sign of Error (Curvature-Only Continuation)")
ax.set_xlabel(f"Prediction index (first {window} points)")
ax.set_ylabel("sign(error)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig3_error_sign_persistence.png"))
plt.close()

# ========================================================
# FIGURE 4
# ========================================================
print("Making Figure 4...")
xvals = (g[1:] - g[:-1]) / (2.0 * p[:-2])
yvals = delta_alpha

fig, ax = plt.subplots(figsize=(4, 4))
lo = 1000
hi = min(50_000, len(xvals))
ax.scatter(xvals[lo:hi], yvals[lo:hi], s=1, alpha=0.3)

ax.set_title("PG6 Fig 4 — Angle Drift vs First-Order Gap Variation")
ax.set_xlabel("(g[n+1] - g[n]) / (2 p[n])")
ax.set_ylabel("Δα_n")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig4_angle_drift_vs_gap_diff.png"))
plt.close()

# ========================================================
# FIGURE 5
# ========================================================
print("Making Figure 5...")
gap_accel = g[2:] - g[:-2]
curv_scaled = chi * (g[1:-1] + g[:-2])

fig, ax = plt.subplots(figsize=(4, 4))
lo = 1000
hi = min(50_000, len(gap_accel))
ax.scatter(gap_accel[lo:hi], curv_scaled[lo:hi], s=1, alpha=0.3)

ax.set_title("PG6 Fig 5 — Curvature Term vs True Gap Acceleration")
ax.set_xlabel("g[n+2] - g[n]")
ax.set_ylabel("χ_n (g_n + g_{n+1})")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig5_curvature_vs_gap_acceleration.png"))
plt.close()

# ========================================================
# FIGURE 6
# ========================================================
print("Making Figure 6...")
Lcommon = min(len(chi), len(delta_alpha))
chi_trim = chi[:Lcommon]
da_trim = delta_alpha[:Lcommon]

idx0 = 200_000
Lwin = 3000
if idx0 + Lwin > Lcommon:
    idx0 = max(0, Lcommon - Lwin)
x = np.arange(Lwin)

scale = np.std(da_trim[idx0:idx0+Lwin]) / np.std(chi_trim[idx0:idx0+Lwin])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(x, da_trim[idx0:idx0+Lwin], label="Δα_n", lw=1)
ax.plot(x, chi_trim[idx0:idx0+Lwin] * scale, label="Scaled χ_n", lw=1)

ax.set_title("PG6 Fig 6 — Angle Drift and Curvature Overlay")
ax.set_xlabel("Index offset")
ax.set_ylabel("Value (scaled)")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig6_angle_vs_curvature_overlay.png"))
plt.close()

# ========================================================
# FIGURE 7
# ========================================================
print("Making Figure 7...")
angle_dev = alpha - (np.pi / 4)

fig, ax = plt.subplots(figsize=(7, 3))
max_n = min(200_000, len(angle_dev))
ax.plot(angle_dev[:max_n], lw=0.5)

ax.set_title("PG6 Fig 7 — α_n - 45° Across a Long Range")
ax.set_xlabel("n")
ax.set_ylabel("α_n - π/4")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig7_angle_deviation_from_45deg.png"))
plt.close()

# ========================================================
# FIGURE 8
# ========================================================
print("Making Figure 8...")
base_idx = 300_000
win = 2000
if base_idx + win > len(alpha):
    base_idx = max(0, len(alpha) - win)

subset_alpha = alpha[base_idx:base_idx+win]
subset_idx = np.arange(len(subset_alpha))

chi_base = base_idx
chi_sub = chi[chi_base:chi_base+len(subset_alpha)]

threshold = np.percentile(np.abs(chi), 99)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(subset_idx, subset_alpha, lw=1, label="α_n")

for i, val in enumerate(chi_sub):
    if abs(val) > threshold:
        ax.axvline(i, color="red", alpha=0.2)

ax.set_title("PG6 Fig 8 — Curvature Spikes Overlaid on Angle Kinks")
ax.set_xlabel("Index offset")
ax.set_ylabel("α_n")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PG6_Fig8_curvature_spikes_vs_angle_kinks.png"))
plt.close()

print("All PG6 figures written to:", OUT_DIR)
