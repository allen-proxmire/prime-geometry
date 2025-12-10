"""
PG11_Figures.py  (FAST VERSION)

Generates all real PG11 figures using:
 - local precomputed zeta zeros (Odlyzko dataset)
 - sympy primes (no primesieve)

Figures produced:
  1. Prime vs Zero Curvature Scatter
  2. Prime Renormalized Attractor
  3. Zero Renormalized Attractor
  4. Overlaid Curvature Histograms
  5. Zero Curvature Coherence + Length Distribution
  6. Cross Spectrum of Prime vs Zero Curvature
  7. Angle Drift Comparison
  8. Curvature Potential Comparison
  9. PGEE vs ZGEE Conceptual Diagram

Runtime: seconds instead of hours.
"""

import os
import math
import numpy as np
from sympy import primerange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

OUT_DIR = "PG11_Figures"

# adjust these if needed
N_PRIMES = 120000    # number of primes to use
N_ZEROS  = 20000     # number of zeros to load from file

SMOOTH_W = 200       # smoothing window
RNG_SEED = 1234

ZERO_FILE = "zeta_zeros_first100k.txt"  # Odlyzko zero file

# ----------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------

def ensure_outdir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def moving_average(arr, W):
    if W <= 1:
        return arr.copy()
    c = np.cumsum(np.insert(arr, 0, 0.0))
    core = (c[W:] - c[:-W]) / float(W)
    pad = np.full(W-1, core[0])
    return np.concatenate([pad, core])

def run_lengths_of_sign(arr):
    s = np.sign(arr)
    out = []
    cur_sign = 0
    cur_len = 0
    for v in s:
        if v == 0:
            if cur_len > 0:
                out.append(cur_len)
            cur_sign = 0
            cur_len = 0
        else:
            if v == cur_sign:
                cur_len += 1
            else:
                if cur_len > 0:
                    out.append(cur_len)
                cur_sign = v
                cur_len = 1
    if cur_len > 0:
        out.append(cur_len)
    return np.array(out, dtype=int)

# ----------------------------------------------------------
# PRIME DATA
# ----------------------------------------------------------

def generate_primes(n_primes):
    if n_primes < 6:
        upper = 15
    else:
        n = n_primes
        upper = int(n * (math.log(n) + math.log(math.log(n))) + 10)
    primes = list(primerange(2, upper + 1))
    return primes[:n_primes]

def prime_geometry(n_primes):
    p = np.array(generate_primes(n_primes), dtype=float)
    g = np.diff(p)

    # curvature
    g0, g1, g2 = g[:-2], g[1:-1], g[2:]
    denom = g0 + g1
    denom[denom == 0] = np.nan
    chi = (g2 - g0) / denom

    # angles
    alpha = np.arctan(p[:-1] / p[1:])
    dalpha = np.diff(alpha)

    max_len = min(len(chi), len(dalpha))
    chi = chi[:max_len]
    dalpha = dalpha[:max_len]

    p_mid = p[1:1+max_len]
    g_mid = g[1:1+max_len]

    logp = np.log(p_mid)
    g_tilde = g_mid / logp
    chi_tilde = chi * logp
    dalpha_tilde = p_mid * dalpha

    return {
        "p": p_mid,
        "g": g_mid,
        "g_tilde": g_tilde,
        "chi": chi,
        "chi_tilde": chi_tilde,
        "alpha": alpha[:max_len+1],
        "dalpha": dalpha,
        "dalpha_tilde": dalpha_tilde,
    }

# ----------------------------------------------------------
# ZETA ZERO DATA (FAST)
# ----------------------------------------------------------

def zeta_zeros_fast(n_zeros, filename=ZERO_FILE):
    """
    Load the first n_zeros imaginary parts from Odlyzko file.
    Each line contains a zero height gamma_n.
    MUCH faster than mpmath.
    """
    zeros = []
    with open(filename, "r") as f:
        for line in f:
            if len(zeros) >= n_zeros:
                break
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
                zeros.append(val)
            except:
                continue
    return np.array(zeros, dtype=float)

def zero_geometry(n_zeros):
    gamma = zeta_zeros_fast(n_zeros)
    delta = np.diff(gamma)

    d0, d1, d2 = delta[:-2], delta[1:-1], delta[2:]
    denom = d0 + d1
    denom[denom == 0] = np.nan
    chi_z = (d2 - d0) / denom

    alpha_z = np.arctan(gamma[:-1] / gamma[1:])
    dalpha_z = np.diff(alpha_z)

    max_len = min(len(chi_z), len(dalpha_z))
    chi_z = chi_z[:max_len]
    dalpha_z = dalpha_z[:max_len]

    gamma_mid = gamma[1:1+max_len]
    delta_mid = delta[1:1+max_len]

    logg = np.log(gamma_mid)
    typical_gap = 2 * math.pi / logg

    delta_tilde = delta_mid / typical_gap
    chi_tilde_z = chi_z * logg
    dalpha_tilde_z = gamma_mid * dalpha_z

    return {
        "gamma": gamma_mid,
        "delta": delta_mid,
        "delta_tilde": delta_tilde,
        "chi": chi_z,
        "chi_tilde": chi_tilde_z,
        "alpha": alpha_z[:max_len+1],
        "dalpha": dalpha_z,
        "dalpha_tilde": dalpha_tilde_z,
    }

# ----------------------------------------------------------
# FIGURES
# ----------------------------------------------------------

def fig1_curvature_scatter(pr, z):
    np.random.seed(RNG_SEED)
    n = min(len(pr["chi_tilde"]), len(z["chi_tilde"]))
    idx = np.random.choice(n, size=min(5000, n), replace=False)

    x = pr["chi_tilde"][idx]
    y = z["chi_tilde"][idx]

    plt.figure(figsize=(5,4))
    plt.scatter(x, y, s=8, alpha=0.4)
    lim = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.1
    plt.plot([-lim, lim], [-lim, lim], "k--")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel(r"Renormalized prime curvature $\tilde{\chi}_n$")
    plt.ylabel(r"Renormalized zero curvature $\tilde{\chi}^{(\zeta)}_n$")
    plt.title("Prime vs Zero Renormalized Curvature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig1_curvature_scatter.png"), dpi=300)
    plt.close()


def fig2_prime_attractor(pr):
    g_t = pr["g_tilde"]
    chi_t = pr["chi_tilde"]
    n = min(len(g_t)-1, len(chi_t))

    g0 = g_t[:n]
    g1 = g_t[1:1+n]
    chi = chi_t[:n]

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(g0, g1, chi, s=2, alpha=0.4)
    ax.set_xlabel(r"$\tilde{g}_n$")
    ax.set_ylabel(r"$\tilde{g}_{n+1}$")
    ax.set_zlabel(r"$\tilde{\chi}_n$")
    ax.set_title("Renormalized Prime Attractor")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig2_prime_attractor.png"), dpi=300)
    plt.close()


def fig3_zero_attractor(z):
    d_t = z["delta_tilde"]
    chi_t = z["chi_tilde"]
    n = min(len(d_t)-1, len(chi_t))

    d0 = d_t[:n]
    d1 = d_t[1:1+n]
    chi = chi_t[:n]

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d0, d1, chi, s=2, alpha=0.4)
    ax.set_xlabel(r"$\tilde{\delta}_n$")
    ax.set_ylabel(r"$\tilde{\delta}_{n+1}$")
    ax.set_zlabel(r"$\tilde{\chi}^{(\zeta)}_n$")
    ax.set_title("Renormalized Zero Attractor")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig3_zero_attractor.png"), dpi=300)
    plt.close()


def fig4_curvature_hist(pr, z):
    x = pr["chi_tilde"]
    y = z["chi_tilde"]

    bins = np.linspace(-3, 3, 80)
    plt.figure(figsize=(5,4))
    plt.hist(x, bins=bins, density=True, alpha=0.5, label="Primes")
    plt.hist(y, bins=bins, density=True, alpha=0.5, label="Zeta zeros",
             hatch="//", edgecolor="k")
    plt.xlabel("Renormalized curvature")
    plt.ylabel("Density")
    plt.title("Curvature Distributions (Renormalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig4_curvature_hist.png"), dpi=300)
    plt.close()


def fig5_zero_coherence(z):
    chi = z["chi"]
    smooth = moving_average(chi, SMOOTH_W)
    lengths = run_lengths_of_sign(smooth)

    fig, ax = plt.subplots(2,1, figsize=(5,5), gridspec_kw={"height_ratios":[2,1]})
    x = np.arange(len(smooth))

    ax[0].plot(x, smooth, lw=0.8)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("Smoothed zero curvature")
    ax[0].set_title("Zero Coherence Phases")

    ax[1].hist(lengths, bins=30, color="gray", edgecolor="k")
    ax[1].set_xlabel("Coherence length")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig5_zero_coherence.png"), dpi=300)
    plt.close()


def fig6_cross_spectrum(pr, z):
    n = min(len(pr["chi_tilde"]), len(z["chi_tilde"]))
    xp = pr["chi_tilde"][:n] - np.mean(pr["chi_tilde"][:n])
    xz = z["chi_tilde"][:n] - np.mean(z["chi_tilde"][:n])

    Fp = np.fft.rfft(xp)
    Fz = np.fft.rfft(xz)
    cross = np.abs(Fp * np.conj(Fz))

    freqs = np.fft.rfftfreq(n, d=1.0)[1:]
    cross = cross[1:]

    plt.figure(figsize=(5,4))
    plt.loglog(freqs, cross, lw=0.8)
    plt.xlabel("Frequency")
    plt.ylabel("Cross-spectrum")
    plt.title("Cross Spectrum: Prime vs Zero Curvature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig6_cross_spectrum.png"), dpi=300)
    plt.close()


def fig7_angle_drift(pr, z):
    g = pr["g"]
    p = pr["p"]

    dg = g[1:] - g[:-1]
    da = pr["dalpha"][:len(dg)]
    lhs_p = dg
    rhs_p = 2 * p[:len(dg)] * da

    d = z["delta"]
    gamma = z["gamma"]

    dd = d[1:] - d[:-1]
    da_z = z["dalpha"][:len(dd)]
    lhs_z = dd
    rhs_z = 2 * gamma[:len(dd)] * da_z

    plt.figure(figsize=(5,4))
    plt.scatter(lhs_p, rhs_p, s=4, alpha=0.4, label="Primes")
    plt.scatter(lhs_z, rhs_z, s=4, alpha=0.4, label="Zeros")

    lim = max(np.max(np.abs(lhs_p)), np.max(np.abs(rhs_p)),
              np.max(np.abs(lhs_z)), np.max(np.abs(rhs_z))) * 1.05

    plt.plot([-lim, lim], [-lim, lim], "k--")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("Gap first difference")
    plt.ylabel(r"$2(\text{scale})\Delta\alpha$")
    plt.title("Angle Drift Relation (Primes vs Zeros)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig7_angle_drift.png"), dpi=300)
    plt.close()


def fig8_potential(pr, z):
    phi_p = np.cumsum(pr["chi"]**2)
    phi_z = np.cumsum(z["chi"]**2)

    n = min(len(phi_p), len(phi_z))
    x = np.arange(1, n+1)

    plt.figure(figsize=(5,4))
    plt.plot(x, phi_p[:n], label="Primes")
    plt.plot(x, phi_z[:n], label="Zeros")
    plt.xlabel("Index (n)")
    plt.ylabel(r"$\Phi(n)=\sum\chi_k^2$")
    plt.title("Curvature Potential Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig8_potential.png"), dpi=300)
    plt.close()


def fig9_conceptual_flow():
    plt.figure(figsize=(5,4))
    ax = plt.gca()
    ax.set_axis_off()

    ax.text(0.1, 0.8, "Prime\nGaps $g_n$", ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white"))
    ax.text(0.5, 0.8, "Prime Curvature\n$\\chi_n$", ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white"))
    ax.text(0.9, 0.8, "Angle Drift\n$\\Delta\\alpha_n$", ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white"))

    ax.annotate("", xy=(0.4,0.8), xytext=(0.2,0.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.8,0.8), xytext=(0.6,0.8), arrowprops=dict(arrowstyle="->"))

    ax.text(0.5, 0.6, "PGEE", ha="center", va="center")

    ax.text(0.1, 0.3, "Zero\nGaps $\\delta_n$", ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white"))
    ax.text(0.5, 0.3, "Zero Curvature\n$\\chi^{(\\zeta)}_n$", ha="center",
            va="center", bbox=dict(boxstyle="round", fc="white"))
    ax.text(0.9, 0.3, "Zero Angle Drift\n$\\Delta\\alpha^{(\\zeta)}_n$",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="white"))

    ax.annotate("", xy=(0.4,0.3), xytext=(0.2,0.3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.8,0.3), xytext=(0.6,0.3), arrowprops=dict(arrowstyle="->"))

    ax.text(0.5, 0.1, "ZGEE", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PG11_Fig9_conceptual_flow.png"), dpi=300)
    plt.close()

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    ensure_outdir(OUT_DIR)

    print("Computing prime geometry...")
    pr = prime_geometry(N_PRIMES)

    print("Loading zeta-zero geometry...")
    z = zero_geometry(N_ZEROS)

    print("Generating PG11 figures...")

    fig1_curvature_scatter(pr, z)
    fig2_prime_attractor(pr)
    fig3_zero_attractor(z)
    fig4_curvature_hist(pr, z)
    fig5_zero_coherence(z)
    fig6_cross_spectrum(pr, z)
    fig7_angle_drift(pr, z)
    fig8_potential(pr, z)
    fig9_conceptual_flow()

    print("Done. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
