import numpy as np
import math
import random
import matplotlib.pyplot as plt

try:
    import primesieve
    HAVE_PRIMESIEVE = True
except ImportError:
    HAVE_PRIMESIEVE = False
    from sympy import primerange


# ---------- Core utilities (same as before) ----------

def generate_primes(n):
    if HAVE_PRIMESIEVE:
        return list(primesieve.generate_n_primes(n))
    else:
        upper = int(n * (math.log(max(n, 2)) + math.log(math.log(max(n, 3))))) + 20
        return list(primerange(2, upper))[:n]


def gaps_from_primes(primes):
    return [primes[i+1] - primes[i] for i in range(len(primes)-1)]


def chis_from_gaps(gaps):
    g = np.array(gaps, dtype=float)
    g_n = g[:-2]
    g_np1 = g[1:-1]
    g_np2 = g[2:]
    denom = g_n + g_np1
    chi = np.zeros_like(g_n, dtype=float)
    mask = denom != 0
    chi[mask] = (g_np2[mask] - g_n[mask]) / denom[mask]
    return chi


def action_from_chis(chi):
    return float(np.sum(chi * chi))


# ---------- Experiment C: Attractor geometry ----------

def experiment_C(
    N=100000,
    out_prefix="PG3_expC"
):
    print(f"Generating primes and gaps for N={N}...")
    primes = generate_primes(N+3)
    gaps = gaps_from_primes(primes)
    gaps = gaps[:N]

    chi = chis_from_gaps(gaps)
    chi_n = chi[:-1]
    chi_np1 = chi[1:]

    # Basic stats
    abs_chi = np.abs(chi)
    max_abs_chi = float(abs_chi.max())
    mean_abs_chi = float(abs_chi.mean())

    # Radius in (chi_n, chi_{n+1}) plane
    r = np.sqrt(chi_n**2 + chi_np1**2)
    max_r = float(r.max())
    mean_r = float(r.mean())

    # Covariance and PCA
    X = np.vstack([chi_n, chi_np1])
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort eigenvalues descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    print("\nExperiment C: Attractor Geometry")
    print(f"  N              = {N}")
    print(f"  max |chi_n|    = {max_abs_chi:.4f}")
    print(f"  mean |chi_n|   = {mean_abs_chi:.4f}")
    print(f"  max radius r   = {max_r:.4f}")
    print(f"  mean radius r  = {mean_r:.4f}")
    print("\n  Covariance matrix:")
    print(cov)
    print("\n  Eigenvalues (principal variances):")
    for i, val in enumerate(eigvals):
        print(f"    λ_{i+1} = {val:.4f}")
    print("\n  Principal axes (unit eigenvectors as columns):")
    print(eigvecs)

    # ---------- Scatter plot of attractor ----------
    plt.figure(figsize=(8, 8))
    plt.scatter(chi_n, chi_np1, s=1, alpha=0.25)
    plt.title(f"Experiment C: Curvature Attractor (N={N})")
    plt.xlabel(r"$\chi_n$")
    plt.ylabel(r"$\chi_{n+1}$")
    plt.grid(True)
    plt.savefig(f"{out_prefix}_scatter.png", dpi=200)
    plt.close()

    # ---------- 2D histogram (density heatmap) ----------
    plt.figure(figsize=(8, 8))
    bins = 200
    plt.hist2d(chi_n, chi_np1, bins=bins)
    plt.title(f"Experiment C: Curvature Density (N={N})")
    plt.xlabel(r"$\chi_n$")
    plt.ylabel(r"$\chi_{n+1}$")
    plt.colorbar(label="Count")
    plt.grid(False)
    plt.savefig(f"{out_prefix}_heatmap.png", dpi=200)
    plt.close()

    print("\nSaved Experiment C plots:")
    print(f"  {out_prefix}_scatter.png")
    print(f"  {out_prefix}_heatmap.png")

    return {
        "N": N,
        "max_abs_chi": max_abs_chi,
        "mean_abs_chi": mean_abs_chi,
        "max_r": max_r,
        "mean_r": mean_r,
        "cov": cov,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }


if __name__ == "__main__":
    experiment_C(
        N=100000,
        out_prefix="PG3_expC"
    )
