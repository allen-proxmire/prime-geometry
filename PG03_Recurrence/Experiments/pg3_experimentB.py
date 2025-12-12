import numpy as np
import random
import math
import matplotlib.pyplot as plt

try:
    import primesieve
    HAVE_PRIMESIEVE = True
except ImportError:
    HAVE_PRIMESIEVE = False
    from sympy import primerange


# ---------- Core utilities (same as Experiment A) ----------

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


def perturb_adjacent_swap(gaps, idx):
    """Swap gaps[idx] and gaps[idx+1]."""
    gaps2 = gaps.copy()
    gaps2[idx], gaps2[idx+1] = gaps2[idx+1], gaps2[idx]
    return gaps2


def perturb_local_adjust(gaps, idx, epsilon):
    """g_i -> g_i + epsilon (ensuring >0)."""
    gaps2 = gaps.copy()
    gaps2[idx] = max(1, gaps2[idx] + epsilon)
    return gaps2


def perturb_window_randomize(gaps, start, length):
    """Randomly permute a window of gaps."""
    gaps2 = gaps.copy()
    block = gaps2[start:start+length]
    random.shuffle(block)
    gaps2[start:start+length] = block
    return gaps2


# ---------- Experiment B core ----------

def experiment_B(N=50000, swap_idx=2000, adjust_idx=2000,
                 epsilon=5, window_start=2000, window_size=50,
                 out_prefix="PG3_expB"):

    print(f"\nGenerating primes and gaps for N={N}...")
    primes = generate_primes(N+3)
    gaps = gaps_from_primes(primes)
    gaps = gaps[:N]

    # true curvature
    chi_true = chis_from_gaps(gaps)
    S_true = action_from_chis(chi_true)

    print(f"\nTrue action S_true = {S_true:.4e}")

    # 1. Adjacent swap perturbation
    print("\nPerforming adjacent swap perturbation...")
    gaps_swap = perturb_adjacent_swap(gaps, swap_idx)
    chi_swap = chis_from_gaps(gaps_swap)
    S_swap = action_from_chis(chi_swap)
    print(f"  S_swap = {S_swap:.4e}, ΔS = {S_swap - S_true:.4e}")

    # 2. Local adjustment
    print("\nPerforming local adjustment perturbation...")
    gaps_adjust = perturb_local_adjust(gaps, adjust_idx, epsilon)
    chi_adjust = chis_from_gaps(gaps_adjust)
    S_adjust = action_from_chis(chi_adjust)
    print(f"  S_adjust = {S_adjust:.4e}, ΔS = {S_adjust - S_true:.4e}")

    # 3. Window randomization
    print("\nPerforming window randomization perturbation...")
    gaps_window = perturb_window_randomize(gaps, window_start, window_size)
    chi_window = chis_from_gaps(gaps_window)
    S_window = action_from_chis(chi_window)
    print(f"  S_window = {S_window:.4e}, ΔS = {S_window - S_true:.4e}")

    # ---------- Plot curvature return maps ----------

    plt.figure(figsize=(8,8))
    plt.scatter(chi_true[:-1], chi_true[1:], s=1, alpha=0.25, label="True primes", c="blue")
    plt.scatter(chi_swap[:-1], chi_swap[1:], s=4, alpha=0.8, label="Adjacent swap", c="red")
    plt.title("Experiment B: Curvature Map — Adjacent Swap")
    plt.xlabel(r"$\chi_n$")
    plt.ylabel(r"$\chi_{n+1}$")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_swap_returnmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8,8))
    plt.scatter(chi_true[:-1], chi_true[1:], s=1, alpha=0.25, label="True primes", c="blue")
    plt.scatter(chi_adjust[:-1], chi_adjust[1:], s=4, alpha=0.8, label="Local Adjust", c="orange")
    plt.title("Experiment B: Curvature Map — Local Adjustment")
    plt.xlabel(r"$\chi_n$")
    plt.ylabel(r"$\chi_{n+1}$")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_adjust_returnmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8,8))
    plt.scatter(chi_true[:-1], chi_true[1:], s=1, alpha=0.25, label="True primes", c="blue")
    plt.scatter(chi_window[:-1], chi_window[1:], s=4, alpha=0.8, label="Window Randomization", c="green")
    plt.title("Experiment B: Curvature Map — Window Randomization")
    plt.xlabel(r"$\chi_n$")
    plt.ylabel(r"$\chi_{n+1}$")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_window_returnmap.png", dpi=200)
    plt.close()

    print("\nSaved Experiment B plots.")
    print(f"{out_prefix}_swap_returnmap.png")
    print(f"{out_prefix}_adjust_returnmap.png")
    print(f"{out_prefix}_window_returnmap.png")

    return {
        "S_true": S_true,
        "S_swap": S_swap,
        "S_adjust": S_adjust,
        "S_window": S_window,
        "chi_true": chi_true,
        "chi_swap": chi_swap,
        "chi_adjust": chi_adjust,
        "chi_window": chi_window
    }


# CLI
if __name__ == "__main__":
    experiment_B(
        N=50000,
        swap_idx=2000,
        adjust_idx=2000,
        epsilon=5,
        window_start=2000,
        window_size=50,
        out_prefix="PG3_expB"
    )
