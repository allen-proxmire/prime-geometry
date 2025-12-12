import numpy as np
import math
import matplotlib.pyplot as plt

try:
    import primesieve
    HAVE_PRIMESIEVE = True
except ImportError:
    HAVE_PRIMESIEVE = False
    from sympy import primerange


# ---------- Core utilities (as before) ----------

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


def sliding_mean(x, W):
    """
    Simple sliding mean with window W.
    Returns array of length len(x) - W + 1.
    """
    x = np.asarray(x, dtype=float)
    if W > len(x):
        raise ValueError("Window W larger than data length")
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[W:] - cumsum[:-W]) / W


def sign_runs(x, tol=0.0):
    """
    Given a 1D array x, compute lengths of contiguous runs
    where x > tol (positive) or x < -tol (negative).
    Returns dict with 'pos' and 'neg' lists of run lengths.
    """
    x = np.asarray(x, dtype=float)
    runs_pos = []
    runs_neg = []
    current_sign = 0
    current_len = 0

    for val in x:
        if val > tol:
            s = 1
        elif val < -tol:
            s = -1
        else:
            s = 0

        if s == current_sign and s != 0:
            current_len += 1
        else:
            # close previous run
            if current_sign == 1 and current_len > 0:
                runs_pos.append(current_len)
            elif current_sign == -1 and current_len > 0:
                runs_neg.append(current_len)
            # start new
            if s != 0:
                current_sign = s
                current_len = 1
            else:
                current_sign = 0
                current_len = 0

    # close final run
    if current_sign == 1 and current_len > 0:
        runs_pos.append(current_len)
    elif current_sign == -1 and current_len > 0:
        runs_neg.append(current_len)

    return {"pos": runs_pos, "neg": runs_neg}


# ---------- Experiment D ----------

def experiment_D(
    N=200000,
    windows=(500, 2000, 5000, 10000),
    tol=0.0,
    out_prefix="PG3_expD"
):
    print(f"Generating primes and gaps for N={N}...")
    primes = generate_primes(N+3)
    gaps = gaps_from_primes(primes)
    gaps = gaps[:N]

    chi = chis_from_gaps(gaps)
    L = chi * chi

    results = {}

    for W in windows:
        print(f"\n--- Window W = {W} ---")
        chi_sm = sliding_mean(chi, W)
        L_sm = sliding_mean(L, W)

        runs = sign_runs(chi_sm, tol=tol)
        pos_runs = np.array(runs["pos"], dtype=int)
        neg_runs = np.array(runs["neg"], dtype=int)

        pos_mean = float(pos_runs.mean()) if len(pos_runs) > 0 else 0.0
        neg_mean = float(neg_runs.mean()) if len(neg_runs) > 0 else 0.0
        pos_max = int(pos_runs.max()) if len(pos_runs) > 0 else 0
        neg_max = int(neg_runs.max()) if len(neg_runs) > 0 else 0

        print(f"  num positive runs = {len(pos_runs)}")
        print(f"  num negative runs = {len(neg_runs)}")
        print(f"  mean pos run len  = {pos_mean:.2f}")
        print(f"  mean neg run len  = {neg_mean:.2f}")
        print(f"  max pos run len   = {pos_max}")
        print(f"  max neg run len   = {neg_max}")

        # Save plots: time series of smoothed chi, and histogram of run lengths
        x_axis = np.arange(len(chi_sm))

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, chi_sm, linewidth=0.7)
        plt.axhline(0.0, linewidth=0.7)
        plt.title(f"Experiment D: Smoothed curvature, W={W}")
        plt.xlabel("index k")
        plt.ylabel(r"$\overline{\chi}^{(W)}_k$")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_chi_sm_W{W}.png", dpi=200)
        plt.close()

        # histogram of run lengths
        plt.figure(figsize=(8, 4))
        bins = min(50, max(pos_max, neg_max, 10))
        if len(pos_runs) > 0:
            plt.hist(pos_runs, bins=bins, alpha=0.6, label="Positive runs")
        if len(neg_runs) > 0:
            plt.hist(neg_runs, bins=bins, alpha=0.6, label="Negative runs")
        plt.title(f"Experiment D: Run length distribution, W={W}")
        plt.xlabel("run length")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_runs_W{W}.png", dpi=200)
        plt.close()

        results[W] = {
            "chi_sm": chi_sm,
            "L_sm": L_sm,
            "runs_pos": pos_runs,
            "runs_neg": neg_runs,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "pos_max": pos_max,
            "neg_max": neg_max,
        }

    print("\nSaved Experiment D plots for all windows.")
    return results


if __name__ == "__main__":
    experiment_D(
        N=200000,
        windows=(500, 2000, 5000, 10000),
        tol=0.0,
        out_prefix="PG3_expD"
    )
