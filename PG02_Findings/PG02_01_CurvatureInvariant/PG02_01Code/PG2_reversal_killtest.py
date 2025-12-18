import numpy as np
import math
import sys

# ============================================================
# Segmented prime sieve (pure Python)
# ============================================================
def primes_up_to_segmented(n: int, segment_size: int = 1_000_000) -> np.ndarray:
    limit = int(math.isqrt(n)) + 1

    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = False
    small_primes = np.nonzero(sieve)[0]

    primes = []
    for low in range(2, n + 1, segment_size):
        high = min(low + segment_size - 1, n)
        segment = np.ones(high - low + 1, dtype=bool)

        for p in small_primes:
            start = max(p*p, ((low + p - 1) // p) * p)
            if start > high:
                continue
            segment[start - low :: p] = False

        if low == 2:
            segment[0] = True

        primes.extend((low + np.nonzero(segment)[0]).tolist())

    return np.array(primes, dtype=np.int64)

# ============================================================
# True PG2 curvature action
# chi_n = (g_{n+2} - g_n) / (g_n + g_{n+1})
# ============================================================
def action_pg2(gaps: np.ndarray) -> float:
    g0 = gaps[:-2].astype(np.float64)
    g1 = gaps[1:-1].astype(np.float64)
    g2 = gaps[2:].astype(np.float64)
    chi = (g2 - g0) / (g0 + g1)
    return float(np.sum(chi * chi))

# ============================================================
# Null models
# ============================================================
def permute_gaps(gaps, rng):
    return rng.permutation(gaps)

def block_permute_gaps(gaps, block_size, rng):
    blocks = [gaps[i:i+block_size] for i in range(0, len(gaps), block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)

# ============================================================
# One sweep (forward or reversed)
# ============================================================
def run_sweep(gaps: np.ndarray, label: str, trials: int, block_sizes, rng):
    S_prime = action_pg2(gaps)

    print(f"\n=== {label} ===")
    print(f"S_prime = {S_prime:,.4f}")
    print("-------------------------------------------------------------")
    print("BlockSize | Perm_z   | Perm_pct | Block_z  | Block_pct")
    print("-------------------------------------------------------------")

    results = []

    for B in block_sizes:
        S_perm = np.empty(trials)
        S_block = np.empty(trials)

        for i in range(trials):
            gp = permute_gaps(gaps, rng)
            gb = block_permute_gaps(gaps, B, rng)
            S_perm[i] = action_pg2(gp)
            S_block[i] = action_pg2(gb)

        def summarize(arr):
            z = (S_prime - arr.mean()) / arr.std(ddof=1)
            pct = 100 * np.mean(arr < S_prime)
            return z, pct

        z_p, pct_p = summarize(S_perm)
        z_b, pct_b = summarize(S_block)

        print(f"{B:9d} | {z_p:7.2f} | {pct_p:8.2f}% | {z_b:7.2f} | {pct_b:8.2f}%")
        results.append((B, z_p, pct_p, z_b, pct_b))

    print("-------------------------------------------------------------")
    return S_prime, results

# ============================================================
# MAIN
# ============================================================
print("=== SANITY CHECK ===")
ps_test = primes_up_to_segmented(1_000_000)
if len(ps_test) != 78498:
    print("❌ Sanity check FAILED")
    sys.exit(1)
print("✅ Sanity check PASSED")

LIMIT = 50_000_000
TRIALS = 100
BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 42

print(f"\nGenerating primes ≤ {LIMIT:,} ...")
primes = primes_up_to_segmented(LIMIT)
gaps_fwd = np.diff(primes)
gaps_rev = gaps_fwd[::-1].copy()

rng = np.random.default_rng(SEED)

# Forward
S_fwd, res_fwd = run_sweep(gaps_fwd, "FORWARD GAPS", TRIALS, BLOCK_SIZES, rng)

# Reversed
# Use a different RNG stream so null draws aren't identical by accident
rng = np.random.default_rng(SEED + 1)
S_rev, res_rev = run_sweep(gaps_rev, "REVERSED GAPS", TRIALS, BLOCK_SIZES, rng)

print("\n=== SUMMARY ===")
print(f"Forward S_prime : {S_fwd:,.4f}")
print(f"Reverse  S_prime : {S_rev:,.4f}")
print("\nIf Forward and Reverse tables match closely, PG2 is not directional.")
print("If they differ materially, that supports an arrow-of-index / evolution interpretation.")
