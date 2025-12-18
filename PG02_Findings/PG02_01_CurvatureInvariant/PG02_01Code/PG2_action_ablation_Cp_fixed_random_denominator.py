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
# PG2 numerator + FIXED randomized denominator
# χ_n = (g_{n+2} - g_n) / permuted(g_n + g_{n+1})
# ============================================================
def action_fixed_random_denominator(gaps: np.ndarray, denom_perm: np.ndarray) -> float:
    g0 = gaps[:-2].astype(np.float64)
    g2 = gaps[2:].astype(np.float64)
    chi = (g2 - g0) / denom_perm
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
# Run one block-size experiment
# ============================================================
def run_block_test(gaps, denom_perm, block_size, trials, rng):
    S_prime = action_fixed_random_denominator(gaps, denom_perm)

    S_perm = np.empty(trials)
    S_block = np.empty(trials)

    for i in range(trials):
        gp = permute_gaps(gaps, rng)
        gb = block_permute_gaps(gaps, block_size, rng)

        S_perm[i]  = action_fixed_random_denominator(gp, denom_perm)
        S_block[i] = action_fixed_random_denominator(gb, denom_perm)

    def summarize(arr):
        z = (S_prime - arr.mean()) / arr.std(ddof=1)
        pct = 100 * np.mean(arr < S_prime)
        return z, pct

    z_p, pct_p = summarize(S_perm)
    z_b, pct_b = summarize(S_block)

    return S_prime, z_p, pct_p, z_b, pct_b

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
gaps = np.diff(primes)

# Build and freeze ONE randomized denominator from primes
rng = np.random.default_rng(SEED)
denom_true = (gaps[:-2] + gaps[1:-1]).astype(np.float64)
denom_perm = rng.permutation(denom_true)

print("\nABLATION C′ — FIXED RANDOM DENOMINATOR (PAIRING DESTROYED)")
print("-------------------------------------------------------------")
print("BlockSize | S_prime      | Perm_z   | Perm_pct | Block_z  | Block_pct")
print("-------------------------------------------------------------")

for B in BLOCK_SIZES:
    S_prime, z_p, pct_p, z_b, pct_b = run_block_test(
        gaps, denom_perm, B, TRIALS, rng
    )

    print(f"{B:9d} | {S_prime:11.1f} | {z_p:7.2f} | {pct_p:8.2f}% | {z_b:7.2f} | {pct_b:8.2f}%")

print("-------------------------------------------------------------")
print("DONE.")
