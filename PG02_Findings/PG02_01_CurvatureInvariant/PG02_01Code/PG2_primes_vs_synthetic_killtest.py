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
# Synthetic prime-like gaps (from your sanity-checked generator)
# ============================================================
def synthetic_prime_like_gaps(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaps = np.empty(N, dtype=np.int64)
    idx = 10.0
    for i in range(N):
        lam = math.log(idx)
        g = int(2 * round(rng.exponential(lam) / 2))
        if g < 2:
            g = 2
        if g % 6 == 0:
            g += 2
        gaps[i] = g
        idx += g
    return gaps

# ============================================================
# True PG2 action
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
# Run one sweep (for any gap sequence)
# ============================================================
def run_sweep(gaps: np.ndarray, label: str, trials: int, block_sizes, seed: int):
    rng = np.random.default_rng(seed)
    S_prime = action_pg2(gaps)

    print(f"\n=== {label} ===")
    print(f"Length gaps = {len(gaps):,}")
    print(f"S_prime = {S_prime:,.4f}")
    print("-------------------------------------------------------------")
    print("BlockSize | Perm_z   | Perm_pct | Block_z  | Block_pct")
    print("-------------------------------------------------------------")

    out = []
    for B in block_sizes:
        S_perm = np.empty(trials)
        S_block = np.empty(trials)

        for i in range(trials):
            gp = permute_gaps(gaps, rng)
            gb = block_permute_gaps(gaps, B, rng)
            S_perm[i] = action_pg2(gp)
            S_block[i] = action_pg2(gb)

        z_p = (S_prime - S_perm.mean()) / S_perm.std(ddof=1)
        pct_p = 100 * np.mean(S_perm < S_prime)

        z_b = (S_prime - S_block.mean()) / S_block.std(ddof=1)
        pct_b = 100 * np.mean(S_block < S_prime)

        print(f"{B:9d} | {z_p:7.2f} | {pct_p:8.2f}% | {z_b:7.2f} | {pct_b:8.2f}%")
        out.append((B, z_p, pct_p, z_b, pct_b))

    print("-------------------------------------------------------------")
    return S_prime, out

# ============================================================
# MAIN
# ============================================================
print("=== SANITY CHECK (primes ≤ 1,000,000) ===")
ps_test = primes_up_to_segmented(1_000_000)
if len(ps_test) != 78498:
    print("❌ Prime sanity check FAILED")
    sys.exit(1)
print("✅ Prime sanity check PASSED")

LIMIT = 50_000_000
TRIALS = 100
BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 42

print(f"\nGenerating primes ≤ {LIMIT:,} ...")
primes = primes_up_to_segmented(LIMIT)
gaps_primes = np.diff(primes)

print("\nGenerating synthetic admissible gaps (same length) ...")
gaps_syn = synthetic_prime_like_gaps(len(gaps_primes), seed=SEED)

# Run sweeps
S_p, table_p = run_sweep(gaps_primes, "PRIMES (≤ 50,000,000)", TRIALS, BLOCK_SIZES, seed=SEED)
S_s, table_s = run_sweep(gaps_syn, "SYNTHETIC (admissible, prime-like)", TRIALS, BLOCK_SIZES, seed=SEED+1)

print("\n=== SUMMARY ===")
print(f"Primes S_prime    : {S_p:,.4f}")
print(f"Synthetic S_prime : {S_s:,.4f}")
print("\nInterpretation:")
print("- If synthetic shows similar Perm_z (~ -12) and crossover (~32), PG is not prime-specific.")
print("- If synthetic is near-null or behaves differently, PG is detecting prime-specific structure.")
