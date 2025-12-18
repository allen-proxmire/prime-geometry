import numpy as np
import math
import sys
import matplotlib.pyplot as plt

# -----------------------------
# Segmented prime sieve (pure Python)
# -----------------------------
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

# -----------------------------
# PG2 curvature action
# -----------------------------
def curvature_action_S(gaps: np.ndarray) -> float:
    g0 = gaps[:-2].astype(np.float64)
    g1 = gaps[1:-1].astype(np.float64)
    g2 = gaps[2:].astype(np.float64)
    chi = (g2 - g0) / (g0 + g1)
    return float(np.sum(chi * chi))

# -----------------------------
# Null models
# -----------------------------
def permute_gaps(gaps, rng):
    return rng.permutation(gaps)

def block_permute_gaps(gaps, block_size, rng):
    blocks = [gaps[i:i+block_size] for i in range(0, len(gaps), block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)

# -----------------------------
# Main test
# -----------------------------
def run_test(limit, trials=100, block_size=32, seed=42):
    rng = np.random.default_rng(seed)

    print(f"\nGenerating primes ≤ {limit:,} ...")
    primes = primes_up_to_segmented(limit)
    gaps = np.diff(primes)

    S_prime = curvature_action_S(gaps)
    print(f"S_prime = {S_prime:,.4f}")

    S_perm = []
    S_block = []

    print("Running null models...")
    for _ in range(trials):
        S_perm.append(curvature_action_S(permute_gaps(gaps, rng)))
        S_block.append(curvature_action_S(block_permute_gaps(gaps, block_size, rng)))

    S_perm = np.array(S_perm)
    S_block = np.array(S_block)

    def summarize(arr):
        z = (S_prime - arr.mean()) / arr.std(ddof=1)
        pct = 100 * np.mean(arr < S_prime)
        return z, pct, arr.mean(), arr.std(ddof=1)

    z_p, pct_p, mu_p, sd_p = summarize(S_perm)
    z_b, pct_b, mu_b, sd_b = summarize(S_block)

    print("\nRESULTS")
    print("-------")
    print(f"Permutation null : z = {z_p:.3f}, percentile = {pct_p:.2f}%")
    print(f"Block null (B={block_size}) : z = {z_b:.3f}, percentile = {pct_b:.2f}%")

    return S_prime, S_perm, S_block

# -----------------------------
# Sanity check + full run
# -----------------------------
print("=== SANITY CHECK ===")
ps_test = primes_up_to_segmented(1_000_000)
if len(ps_test) != 78498:
    print("❌ Sanity check FAILED")
    print(f"Expected 78,498 primes, got {len(ps_test)}")
    sys.exit(1)

print("✅ Sanity check PASSED (78,498 primes)")

# -----------------------------
# Full 50M run
# -----------------------------
S_prime, S_perm, S_block = run_test(
    limit=50_000_000,
    trials=100,
    block_size=32,
    seed=42
)

# Optional quick plots
plt.figure()
plt.hist(S_perm, bins=30)
plt.axvline(S_prime)
plt.title("PG2 Action — Permutation Null")
plt.show()

plt.figure()
plt.hist(S_block, bins=30)
plt.axvline(S_prime)
plt.title("PG2 Action — Block Permutation Null")
plt.show()
