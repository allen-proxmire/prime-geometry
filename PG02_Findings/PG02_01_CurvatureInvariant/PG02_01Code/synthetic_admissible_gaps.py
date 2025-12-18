import numpy as np
import math

def synthetic_prime_like_gaps(
    N: int,
    seed: int = 0
):
    """
    Generate N synthetic gaps that:
    - are even
    - respect mod-6 admissibility
    - have mean ~ log n
    - are not derived from primes
    """

    rng = np.random.default_rng(seed)

    gaps = np.empty(N, dtype=np.int64)

    idx = 10.0  # starting scale

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
