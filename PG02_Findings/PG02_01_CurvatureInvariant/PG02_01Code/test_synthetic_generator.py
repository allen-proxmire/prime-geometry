import numpy as np
from synthetic_admissible_gaps import synthetic_prime_like_gaps

gaps = synthetic_prime_like_gaps(100_000, seed=42)

print("Mean gap:", gaps.mean())
print("Min gap:", gaps.min())
print("All even:", np.all(gaps % 2 == 0))

mods, counts = np.unique(gaps % 6, return_counts=True)
print("mod 6 distribution:")
for m, c in zip(mods, counts):
    print(f"  {m}: {c}")
