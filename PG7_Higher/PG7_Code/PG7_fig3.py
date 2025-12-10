Theta = g[3:] - 3*g[2:-1] + 3*g[1:-2] - g[:-3]

plt.figure(figsize=(7,5))
plt.hist(Theta, bins=200, alpha=0.7, label="True primes", density=True)

# permutation baseline
perm = np.random.permutation(g[:-3])
Theta_perm = perm[3:] - 3*perm[2:-1] + 3*perm[1:-2] - perm[:-3]
plt.hist(Theta_perm, bins=200, alpha=0.5, label="Permutation", density=True)

plt.title("Distribution of Θₙ (Third-Order Functional)")
plt.legend()
plt.savefig("PG7_Fig3_theta_distribution.png", dpi=300)
