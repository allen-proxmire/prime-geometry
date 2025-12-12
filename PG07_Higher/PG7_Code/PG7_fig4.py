def smooth(x, W):
    return np.convolve(x, np.ones(W)/W, mode='valid')

plt.figure(figsize=(10,6))

for W in [100, 250, 500]:
    plt.plot(smooth(Theta, W), label=f"W={W}")

plt.title("Smoothed Third-Order Functional Θₙ")
plt.legend()
plt.savefig("PG7_Fig4_theta_smoothed.png", dpi=300)
