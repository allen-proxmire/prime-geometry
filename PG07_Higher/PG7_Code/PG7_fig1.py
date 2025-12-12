import numpy as np
import matplotlib.pyplot as plt

# assume you already have p, g, chi, alpha arrays

N = len(alpha) - 1

# actual deviation
adev = np.abs(alpha - np.pi/4)

# stability bound partial sums
bound = np.cumsum(np.abs(chi[:-1]) / p[:-2]) * 0.5

plt.figure(figsize=(10,5))
plt.plot(adev[1:], label="Actual |α_n − π/4|")
plt.plot(bound, label="Stability Bound", alpha=0.7)
plt.yscale("log")
plt.legend()
plt.title("Stability Inequality for Prime Triangle Angle")
plt.savefig("PG7_Fig1_stability_bound.png", dpi=300)
