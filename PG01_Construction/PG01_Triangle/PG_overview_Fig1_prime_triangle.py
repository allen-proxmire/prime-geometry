import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figures_overview", exist_ok=True)

# Sample primes for visualization
pn = 101
pn1 = 103

# Coordinates
A = (0,0)          # Right angle
B = (pn,0)         # Horizontal leg length = p_n
C = (0,pn1)        # Vertical leg length = p_{n+1}

plt.figure(figsize=(6,6))

# Draw triangle edges
plt.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2)  # bottom side (p_n)
plt.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2)  # left side (p_{n+1})
plt.plot([B[0], C[0]], [B[1], C[1]], 'k-', linewidth=2)  # hypotenuse

# Label sides
plt.text(pn/2, -5, r"$p_n$", fontsize=16, ha='center')
plt.text(-10, pn1/2, r"$p_{n+1}$", fontsize=16, va='center', rotation=90)

# Draw angle arc for alpha_n at vertex C
r = 12
theta = np.linspace(np.pi/2, np.pi/2 + np.arctan(pn / pn1), 100)
x_arc = C[0] + r*np.cos(theta)
y_arc = C[1] + r*np.sin(theta)
plt.plot(x_arc, y_arc, 'r-', linewidth=2)

# Label alpha_n
plt.text(C[0] + r*1.2, C[1] - r*0.1, r"$\alpha_n$", fontsize=20, color='red')

plt.title("Prime Triangle for $(p_n,\,p_{n+1})$", fontsize=16)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.savefig("figures_overview/PG_overview_Fig1_prime_triangle.png", dpi=300)
plt.close()
