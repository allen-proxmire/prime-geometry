from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(g[:-1], g[1:], chi[:-1],
           s=2, alpha=0.3)

ax.set_xlabel("g_n")
ax.set_ylabel("g_{n+1}")
ax.set_zlabel("chi_n")
ax.set_title("Prime Geometry Attractor in (g_n, g_{n+1}, χ_n) Space")

plt.savefig("PG7_Fig5_attractor_3d.png", dpi=300)
