import numpy as np
import matplotlib.pyplot as plt


eps0 = 8.854187817e-12   # F/m
mu0  = 4 * np.pi * 1e-7  # H/m
c0   = 1.0 / np.sqrt(eps0 * mu0)


f = 1e9                  # 1 GHz
omega = 2 * np.pi * f
eps_r_prime = 1.4

# Thickness and conductivity grids
d_cm = np.linspace(5, 40, 360)          
d_m  = d_cm / 100.0
sigma = np.logspace(-3, 1, 400)         


# Complex index with losses (e^{-i ω t})
eps_r = eps_r_prime + 1j * (sigma / (omega * eps0))   # shape: (Ns,)
n = np.sqrt(eps_r)                                    # (Ns,)
beta = (2 * np.pi * f / c0) * n[:, None] * d_m[None, :]  # (Ns, Nd)

# Fresnel coefficients
r01 = (1.0 - n) / (1.0 + n)        
r12 = -1.0                         
e2jbeta = np.exp(2j * beta)

# Total reflection amplitude (broadcast to 2D)
r_tot = (r01[:, None] + r12 * e2jbeta) / (1.0 + r01[:, None] * r12 * e2jbeta)
R = np.abs(r_tot) ** 2                               # (Ns, Nd)
R_dB = 10.0 * np.log10(np.clip(R, 1e-16, 1.0))       # avoid -inf


imin = np.unravel_index(np.argmin(R, axis=None), R.shape)
sigma_min = sigma[imin[0]]
dmin_cm   = d_cm[imin[1]]
Rmin_dB   = 10*np.log10(R[imin])


plt.figure(figsize=(9, 5.8))

# Clip only for display so color scale is readable
R_dB_disp = np.clip(R_dB, -60, 0)
# pcolormesh expects axis arrays for grid corners; use shading="auto"
pc = plt.pcolormesh(d_cm, sigma, R_dB_disp, shading="auto", cmap="viridis", vmin=-60, vmax=0)

# Contours at key thresholds
levels = [-40, -30, -20, -10]
CS = plt.contour(d_cm, sigma, R_dB, levels=levels, colors="white", linestyles=["--","--","--","-"], linewidths=1.3)
plt.clabel(CS, fmt=lambda v: f"{int(v)} dB", inline=True, fontsize=9, colors="white")

# Mark global minimum
plt.scatter([dmin_cm], [sigma_min], s=35, c="red", edgecolors="black", zorder=5, label=f"Min: {Rmin_dB:.1f} dB\nσ≈{sigma_min:.3g} S/m, d≈{dmin_cm:.1f} cm")

# Axes, scales, cosmetics
plt.yscale("log")
plt.xlabel("Thickness d (cm)")
plt.ylabel("Conductivity σ (S/m)")
plt.title("Monolayer on PEC — Reflectivity at 1 GHz vs d and σ (Fresnel)")
cbar = plt.colorbar(pc, label="Reflectivity R (dB)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("reflectivity_map_mono_v2.png", dpi=300)
plt.show()

print(f"Global minimum at 1 GHz: R = {10**(Rmin_dB/10):.3e} ({Rmin_dB:.2f} dB), σ ≈ {sigma_min:.5g} S/m, d ≈ {dmin_cm:.3g} cm")
