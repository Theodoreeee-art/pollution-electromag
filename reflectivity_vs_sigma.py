import numpy as np
import matplotlib.pyplot as plt


# Constantes physiques
eps0 = 8.854187817e-12  # F/m
mu0 = 4 * np.pi * 1e-7   # H/m
c0 = 1.0 / np.sqrt(eps0 * mu0)
eta0 = np.sqrt(mu0 / eps0)  

# Paramètres du problème
f = 1e9                  # Fréquence : 1 GHz
omega = 2 * np.pi * f
d = 0.15                 # Épaisseur de la couche : 15 cm
eps_r_prime = 1.4        
sigma = np.logspace(-4, 1.5, 1200)  

# Modèle Fresnel (incidence normale)

eps_r_complex = eps_r_prime + 1j * sigma / (omega * eps0)
n_tilde = np.sqrt(eps_r_complex)            # Indice de réfraction complexe
beta = (2 * np.pi * f / c0) * n_tilde * d   # Phase de propagation

r01 = (1 - n_tilde) / (1 + n_tilde)         # Réflexion air/couche
r12 = -1                                    # Réflexion couche/PEC
r_tot = (r01 + r12 * np.exp(2j * beta)) / (1 + r01 * r12 * np.exp(2j * beta))

R = np.abs(r_tot) ** 2                      # Réflectivité en puissance
R_dB = 10 * np.log10(R)

# Post-processing: minima and -10 dB range
min_idx = np.argmin(R)
sigma_min = sigma[min_idx]
Rmin = R[min_idx]
Rmin_dB = R_dB[min_idx]

mask_10dB = R <= 0.1  # Seuil -10 dB
intervals = []
if np.any(mask_10dB):
    idx = np.where(mask_10dB)[0]
    start = idx[0]
    prev = idx[0]
    for idc in idx[1:]:
        if idc != prev + 1:
            intervals.append((start, prev))
            start = idc
        prev = idc
    intervals.append((start, prev))

# Plot
plt.figure(figsize=(8, 5.2))
plt.semilogx(sigma, R_dB, linewidth=2)
plt.axhline(-10, linestyle='--', color='gray', label='-10 dB')
plt.scatter([sigma_min], [Rmin_dB], color='red', zorder=3, label='Minimum')
plt.title("Monocouche (d = 15 cm) — Réflectivité à 1 GHz selon la conductivité σ")
plt.xlabel("σ (S/m)")
plt.ylabel("R (dB)")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("reflectivity_vs_sigma.png", dpi=300)
plt.show()

# Post-processing: minima and -10 dB range
print(f"Minimum: R_min = {Rmin:.4f} ({Rmin_dB:.2f} dB) at σ ≈ {sigma_min:.5g} S/m")
if intervals:
    for a, b in intervals:
        print(f"Sigma range for R ≤ -10 dB: [{sigma[a]:.5g}, {sigma[b]:.5g}] S/m")
else:
    print("No sigma interval reaches R ≤ -10 dB on this sweep.")
