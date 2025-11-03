import numpy as np
import matplotlib.pyplot as plt

# --- Données du problème ---
f = 1e9                    # fréquence (Hz)
eps0 = 8.854e-12           # permittivité du vide (F/m)
mu0 = 4 * np.pi * 1e-7     # perméabilité du vide (H/m)
eps_r = 1.4                # permittivité relative
e = 0.15                   # épaisseur (m)
eta0 = 377                 # impédance du vide (Ω)
omega = 2 * np.pi * f

# --- Plage de conductivité (S/m) ---
sigma = np.logspace(-3, 1, 500)  # de 1e-3 à 10 S/m

# --- Calcul de la réflectivité ---
Gamma_abs = []

for s in sigma:
    eps_c = eps0 * eps_r - 1j * s / omega
    eta = np.sqrt(1j * omega * mu0 / (s + 1j * omega * eps0 * eps_r))
    gamma = np.sqrt(1j * omega * mu0 * (s + 1j * omega * eps0 * eps_r))
    Zin = eta * np.tanh(gamma * e)
    Gamma = (Zin - eta0) / (Zin + eta0)
    R_dB = 20 * np.log10(abs(Gamma))
    Gamma_abs.append(R_dB)

Gamma_abs = np.array(Gamma_abs)

# --- Tracé ---
plt.figure(figsize=(8, 5))
plt.semilogx(sigma, Gamma_abs, linewidth=2)
plt.axhline(-10, color='r', linestyle='--', label='Réflectivité = -10 dB')
plt.xlabel('Conductivité σ (S/m)')
plt.ylabel('Réflectivité (dB)')
plt.title('Réflectivité en fonction de la conductivité (f = 1 GHz, e = 15 cm)')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()
