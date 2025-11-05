import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 0) Physical constants
# -----------------------------
eps0 = 8.854187817e-12     # Vacuum permittivity [F/m]
mu0  = 4 * np.pi * 1e-7     # Vacuum permeability [H/m]
c0   = 1.0 / np.sqrt(eps0 * mu0)
eta0 = np.sqrt(mu0 / eps0)

# -----------------------------
# 1) Material and frequency parameters
# -----------------------------
eps_rp = 1.4                # Real part of permittivity
f = 1e9                     # Frequency = 1 GHz
omega = 2 * np.pi * f

# -----------------------------
# 2) Layer parameters (function)
# -----------------------------
def layer_params(sigma):
    """Return intrinsic impedance and wave number for a given conductivity."""
    eps_r = eps_rp - 1j * sigma / (omega * eps0)
    n = np.sqrt(eps_r)
    eta = eta0 / n
    k = omega / c0 * n
    return eta, k

# -----------------------------
# 3) Compute total reflectivity for a trilayer
# -----------------------------
def R_trilayer(sigmas, D):
    """Compute reflectivity |Γ|² for a 3-layer absorber with PEC backing."""
    sigma1, sigma2, sigma3 = sigmas
    d = D / 3.0  # equal thickness for each layer
    eta1, k1 = layer_params(sigma1)
    eta2, k2 = layer_params(sigma2)
    eta3, k3 = layer_params(sigma3)
    # Recursive impedances (PEC at back)
    Zin3 = 1j * eta3 * np.tan(k3 * d)
    Zin2 = eta2 * (Zin3 + 1j * eta2 * np.tan(k2 * d)) / (eta2 + 1j * Zin3 * np.tan(k2 * d))
    Zin1 = eta1 * (Zin2 + 1j * eta1 * np.tan(k1 * d)) / (eta1 + 1j * Zin2 * np.tan(k1 * d))
    Gamma = (Zin1 - eta0) / (Zin1 + eta0)
    return np.abs(Gamma)**2

# -----------------------------
# 4) Reference optimized values (D = 20 cm)
# -----------------------------
D = 0.20  # total thickness [m]
sigmas_ref = (0.03925, 0.28708, 1.99992)  # optimized conductivities at 1 GHz

# -----------------------------
# 5) Parametric sweep of σ₁, σ₂, σ₃
# -----------------------------
sigma_range = np.logspace(-3, 2, 600)

R_sigma1 = [R_trilayer((s, sigmas_ref[1], sigmas_ref[2]), D) for s in sigma_range]
R_sigma2 = [R_trilayer((sigmas_ref[0], s, sigmas_ref[2]), D) for s in sigma_range]
R_sigma3 = [R_trilayer((sigmas_ref[0], sigmas_ref[1], s), D) for s in sigma_range]

# -----------------------------
# 6) Plot
# -----------------------------
plt.figure(figsize=(8, 5.5))
plt.semilogx(sigma_range, 10 * np.log10(R_sigma1), label=r"$\sigma_1$ varied")
plt.semilogx(sigma_range, 10 * np.log10(R_sigma2), label=r"$\sigma_2$ varied")
plt.semilogx(sigma_range, 10 * np.log10(R_sigma3), label=r"$\sigma_3$ varied")

plt.axhline(-10, color='gray', linestyle='--', lw=1, label='-10 dB')
plt.title("Tri-layer absorber (D = 20 cm) — Reflectivity at 1 GHz vs σ₁, σ₂, σ₃")
plt.xlabel("Conductivity σ (S/m)")
plt.ylabel("Reflectivity R (dB)")
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig("reflectivity_vs_sigma_parametric.png", dpi=300)
plt.show()
