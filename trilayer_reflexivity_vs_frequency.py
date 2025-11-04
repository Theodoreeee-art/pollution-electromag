import numpy as np
import matplotlib.pyplot as plt


eps0 = 8.854187817e-12   # F/m
mu0  = 4 * np.pi * 1e-7  # H/m
c0   = 1.0 / np.sqrt(eps0 * mu0)
eps_rp = 1.4             


def r_coef(nL, nR):
    """Fresnel amplitude coefficient for normal incidence between media nL → nR."""
    return (nL - nR) / (nL + nR)


def R_trilayer_fresnel(sigmas, D, f):
    """
    Reflectivity R for a three-layer absorber on PEC with equal thicknesses D/3 at frequency f (Hz).
    sigmas = (σ1, σ2, σ3)
    """
    sigma1, sigma2, sigma3 = sigmas
    d = D / 3.0
    omega = 2 * np.pi * f

    # Complex permittivities (losses with e^{-i ω t})
    eps1 = eps_rp + 1j * sigma1 / (omega * eps0)
    eps2 = eps_rp + 1j * sigma2 / (omega * eps0)
    eps3 = eps_rp + 1j * sigma3 / (omega * eps0)

    n0 = 1.0
    n1, n2, n3 = np.sqrt(eps1), np.sqrt(eps2), np.sqrt(eps3)

    # Phase thicknesses
    beta1 = (2 * np.pi * f / c0) * n1 * d
    beta2 = (2 * np.pi * f / c0) * n2 * d
    beta3 = (2 * np.pi * f / c0) * n3 * d

    # Recursive effective reflection coefficients (PEC at the back)
    r34_eff = -1.0  
    r23 = r_coef(n2, n3)
    r23_eff = (r23 + r34_eff * np.exp(2j * beta3)) / (1 + r23 * r34_eff * np.exp(2j * beta3))

    r12 = r_coef(n1, n2)
    r12_eff = (r12 + r23_eff * np.exp(2j * beta2)) / (1 + r12 * r23_eff * np.exp(2j * beta2))

    r01 = r_coef(n0, n1)
    r01_eff = (r01 + r12_eff * np.exp(2j * beta1)) / (1 + r01 * r12_eff * np.exp(2j * beta1))

    return np.abs(r01_eff) ** 2  # Power reflectivity

configs = {
    "D = 20 cm": dict(D=0.20, sigmas=(0.03939, 0.29015, 1.99993)),
    "D = 30 cm": dict(D=0.30, sigmas=(0.02953, 0.02952, 10.00000)),
    "D = 40 cm": dict(D=0.40, sigmas=(0.01454, 0.02689, 0.07440)),
}


freqs_GHz = np.logspace(-1, 2, 800)   
freqs_Hz  = freqs_GHz * 1e9

plt.figure(figsize=(8, 5.5))
for label, cfg in configs.items():
    R_vals = [R_trilayer_fresnel(cfg["sigmas"], cfg["D"], f) for f in freqs_Hz]
    R_dB = 10 * np.log10(np.clip(R_vals, 1e-16, 1))  # clamp for stability
    plt.semilogx(freqs_GHz, R_dB, lw=2, label=f"{label} — σ={cfg['sigmas']} S/m")


plt.axhline(-10, color="gray", ls="--", lw=1.2, label="-10 dB threshold")
plt.axvline(1, color="gray", ls=":", lw=0.9)
plt.text(1.03, -9.6, "1 GHz", fontsize=9, color="gray", va="bottom")

plt.title("Three-layer absorber on PEC — Reflectivity vs Frequency (Fresnel model)", fontsize=12)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflectivity (dB)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("reflectivity_vs_freq_trilayer.png", dpi=300)
plt.show()
