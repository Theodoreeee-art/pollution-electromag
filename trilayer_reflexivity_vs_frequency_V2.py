import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constants
# -----------------------------
eps0 = 8.854187817e-12   # F/m
mu0  = 4 * np.pi * 1e-7  # H/m
c0   = 1.0 / np.sqrt(eps0 * mu0)
eps_rp = 1.4             # real part of permittivity (relative)

# -----------------------------
# Fresnel helpers
# -----------------------------
def r_coef(nL, nR):
    """Fresnel amplitude coefficient (normal incidence) from nL to nR."""
    return (nL - nR) / (nL + nR)

def R_trilayer_fresnel(sigmas, D, f_hz):
    """
    Power reflectivity R for a 3-layer absorber on PEC, equal thickness D/3, at frequency f_hz.
    Convention: losses with e^{-i ω t} -> eps = eps_rp + i*sigma/(ω*eps0).
    """
    sigma1, sigma2, sigma3 = sigmas
    d = D / 3.0
    omega = 2 * np.pi * f_hz

    # Complex permittivities
    eps1 = eps_rp + 1j * sigma1 / (omega * eps0)
    eps2 = eps_rp + 1j * sigma2 / (omega * eps0)
    eps3 = eps_rp + 1j * sigma3 / (omega * eps0)

    # Refractive indices
    n0 = 1.0
    n1, n2, n3 = np.sqrt(eps1), np.sqrt(eps2), np.sqrt(eps3)

    # Phase thicknesses
    beta1 = (2 * np.pi * f_hz / c0) * n1 * d
    beta2 = (2 * np.pi * f_hz / c0) * n2 * d
    beta3 = (2 * np.pi * f_hz / c0) * n3 * d

    # Recursive effective reflection (PEC backing -> r= -1 at layer 3-back interface)
    r34_eff = -1.0
    r23 = r_coef(n2, n3)
    r23_eff = (r23 + r34_eff * np.exp(2j * beta3)) / (1 + r23 * r34_eff * np.exp(2j * beta3))

    r12 = r_coef(n1, n2)
    r12_eff = (r12 + r23_eff * np.exp(2j * beta2)) / (1 + r12 * r23_eff * np.exp(2j * beta2))

    r01 = r_coef(n0, n1)
    r01_eff = (r01 + r12_eff * np.exp(2j * beta1)) / (1 + r01 * r12_eff * np.exp(2j * beta1))

    return np.abs(r01_eff) ** 2  # power reflectivity

# -----------------------------
# Configs with your UPDATED sigmas
# -----------------------------
configs = {
    "D = 20 cm": dict(D=0.20, sigmas=(0.03925, 0.28708, 1.99992)),
    "D = 30 cm": dict(D=0.30, sigmas=(0.04024, 0.01327, 2.18484)),
    "D = 40 cm": dict(D=0.40, sigmas=(0.01464, 0.02701, 0.07444)),
}

# -----------------------------
# Frequency grid
# -----------------------------
freqs_GHz = np.logspace(-1, 2, 800)  # 0.1 -> 100 GHz
freqs_Hz  = freqs_GHz * 1e9

# -----------------------------
# Utility: find contiguous intervals where mask is True
# -----------------------------
def true_intervals(x, mask):
    """
    Given x (monotonic) and boolean mask, return list of [x_start, x_end] intervals
    where mask is True. If mask starts/ends True, intervals are closed at the ends.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0]

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]

    intervals = []
    for s, e in zip(starts, ends):
        intervals.append([x[s], x[e]])
    return intervals

# -----------------------------
# Compute, plot, and annotate
# -----------------------------
plt.figure(figsize=(9, 6))

for label, cfg in configs.items():
    # Compute R over frequencies
    R_vals = np.array([R_trilayer_fresnel(cfg["sigmas"], cfg["D"], f) for f in freqs_Hz])
    R_dB = 10 * np.log10(np.clip(R_vals, 1e-20, 1.0))  # clamp for stability

    # Plot curve
    plt.semilogx(freqs_GHz, R_dB, lw=2, label=f"{label} — σ={cfg['sigmas']} S/m")

    # Domain of interest: R <= 0.1  <=>  R_dB <= -10
    mask_ok = R_vals <= 0.1
    intervals = true_intervals(freqs_GHz, mask_ok)

    # Print intervals to console
    if intervals:
        for (x0, x1) in intervals:
            print(f"{label}:  -10 dB band ≈ [{x0:.2f}, {x1:.2f}] GHz")
    else:
        print(f"{label}:  no -10 dB band on [0.10, 100] GHz")

    # Annotate first entry into the domain and shade valid regions
    # for j, (x0, x1) in enumerate(intervals):
    #     # Shade the region where R <= 0.1
    #     xmask = (freqs_GHz >= x0) & (freqs_GHz <= x1)
    #     plt.fill_between(freqs_GHz[xmask], R_dB[xmask], -10, alpha=0.10)

    #     # Mark the entry point of the first interval only (to avoid clutter)
    #     if j == 0:
    #         y0 = np.interp(x0, freqs_GHz, R_dB)
    #         plt.plot(x0, y0, marker="o")
    #         plt.text(x0*1.02, y0+1.0, f"entry ~ {x0:.2f} GHz", fontsize=9)

# -10 dB reference and 1 GHz marker
plt.axhline(-10, color="gray", ls="--", lw=1.2, label="-10 dB threshold")
plt.axvline(1, color="gray", ls=":", lw=1.0)
plt.text(1.03, -9.6, "1 GHz", fontsize=9, color="gray", va="bottom")

# Cosmetics
plt.title("Three-layer absorber on PEC — Reflectivity vs Frequency (Fresnel model)")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflectivity (dB)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("reflectivity_vs_freq_trilayer.png", dpi=300)
plt.show()
