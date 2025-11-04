import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


eps0 = 8.854187817e-12
mu0  = 4 * np.pi * 1e-7
c0   = 1.0 / np.sqrt(eps0 * mu0)
eps_rp = 1.4  # Re(eps_r)


def R_monolayer_fresnel(f_GHz, d_m, sigma):
    f = np.asarray(f_GHz) * 1e9
    w = 2*np.pi*f
    eps_r = eps_rp + 1j * sigma/(w*eps0)
    n = np.sqrt(eps_r)
    beta = (2*np.pi*f/c0)*n*d_m
    r01 = (1 - n) / (1 + n)
    r12 = -1.0  # layer -> PEC
    rtot = (r01 + r12*np.exp(2j*beta)) / (1 + r01*r12*np.exp(2j*beta))
    return np.abs(rtot)**2

def R_trilayer_fresnel(f_GHz, D_m, sigmas):
    f = np.asarray(f_GHz) * 1e9
    w = 2*np.pi*f
    d = D_m/3.0
    s1, s2, s3 = sigmas
    n0 = 1.0
    eps1 = eps_rp + 1j*s1/(w*eps0)
    eps2 = eps_rp + 1j*s2/(w*eps0)
    eps3 = eps_rp + 1j*s3/(w*eps0)
    n1, n2, n3 = np.sqrt(eps1), np.sqrt(eps2), np.sqrt(eps3)
    b1 = (2*np.pi*f/c0)*n1*d
    b2 = (2*np.pi*f/c0)*n2*d
    b3 = (2*np.pi*f/c0)*n3*d
    r = lambda nl, nr: (nl - nr)/(nl + nr)
    r34_eff = -1.0
    r23 = r(n2, n3)
    r23_eff = (r23 + r34_eff*np.exp(2j*b3)) / (1 + r23*r34_eff*np.exp(2j*b3))
    r12 = r(n1, n2)
    r12_eff = (r12 + r23_eff*np.exp(2j*b2)) / (1 + r12*r23_eff*np.exp(2j*b2))
    r01 = r(n0, n1)
    r01_eff = (r01 + r12_eff*np.exp(2j*b1)) / (1 + r01*r12_eff*np.exp(2j*b1))
    return np.abs(r01_eff)**2


def sigma_opt_for_d(d_m):
    f1 = 1.0  # GHz
    obj = lambda s: R_monolayer_fresnel(f1, d_m, s)
    res = minimize_scalar(obj, bounds=(1e-4, 10.0), method="bounded")
    return float(res.x)


tri_cases = {
    "D=20 cm": dict(D=0.20, sigmas=(0.03939, 0.29015, 1.99993)),
    "D=30 cm": dict(D=0.30, sigmas=(0.02953, 0.02952, 10.00000)),
    "D=40 cm": dict(D=0.40, sigmas=(0.01454, 0.02689, 0.07440)),
}

f_GHz = np.logspace(-1, 2, 1500)  # 0.1–100 GHz

for label, cfg in tri_cases.items():
    D = cfg["D"]
    sigmas = cfg["sigmas"]
    # Optimize monolayer sigma for this thickness at 1 GHz
    sigma_mono = sigma_opt_for_d(D)

    R_mono = R_monolayer_fresnel(f_GHz, D, sigma_mono)
    R_tri  = R_trilayer_fresnel(f_GHz, D, sigmas)

    plt.figure(figsize=(8,5.2))
    plt.semilogx(f_GHz, 10*np.log10(np.clip(R_mono, 1e-16, 1)), lw=2,
                 label=f"Monolayer (d={D*100:.0f} cm, σ*={sigma_mono:.4g} S/m)")
    plt.semilogx(f_GHz, 10*np.log10(np.clip(R_tri,  1e-16, 1)), lw=2,
                 label=f"Tri-layer ({label}, σ={sigmas})")
    plt.axhline(-10, ls="--", color="gray", label="-10 dB")
    plt.axvline(1.0, ls=":",  color="gray", lw=0.8)
    plt.text(1.03, -9.3, "1 GHz", fontsize=9, color="gray", va="bottom")
    plt.title(f"Reflectivity vs Frequency — Monolayer vs Three-layer ( {label} )")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflectivity (dB)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    fn = f"compare_mono_vs_tri_{int(D*100)}cm.png"
    plt.savefig(fn, dpi=300)
    # plt.show()
    print(f"[Saved] {fn}  |  Monolayer sigma* @1 GHz = {sigma_mono:.5f} S/m")
