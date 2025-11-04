import numpy as np
import matplotlib.pyplot as plt

eps0 = 8.854187817e-12
mu0  = 4 * np.pi * 1e-7
c0   = 1.0 / np.sqrt(eps0 * mu0)
eps_rp = 1.4

def R_mono(f_GHz, d_m, sigma):
    f = np.asarray(f_GHz) * 1e9
    w = 2*np.pi*f
    eps_r = eps_rp + 1j*sigma/(w*eps0)
    n = np.sqrt(eps_r)
    beta = (2*np.pi*f/c0)*n*d_m
    r01 = (1 - n)/(1 + n)
    r12 = -1.0
    rtot = (r01 + r12*np.exp(2j*beta)) / (1 + r01*r12*np.exp(2j*beta))
    return np.abs(rtot)**2

def R_tri(f_GHz, D_m, sigmas):
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


f_GHz = np.logspace(-1, 2, 1500)
d = 0.15
sigma_nom = 0.062

R_nom = R_mono(f_GHz, d, sigma_nom)
R_low = R_mono(f_GHz, d, 0.8*sigma_nom)
R_high= R_mono(f_GHz, d, 1.2*sigma_nom)

plt.figure(figsize=(8,5.2))
plt.semilogx(f_GHz, 10*np.log10(np.clip(R_nom, 1e-16, 1)), lw=2, label=f"Nominal σ={sigma_nom}")
plt.fill_between(
    f_GHz,
    10*np.log10(np.clip(np.minimum(R_low, R_high), 1e-16, 1)),
    10*np.log10(np.clip(np.maximum(R_low, R_high), 1e-16, 1)),
    alpha=0.20, label="±20% envelope"
)
plt.axhline(-10, ls="--", color="gray", label="-10 dB")
plt.axvline(1.0, ls=":",  color="gray", lw=0.8)
plt.text(1.03, -9.3, "1 GHz", fontsize=9, color="gray", va="bottom")
plt.title("Monolayer (d=15cm) — Reflectivity with ±20% σ uncertainty")
plt.xlabel("Frequency (GHz)"); plt.ylabel("Reflectivity (dB)")
plt.grid(True, which="both", ls=":")
plt.legend(); plt.tight_layout()
plt.savefig("uncertainty_mono_20pc.png", dpi=300)
print("[Saved] uncertainty_mono_20pc.png")


tri_sets = [
    ("D=20 cm", 0.20, (0.03939, 0.29015, 1.99993)),
    ("D=30 cm", 0.30, (0.02953, 0.02952, 10.00000)),
    ("D=40 cm", 0.40, (0.01454, 0.02689, 0.07440)),
]

for label, Dm, sig in tri_sets:
    S = np.array(sig)
    corners = []
    for s1 in [0.8*S[0], 1.2*S[0]]:
        for s2 in [0.8*S[1], 1.2*S[1]]:
            for s3 in [0.8*S[2], 1.2*S[2]]:
                corners.append((s1, s2, s3))
    R_nom = R_tri(f_GHz, Dm, S)
    R_stack = np.vstack([R_tri(f_GHz, Dm, c) for c in corners])
    R_min = R_stack.min(axis=0)
    R_max = R_stack.max(axis=0)

    plt.figure(figsize=(8,5.2))
    plt.semilogx(f_GHz, 10*np.log10(np.clip(R_nom, 1e-16, 1)), lw=2,
                 label=f"Nominal σ={tuple(S)}")
    plt.fill_between(
        f_GHz,
        10*np.log10(np.clip(R_min, 1e-16, 1)),
        10*np.log10(np.clip(R_max, 1e-16, 1)),
        alpha=0.20, label="±20% (corner envelope)"
    )
    plt.axhline(-10, ls="--", color="gray", label="-10 dB")
    plt.axvline(1.0, ls=":",  color="gray", lw=0.8)
    plt.text(1.03, -9.3, "1 GHz", fontsize=9, color="gray", va="bottom")
    plt.title(f"Three-layer {label} — Reflectivity with ±20% σ uncertainty")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Reflectivity (dB)")
    plt.grid(True, which="both", ls=":")
    plt.legend(); plt.tight_layout()
    out = f"uncertainty_tri_20pc_{int(Dm*100)}cm.png"
    plt.savefig(out, dpi=300)
    print(f"[Saved] {out}")
