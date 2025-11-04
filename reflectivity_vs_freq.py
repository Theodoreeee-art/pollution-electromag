import numpy as np
import matplotlib.pyplot as plt

eps0 = 8.854187817e-12  # F/m
mu0  = 4 * np.pi * 1e-7 # H/m
c0   = 1.0 / np.sqrt(eps0 * mu0)
eta0 = np.sqrt(mu0 / eps0)  


def minus10dB_band(freq_GHz, R_linear):
    
    mask = (R_linear <= 0.1)
    intervals = []
    if not np.any(mask):
        return intervals
    idx = np.where(mask)[0]
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k != prev + 1:
            intervals.append((freq_GHz[start], freq_GHz[prev]))
            start = k
        prev = k
    intervals.append((freq_GHz[start], freq_GHz[prev]))
    return intervals

def reflectivity_monolayer_fresnel(f_GHz, d_m, eps_r_prime, sigma_Spm):
    f = np.asarray(f_GHz) * 1e9
    omega = 2 * np.pi * f
    eps_r = eps_r_prime + 1j * (sigma_Spm / (omega * eps0))  # losses (e^{-i ω t})
    n = np.sqrt(eps_r)
    beta = (2 * np.pi * f / c0) * n * d_m

    r01 = (1 - n) / (1 + n)   
    r12 = -1.0                
    r_tot = (r01 + r12 * np.exp(2j * beta)) / (1 + r01 * r12 * np.exp(2j * beta))
    return np.abs(r_tot) ** 2


def reflectivity_trilayer_fresnel(f_GHz, D_m, eps_r_prime, sigmas_Spm):
    f = np.asarray(f_GHz) * 1e9
    omega = 2 * np.pi * f
    d = D_m / 3.0

    s1, s2, s3 = sigmas_Spm
    eps1 = eps_r_prime + 1j * (s1 / (omega * eps0))
    eps2 = eps_r_prime + 1j * (s2 / (omega * eps0))
    eps3 = eps_r_prime + 1j * (s3 / (omega * eps0))

    n0 = 1.0
    n1, n2, n3 = np.sqrt(eps1), np.sqrt(eps2), np.sqrt(eps3)
    beta1 = (2 * np.pi * f / c0) * n1 * d
    beta2 = (2 * np.pi * f / c0) * n2 * d
    beta3 = (2 * np.pi * f / c0) * n3 * d

    
    def r(n_left, n_right):
        return (n_left - n_right) / (n_left + n_right)


    r34_eff = -1.0  
    r23 = r(n2, n3)
    r23_eff = (r23 + r34_eff * np.exp(2j * beta3)) / (1 + r23 * r34_eff * np.exp(2j * beta3))

    r12 = r(n1, n2)
    r12_eff = (r12 + r23_eff * np.exp(2j * beta2)) / (1 + r12 * r23_eff * np.exp(2j * beta2))

    r01 = r(n0, n1)
    r01_eff = (r01 + r12_eff * np.exp(2j * beta1)) / (1 + r01 * r12_eff * np.exp(2j * beta1))

    return np.abs(r01_eff) ** 2

# === Main ===
if __name__ == "__main__":
    
    f_GHz = np.logspace(-1, 2, 1500)
    eps_r_prime = 1.4

    
    d = 0.15            # 15 cm
    sigma_mono = 0.062  # S/m (≈ optimum at 1 GHz)
    R_mono = reflectivity_monolayer_fresnel(f_GHz, d, eps_r_prime, sigma_mono)
    Rm_dB = 10 * np.log10(R_mono)
    band10_mono = minus10dB_band(f_GHz, R_mono)

    plt.figure(figsize=(8, 5.2))
    plt.semilogx(f_GHz, Rm_dB, linewidth=2, label=f"Monolayer, d={d*100:.0f} cm, σ={sigma_mono} S/m")
    plt.axhline(-10, linestyle='--', color='gray', label='-10 dB threshold')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflectivity (dB)")
    plt.title("Reflectivity vs Frequency — Monolayer on PEC (normal incidence)")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("R_vs_f_monolayer_fresnel.png", dpi=300)

    print("== Monolayer ==")
    if band10_mono:
        for a, b in band10_mono:
            print(f"-10 dB band: [{a:.3f}, {b:.3f}] GHz")
    else:
        print("No -10 dB band over this range.")

    
    cases = [
        (0.20, (0.0394, 0.290, 2.00),  "D=20 cm"),
        (0.30, (0.0295, 0.0295, 10.0), "D=30 cm"),
        (0.40, (0.0145, 0.0269, 0.0744), "D=40 cm"),
    ]

    for D, sigmas, label in cases:
        R_tri = reflectivity_trilayer_fresnel(f_GHz, D, eps_r_prime, sigmas)
        Rt_dB = 10 * np.log10(R_tri)
        band10 = minus10dB_band(f_GHz, R_tri)

        plt.figure(figsize=(8, 5.2))
        plt.semilogx(f_GHz, Rt_dB, linewidth=2, label=f"Three-layer, {label}, σ={sigmas}")
        plt.axhline(-10, linestyle='--', color='gray', label='-10 dB threshold')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Reflectivity (dB)")
        plt.title(f"Reflectivity vs Frequency — Three-layer on PEC ({label})")
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        fn = f"R_vs_f_trilayer_{int(D*100)}cm_fresnel.png"
        plt.savefig(fn, dpi=300)

        print(f"== Three-layer ({label}) ==")
        if band10:
            for a, b in band10:
                print(f"-10 dB band: [{a:.3f}, {b:.3f}] GHz")
        else:
            print("No -10 dB band over this range.")
