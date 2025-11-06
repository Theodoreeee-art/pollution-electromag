import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm  # progress bar (pip install tqdm)

# === Physical constants ===
eps0 = 8.854187817e-12  # F/m
c0 = 3e8
eps_rp = 1.4
f = 1e9
omega = 2 * np.pi * f


# === Physical functions ===
def layer_params(sigma):
    """Return (n, k) for a given layer: complex refractive index and wavenumber."""
    eps_r = eps_rp + 1j * sigma / (omega * eps0)
    n = np.sqrt(eps_r)
    k = omega / c0 * n
    return n, k


def R_trilayer(sigmas, D):
    """Compute reflectivity R for a tri-layer over PEC (equal thicknesses D/3)."""
    sigma1, sigma2, sigma3 = sigmas
    d = D / 3.0
    n0 = 1.0  # air
    n1, k1 = layer_params(sigma1)
    n2, k2 = layer_params(sigma2)
    n3, k3 = layer_params(sigma3)
    beta1, beta2, beta3 = k1 * d, k2 * d, k3 * d

    # Fresnel coefficients
    def r(nL, nR):
        return (nL - nR) / (nL + nR)

    r34_eff = -1.0  # layer 3 / PEC
    r23 = r(n2, n3)
    r23_eff = (r23 + r34_eff * np.exp(2j * beta3)) / (1 + r23 * r34_eff * np.exp(2j * beta3))
    r12 = r(n1, n2)
    r12_eff = (r12 + r23_eff * np.exp(2j * beta2)) / (1 + r12 * r23_eff * np.exp(2j * beta2))
    r01 = r(n0, n1)
    r01_eff = (r01 + r12_eff * np.exp(2j * beta1)) / (1 + r01 * r12_eff * np.exp(2j * beta1))
    R = np.abs(r01_eff) ** 2
    return R


def objective(sigmas, D):
    """Objective function: minimize reflectivity."""
    return R_trilayer(sigmas, D)


# === Thickness sweep ===
Ds = np.linspace(0.10, 1.00, 200)  # 200 points between 10 cm and 1 m
Rmins = []

print("Optimizing conductivities for each total thickness...\n")

for D in tqdm(Ds):
    # Initial guess depending on D
    if D < 0.25:
        x0 = [0.05, 0.3, 2.0]
    elif D < 0.5:
        x0 = [0.01, 0.05, 0.5]
    else:
        x0 = [0.01, 0.03, 0.1]

    bounds = [(1e-4, 120)] * 3
    res = minimize(objective, x0=x0, args=(D,), method='L-BFGS-B', bounds=bounds)
    if not res.success:
        tqdm.write(f"Optimization did not fully converge for D={D:.2f} m")

    R_opt = R_trilayer(res.x, D)
    Rmins.append(R_opt)


# === Plotting ===
Rmins = np.array(Rmins)
Rmins_dB = 10 * np.log10(Rmins)

plt.figure(figsize=(9, 5))
plt.plot(Ds * 100, Rmins_dB, '-', color='navy', linewidth=2)
plt.xlabel("Total thickness D (cm)")
plt.ylabel("Minimum reflectivity (dB)")
plt.title("Evolution of the minimum reflectivity versus total thickness (1 GHz)")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
