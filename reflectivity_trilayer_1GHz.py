import numpy as np
from scipy.optimize import minimize

eps0   = 8.854187817e-12  # F/m
c0     = 3e8
eps_rp = 1.4
f      = 1e9
omega  = 2 * np.pi * f

def layer_params(sigma):
    """Retourne (n, k) pour une couche donnée : indice complexe et nombre d’onde.
    Convention temporelle e^{-i ω t} => ε̃ = ε' + i σ/(ωε0)."""
    eps_r = eps_rp + 1j * sigma / (omega * eps0)  # pertes (e^{-i ω t})
    n = np.sqrt(eps_r)
    k = omega / c0 * n
    return n, k

def rtot_trilayer(sigmas, D):
    """Coefficient de réflexion complexe r_tot pour tri-couche sur PEC (épaisseurs égales D/3)."""
    sigma1, sigma2, sigma3 = sigmas
    d = D / 3.0
    n0 = 1.0  # air
    n1, k1 = layer_params(sigma1)
    n2, k2 = layer_params(sigma2)
    n3, k3 = layer_params(sigma3)

    beta1 = k1 * d
    beta2 = k2 * d
    beta3 = k3 * d

    # Coefficient de Fresnel à une interface (incidence normale)
    def r(nL, nR):
        return (nL - nR) / (nL + nR)

    # Remontée récursive depuis le PEC
    r34_eff = -1.0  # couche 3 / PEC
    r23 = r(n2, n3)
    r23_eff = (r23 + r34_eff * np.exp(2j * beta3)) / (1 + r23 * r34_eff * np.exp(2j * beta3))

    r12 = r(n1, n2)
    r12_eff = (r12 + r23_eff * np.exp(2j * beta2)) / (1 + r12 * r23_eff * np.exp(2j * beta2))

    r01 = r(n0, n1)
    r01_eff = (r01 + r12_eff * np.exp(2j * beta1)) / (1 + r01 * r12_eff * np.exp(2j * beta1))

    return r01_eff  # c'est r_tot

def R_trilayer(sigmas, D):
    """Réflectivité R = |r_tot|^2."""
    r_tot = rtot_trilayer(sigmas, D)
    return np.abs(r_tot)**2

def objective(sigmas, D):
    R = R_trilayer(sigmas, D)
    # (Option) pénalités d’ordre si tu veux les remettre un jour :
    # penalty = 1e3 * max(0, sigmas[0] - sigmas[1])**2 + 1e3 * max(0, sigmas[1] - sigmas[2])**2
    return R  # + penalty

for D in [0.20, 0.30, 0.40]:
    if D == 0.20:
        x0 = [0.05, 0.3, 2.0]
    elif D == 0.30:
        x0 = [0.005, 0.05, 0.5]
    else:
        x0 = [0.01, 0.03, 0.1]

    bounds = [(1e-4, 120)] * 3

    res = minimize(objective, x0=x0, args=(D,), method='L-BFGS-B', bounds=bounds)
    if not res.success:
        print(f"Attention : optimisation pour D={D*100:.0f} cm non parfaitement convergée.")

    sigma1, sigma2, sigma3 = res.x
    r_tot = rtot_trilayer(res.x, D)
    R_opt = np.abs(r_tot)**2
    R_dB = 10 * np.log10(R_opt)

    # Module/phase utiles pour reporting
    r_mod = np.abs(r_tot)
    r_arg_deg = np.degrees(np.angle(r_tot))

    print(f"----- D = {D*100:.0f} cm -----")
    print(f"σ1 = {sigma1:.5f} S/m")
    print(f"σ2 = {sigma2:.5f} S/m")
    print(f"σ3 = {sigma3:.5f} S/m")
    print(f"r_tot = {r_tot.real:+.4e} {r_tot.imag:+.4e}j   |r|={r_mod:.4e}, ∠r={r_arg_deg:.2f}°")
    print(f"R_min = {R_opt:.3e}  →  {R_dB:.2f} dB\n")

    # (Option) Si tu veux l'impédance/Robin équivalent pour Helmholtz :
    # k0 = omega / c0
    # alpha_eq = 1j * k0 * (1 - r_tot) / (1 + r_tot)
    # print(f"alpha_eq = {alpha_eq.real:+.4e} {alpha_eq.imag:+.4e}j")
