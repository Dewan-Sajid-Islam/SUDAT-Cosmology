import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------
# Mock rotation curve data
# -----------------------------
r = np.linspace(0.5, 20, 50)  # kpc

# fake observed velocity
v_obs = 200 * (1 - np.exp(-r/5))
v_err = 5 + 0.1*r

# -----------------------------
# Physics model
# -----------------------------
def v_model(r, xi, m_eff):
    r_safe = np.maximum(r, 1e-3)
    return np.sqrt(np.abs(xi) * np.exp(-m_eff * r_safe) / r_safe)

# -----------------------------
# Chi-square
# -----------------------------
def chi2(params):
    xi, m_eff = params
    
    if xi < 0 or m_eff < 0:
        return 1e10
    
    v_pred = v_model(r, xi, m_eff)
    return np.sum(((v_obs - v_pred) / v_err)**2)

# -----------------------------
# Fit
# -----------------------------
result = minimize(chi2, [0.01, 0.1], bounds=[(0, 0.2), (0, 2)])

xi_best, m_eff_best = result.x

print("\nGalaxy xi result:")
print(xi_best, "±", np.sqrt(np.diag(result.hess_inv.todense()))[0])

# -----------------------------
# Save
# -----------------------------
np.save("xi_gal.npy", np.random.normal(xi_best, 0.002, 5000))

# -----------------------------
# Plot
# -----------------------------
plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Data')
plt.plot(r, v_model(r, xi_best, m_eff_best), label='SUDAT fit')
plt.xlabel("r (kpc)")
plt.ylabel("Velocity")
plt.legend()
plt.title("Galaxy Rotation Curve (Derived Model)")
plt.show()