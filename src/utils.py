import numpy as np

# -----------------------------
# Cosmology model
# -----------------------------
def H_model(z, H0, xi):
    return H0 * np.sqrt((1 + z)**3 * (1 - xi) + xi)

z_data = np.linspace(0.01, 1.0, 20)
H_obs = 70 * np.sqrt((1 + z_data)**3)
H_err = 2.0 * np.ones_like(H_obs)

# -----------------------------
# Galaxy physics (UPGRADED)
# -----------------------------
def v_b(r):
    return 50 * (1 - np.exp(-r / 5))

def v_phi(r, xi, rs=5.0):
    r = np.maximum(r, 1e-3)  # avoid division issues
    return np.sqrt(xi * (1 - np.exp(-r / rs)) / r)