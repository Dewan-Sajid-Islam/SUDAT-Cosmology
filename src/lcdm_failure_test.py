import numpy as np
from scipy.optimize import minimize
from likelihood import loglike_lcdm, loglike_sudat

# -------------------------------
# Safe wrapper (CRITICAL FIX)
# -------------------------------
def safe_neg_loglike_sudat(x):
    val = loglike_sudat(x)
    if not np.isfinite(val):
        return 1e10
    return -val

def safe_neg_loglike_lcdm(x):
    val = loglike_lcdm(x)
    if not np.isfinite(val):
        return 1e10
    return -val

# -------------------------------
# Initial guesses
# -------------------------------
init_sudat = [1.0, 0.96, 0.5]
init_lcdm  = [1.0, 0.96]

# -------------------------------
# Bounds
# -------------------------------
bounds_sudat = [
    (0.5, 2.0),     # As (tightened)
    (0.92, 1.0),    # ns (realistic)
    (0.0, 1.5)      # xi (CRITICAL: reduced)
]

bounds_lcdm = [
    (0.5, 2.0),
    (0.92, 1.0)
]

# -------------------------------
# Fit SUDAT
# -------------------------------
res_sudat = minimize(
    safe_neg_loglike_sudat,
    init_sudat,
    bounds=bounds_sudat,
    method="L-BFGS-B"
)

chi2_sudat = -2 * loglike_sudat(res_sudat.x)

# -------------------------------
# Fit LCDM
# -------------------------------
res_lcdm = minimize(
    safe_neg_loglike_lcdm,
    init_lcdm,
    bounds=bounds_lcdm,
    method="L-BFGS-B"
)

chi2_lcdm = -2 * loglike_lcdm(res_lcdm.x)

# -------------------------------
# Compare
# -------------------------------
delta_chi2 = chi2_lcdm - chi2_sudat

# -------------------------------
# Output
# -------------------------------
print("\n===== FIT RESULTS =====")

print("\nSUDAT best-fit parameters:")
print(f"As = {res_sudat.x[0]:.4f}, ns = {res_sudat.x[1]:.4f}, xi = {res_sudat.x[2]:.4f}")
print(f"chi2_sudat = {chi2_sudat:.3f}")

print("\nLCDM best-fit parameters:")
print(f"As = {res_lcdm.x[0]:.4f}, ns = {res_lcdm.x[1]:.4f}")
print(f"chi2_lcdm = {chi2_lcdm:.3f}")

print("\n===== COMPARISON =====")
print(f"delta_chi2 = {delta_chi2:.3f}")

if delta_chi2 > 10:
    print("🔥 Strong evidence: LCDM fails to reproduce SUDAT")
elif delta_chi2 > 5:
    print("⚡ Moderate evidence: tension with LCDM")
elif delta_chi2 > 0:
    print("⚠️ Weak preference for SUDAT")
else:
    print("❌ LCDM still fits as well or better")