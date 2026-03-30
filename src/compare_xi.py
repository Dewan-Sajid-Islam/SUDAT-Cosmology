import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load results
# -----------------------------
xi_cosmo = np.load("xi_cosmo.npy")
xi_gal = np.load("xi_gal.npy")

# -----------------------------
# Stats
# -----------------------------
mean_cosmo = np.mean(xi_cosmo)
std_cosmo = np.std(xi_cosmo)

mean_gal = np.mean(xi_gal)
std_gal = np.std(xi_gal)

# significance
sigma = np.abs(mean_cosmo - mean_gal) / np.sqrt(std_cosmo**2 + std_gal**2)

print("\n=== CROSS-SCALE CONSISTENCY TEST ===")
print(f"xi_cosmo = {mean_cosmo:.5f} ± {std_cosmo:.5f}")
print(f"xi_gal   = {mean_gal:.5f} ± {std_gal:.5f}")
print(f"Difference significance = {sigma:.2f} sigma")

# -----------------------------
# Plot
# -----------------------------
plt.hist(xi_cosmo, bins=50, alpha=0.6, label="Cosmology", density=True)
plt.hist(xi_gal, bins=50, alpha=0.6, label="Galaxy", density=True)

plt.axvline(mean_cosmo, linestyle='--')
plt.axvline(mean_gal, linestyle='--')

plt.xlabel("xi")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Cross-Scale Consistency Test (Derived Physics)")
plt.show()