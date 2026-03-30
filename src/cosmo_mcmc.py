import numpy as np
import emcee
from tqdm import tqdm

# -----------------------------
# Mock cosmological "data"
# (Replace later with real data)
# -----------------------------
z_data = np.linspace(0, 2, 30)
H_obs = 70 * np.sqrt(0.3*(1+z_data)**3 + 0.7)
H_err = 3.0 * np.ones_like(H_obs)

# -----------------------------
# SUDAT expansion model
# -----------------------------
def H_model(z, H0, Omega_b, xi):
    Omega_phi = 1.0 - Omega_b
    transition = np.exp(-xi * z)
    return H0 * np.sqrt(Omega_b*(1+z)**3 + Omega_phi*(1 - transition + 1e-6))

# -----------------------------
# Likelihood
# -----------------------------
def log_likelihood(theta):
    H0, Omega_b, xi = theta
    
    if not (50 < H0 < 90 and 0.01 < Omega_b < 0.1 and 0 <= xi < 0.2):
        return -np.inf
    
    H_pred = H_model(z_data, H0, Omega_b, xi)
    chi2 = np.sum(((H_obs - H_pred) / H_err)**2)
    
    return -0.5 * chi2

# -----------------------------
# MCMC setup
# -----------------------------
ndim = 3
nwalkers = 32
steps = 4000

initial = np.array([70, 0.05, 0.02])
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)

print("Running cosmology MCMC...")
for _ in tqdm(sampler.sample(pos, iterations=steps), total=steps):
    pass

samples = sampler.get_chain(discard=1000, thin=10, flat=True)

xi_samples = samples[:, 2]

print("\nCosmo xi result:")
print(np.mean(xi_samples), "±", np.std(xi_samples))

np.save("xi_cosmo.npy", xi_samples)