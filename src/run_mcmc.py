import numpy as np
import emcee
from likelihood import loglike_sudat

# number of parameters
ndim = 3
nwalkers = 32

# initial guess (As, ns, xi)
initial = np.array([1.0, 0.96, 0.5])
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

def log_prior(theta):
    As, ns, xi = theta
    
    if 0.5 < As < 2.0 and 0.8 < ns < 1.2 and -2 < xi < 3:
        return 0.0
    return -np.inf

def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_sudat(theta)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

print("Running MCMC...")
sampler.run_mcmc(pos, 3000, progress=True)

samples = sampler.get_chain(discard=500, thin=10, flat=True)

np.save("mcmc_samples.npy", samples)

print("MCMC complete. Saved samples.")