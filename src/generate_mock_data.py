import numpy as np
from pk_model import P_sudat

np.random.seed(42)

k = np.logspace(-3, 0, 60)

Pk_true = P_sudat(k, As=1.0, ns=0.965, xi=1.0)

# realistic cosmology noise
sigma = 0.05 * Pk_true * (1 + 0.5*(k/0.2))

noise = sigma * np.random.randn(len(k))
Pk_obs = Pk_true + noise

np.savez("mock_pk_data.npz", k=k, Pk=Pk_obs, sigma=sigma)

print("Generated SUDAT-based mock data")