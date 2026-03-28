import numpy as np
from pk_model import P_lcdm, P_sudat

data = np.load("data/mock_pk_data.npz")
k = data["k"]
Pk_obs = data["Pk"]
sigma = data["sigma"]

def chi2(model):
    return np.sum(((Pk_obs - model) / sigma)**2)

def loglike_sudat(theta):
    As, ns, xi = theta
    
    # physical priors (IMPORTANT)
    if not (0.1 < As < 5 and 0.9 < ns < 1.05 and 0 < xi < 3):
        return -np.inf
    
    model = P_sudat(k, As, ns, xi)
    return -0.5 * chi2(model)

def loglike_lcdm(theta):
    As, ns = theta
    
    if not (0.1 < As < 5 and 0.9 < ns < 1.05):
        return -np.inf
    
    model = P_lcdm(k, As, ns)
    return -0.5 * chi2(model)