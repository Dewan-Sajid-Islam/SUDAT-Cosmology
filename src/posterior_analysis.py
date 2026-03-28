import numpy as np

# fake posterior samples
xi_samples = 10**np.random.normal(-4, 0.1, 10000)

prior_width = 10**(-3) - 10**(-5)
posterior_width = np.std(xi_samples)

ratio = posterior_width / prior_width

print("Posterior width:", posterior_width)
print("Prior width:", prior_width)
print("Collapse ratio:", ratio)