import numpy as np
import corner
import matplotlib.pyplot as plt

samples = np.load("mcmc_samples.npy")

labels = ["As", "ns", "xi"]

fig = corner.corner(samples, labels=labels, show_titles=True)
plt.savefig("corner.png")