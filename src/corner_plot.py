import numpy as np
import corner
import matplotlib.pyplot as plt

samples = np.random.normal(loc=[68, 0.048, -4], scale=[0.5, 0.002, 0.1], size=(10000,3))

figure = corner.corner(samples, labels=["H0","Omega_b","log10 xi"])

plt.savefig("corner.png")
plt.show()