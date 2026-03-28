import numpy as np
import matplotlib.pyplot as plt

k = np.linspace(0.01, 1, 100)

Pk_lcdm = np.exp(-k)
Pk_sudat = Pk_lcdm * (1 - 0.1*np.exp(-(k/0.1)**2))

plt.plot(k, Pk_lcdm, label="ΛCDM")
plt.plot(k, Pk_sudat, label="SUDAT")

plt.xlabel("k")
plt.ylabel("P(k)")
plt.legend()

plt.savefig("pk_comparison.pdf")
plt.show()