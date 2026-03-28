import numpy as np
import matplotlib.pyplot as plt

k = np.linspace(0.01, 1, 100)

Pk_lcdm = np.exp(-k)
Pk_sudat = Pk_lcdm * (1 - 0.1*np.exp(-(k/0.1)**2))

delta = (Pk_sudat - Pk_lcdm)/Pk_lcdm

# simulate error bars (Euclid-like)
error = 0.01 * np.ones_like(k)

plt.errorbar(k, delta, yerr=error, fmt='o', markersize=3)
plt.axhline(0, linestyle='--')
plt.xlabel("k [Mpc^-1]")
plt.ylabel("ΔP/P")
plt.title("SUDAT signature vs ΛCDM")

plt.savefig("delta_pk.pdf")
plt.show()