from likelihood import loglike_sudat, loglike_lcdm
from dynesty import NestedSampler
import numpy as np

# ---------- SUDAT ----------
def prior_sudat(u):
    As = 0.1 + 4.9*u[0]
    ns = 0.9 + 0.15*u[1]
    xi = 3.0 * u[2]
    return [As, ns, xi]

sampler = NestedSampler(loglike_sudat, prior_sudat, ndim=3)
sampler.run_nested()
logZ_sudat = sampler.results.logz[-1]

# ---------- LCDM ----------
def prior_lcdm(u):
    As = 0.1 + 4.9*u[0]
    ns = 0.9 + 0.15*u[1]
    return [As, ns]

sampler_lcdm = NestedSampler(loglike_lcdm, prior_lcdm, ndim=2)
sampler_lcdm.run_nested()
logZ_lcdm = sampler_lcdm.results.logz[-1]

lnB = logZ_sudat - logZ_lcdm

print("\n===== RESULTS =====")
print("logZ_sudat =", logZ_sudat)
print("logZ_lcdm  =", logZ_lcdm)
print("lnB =", lnB)