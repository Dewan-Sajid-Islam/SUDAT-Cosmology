import numpy as np

def transfer_function(k):
    # Simple Eisenstein-Hu inspired shape
    keq = 0.01
    return 1.0 / (1.0 + (k/keq)**2)

def growth_factor():
    return 1.0  # keep simple for now

def P_lcdm(k, As, ns):
    T = transfer_function(k)
    D = growth_factor()
    
    return As * k**ns * T**2 * D**2

def P_sudat(k, As, ns, xi):
    T = transfer_function(k)
    D = growth_factor()
    
    # scale-dependent transition feature
    k_c = 0.1

    transition = np.tanh((k - k_c) / (0.3 * k_c))

    # sharper, more distinctive feature
    modulation = np.exp(-((k - k_c)**2) / (2 * (0.08 * k_c)**2))

    # stronger but still controlled signal
    modification = np.exp(xi * 0.5 * transition * modulation)
    
    return As * k**ns * T**2 * D**2 * modification