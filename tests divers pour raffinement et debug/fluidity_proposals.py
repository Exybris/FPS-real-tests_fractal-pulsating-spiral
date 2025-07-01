
import numpy as np

def compute_fluidity_adaptive(variance_d2S, ref_value=100.0):
    """Fluidity avec normalisation adaptative"""
    return 1 - np.tanh(variance_d2S / ref_value)

def compute_fluidity_log(variance_d2S, scale=50.0):
    """Fluidity avec échelle logarithmique"""
    return np.exp(-variance_d2S / scale)

def compute_fluidity_gamma_aware(variance_d2S, gamma_history):
    """Fluidity qui prend en compte la stabilité de gamma"""
    base_fluidity = 1 / (1 + variance_d2S)
    
    # Calculer la stabilité de gamma
    if len(gamma_history) > 10:
        gamma_std = np.std(gamma_history[-50:])
        gamma_stability = 1 / (1 + gamma_std)
    else:
        gamma_stability = 1.0
    
    return base_fluidity * gamma_stability

def compute_fluidity_percentile(variance_d2S, variance_history):
    """Fluidity basée sur le percentile dans l'historique"""
    if len(variance_history) < 100:
        return 1 / (1 + variance_d2S)
    
    # Calculer le percentile de la variance actuelle
    percentile = np.sum(variance_history <= variance_d2S) / len(variance_history)
    
    # Fluidity inversement proportionnelle au percentile
    return 1 - percentile
