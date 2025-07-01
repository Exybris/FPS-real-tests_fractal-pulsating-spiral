"""
Analyse des propositions de fluidité et nouvelle approche physique
"""

import numpy as np
import matplotlib.pyplot as plt

# Analyse des propositions existantes
def analyze_fluidity_proposals():
    """Analyse comparative des différentes formules de fluidité"""
    
    # Plage de variance_d2S observée dans les données
    variance_range = np.linspace(164, 189, 100)
    
    # 1. Formule actuelle (problématique)
    fluidity_current = 1 / (1 + variance_range)
    
    # 2. Adaptive tanh (ref_value=100)
    fluidity_adaptive = 1 - np.tanh(variance_range / 100.0)
    
    # 3. Log scale (scale=50)
    fluidity_log = np.exp(-variance_range / 50.0)
    
    # 4. Percentile (simulé avec distribution uniforme)
    percentiles = (variance_range - variance_range.min()) / (variance_range.max() - variance_range.min())
    fluidity_percentile = 1 - percentiles
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(variance_range, fluidity_current, 'r-', linewidth=2)
    plt.title('Formule actuelle: 1/(1+x)')
    plt.xlabel('variance_d2S')
    plt.ylabel('fluidity')
    plt.grid(True, alpha=0.3)
    variation_current = fluidity_current.max() - fluidity_current.min()
    plt.text(0.5, 0.9, f'Variation: {variation_current:.6f}', transform=plt.gca().transAxes)
    
    plt.subplot(2, 2, 2)
    plt.plot(variance_range, fluidity_adaptive, 'g-', linewidth=2)
    plt.title('Adaptive: 1 - tanh(x/100)')
    plt.xlabel('variance_d2S')
    plt.ylabel('fluidity')
    plt.grid(True, alpha=0.3)
    variation_adaptive = fluidity_adaptive.max() - fluidity_adaptive.min()
    plt.text(0.5, 0.9, f'Variation: {variation_adaptive:.6f}', transform=plt.gca().transAxes)
    
    plt.subplot(2, 2, 3)
    plt.plot(variance_range, fluidity_log, 'b-', linewidth=2)
    plt.title('Log: exp(-x/50)')
    plt.xlabel('variance_d2S')
    plt.ylabel('fluidity')
    plt.grid(True, alpha=0.3)
    variation_log = fluidity_log.max() - fluidity_log.min()
    plt.text(0.5, 0.9, f'Variation: {variation_log:.6f}', transform=plt.gca().transAxes)
    
    plt.subplot(2, 2, 4)
    plt.plot(variance_range, fluidity_percentile, 'm-', linewidth=2)
    plt.title('Percentile: 1 - percentile(x)')
    plt.xlabel('variance_d2S')
    plt.ylabel('fluidity')
    plt.grid(True, alpha=0.3)
    variation_percentile = fluidity_percentile.max() - fluidity_percentile.min()
    plt.text(0.5, 0.9, f'Variation: {variation_percentile:.6f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('fluidity_proposals_comparison.png', dpi=150)
    plt.close()
    
    return {
        'current': variation_current,
        'adaptive': variation_adaptive,
        'log': variation_log,
        'percentile': variation_percentile
    }


def compute_fluidity_physical(variance_d2S, S_history=None, dt=0.1):
    """
    Nouvelle approche physique de la fluidité basée sur plusieurs composantes
    
    La fluidité devrait capturer:
    1. Lissage des transitions (variance_d2S)
    2. Continuité temporelle (autocorrélation)
    3. Absence de discontinuités (détection de sauts)
    4. Régularité spectrale (pas de hautes fréquences parasites)
    
    Args:
        variance_d2S: variance de la dérivée seconde
        S_history: historique du signal pour analyse avancée
        dt: pas de temps
    
    Returns:
        float: fluidité entre 0 et 1
    """
    
    # Composante 1: Lissage (variance normalisée)
    # Utiliser une échelle adaptative basée sur l'ordre de grandeur
    scale = 10 ** np.floor(np.log10(variance_d2S + 1))
    smoothness = np.exp(-variance_d2S / scale)
    
    # Si pas d'historique, retourner seulement le lissage
    if S_history is None or len(S_history) < 10:
        return smoothness
    
    # Composante 2: Continuité temporelle (autocorrélation)
    S_array = np.array(S_history[-100:])  # Derniers 100 points
    if len(S_array) > 1:
        # Autocorrélation à lag 1
        autocorr = np.corrcoef(S_array[:-1], S_array[1:])[0, 1]
        continuity = (autocorr + 1) / 2  # Normaliser entre 0 et 1
    else:
        continuity = 1.0
    
    # Composante 3: Absence de discontinuités
    if len(S_array) > 2:
        # Détecter les sauts (dérivée première)
        dS = np.diff(S_array)
        # Seuil adaptatif: 3 écarts-types
        threshold = 3 * np.std(dS)
        n_jumps = np.sum(np.abs(dS) > threshold)
        jump_penalty = np.exp(-n_jumps / 10)  # Pénalité exponentielle
    else:
        jump_penalty = 1.0
    
    # Composante 4: Régularité spectrale (optionnel, coûteux)
    if len(S_array) >= 50:
        # FFT pour analyser le contenu fréquentiel
        fft = np.fft.fft(S_array)
        freqs = np.fft.fftfreq(len(S_array), dt)
        
        # Ratio énergie basse fréquence / haute fréquence
        mid_freq = len(freqs) // 4
        low_energy = np.sum(np.abs(fft[:mid_freq])**2)
        high_energy = np.sum(np.abs(fft[mid_freq:])**2)
        
        if high_energy > 0:
            spectral_regularity = low_energy / (low_energy + high_energy)
        else:
            spectral_regularity = 1.0
    else:
        spectral_regularity = 1.0
    
    # Combinaison pondérée des composantes
    weights = {
        'smoothness': 0.4,      # Priorité au lissage
        'continuity': 0.3,      # Continuité temporelle
        'jump_penalty': 0.2,    # Absence de sauts
        'spectral': 0.1         # Régularité spectrale
    }
    
    fluidity = (
        weights['smoothness'] * smoothness +
        weights['continuity'] * continuity +
        weights['jump_penalty'] * jump_penalty +
        weights['spectral'] * spectral_regularity
    )
    
    return fluidity


def compute_fluidity_simplified(variance_d2S, reference_variance=None):
    """
    Version simplifiée mais efficace de la fluidité
    
    Utilise une normalisation intelligente qui s'adapte à l'échelle des données
    tout en préservant une bonne sensibilité aux variations.
    
    Args:
        variance_d2S: variance de la dérivée seconde
        reference_variance: variance de référence (médiane historique par exemple)
    
    Returns:
        float: fluidité entre 0 et 1
    """
    
    if reference_variance is None:
        # Estimation automatique de l'échelle
        # Pour les valeurs observées (164-189), on prend 175 comme référence
        reference_variance = 175.0
    
    # Normalisation avec fonction sigmoïde inversée
    # Plus sensible que 1/(1+x) mais plus stable que exp(-x)
    x = variance_d2S / reference_variance
    
    # Fonction sigmoïde avec pente ajustable
    k = 5.0  # Contrôle la pente (plus k est grand, plus la transition est abrupte)
    fluidity = 1 / (1 + np.exp(k * (x - 1)))
    
    return fluidity


# Test des propositions
if __name__ == "__main__":
    # Analyser les propositions
    variations = analyze_fluidity_proposals()
    
    print("Analyse des variations de fluidité:")
    print("-" * 40)
    for name, var in variations.items():
        print(f"{name:15s}: {var:.6f}")
    
    # Tester la nouvelle approche physique
    print("\nNouvelle approche physique:")
    print("-" * 40)
    
    # Simuler différents cas
    test_cases = [
        (164, "Très fluide (variance min)"),
        (175, "Fluide moyen"),
        (189, "Peu fluide (variance max)")
    ]
    
    for var, desc in test_cases:
        fluidity_simple = compute_fluidity_simplified(var)
        print(f"{desc:30s}: variance={var:3.0f}, fluidity={fluidity_simple:.4f}")
    
    # Comparaison sensibilité
    print("\nSensibilité des formules (variation sur plage 164-189):")
    print("-" * 50)
    
    # Formule actuelle
    f_min = 1/(1+164)
    f_max = 1/(1+189)
    print(f"Actuelle:    {f_min:.6f} → {f_max:.6f} (Δ={f_min-f_max:.6f})")
    
    # Nouvelle simplifiée
    f_min = compute_fluidity_simplified(164)
    f_max = compute_fluidity_simplified(189)
    print(f"Simplifiée:  {f_min:.6f} → {f_max:.6f} (Δ={f_min-f_max:.6f})")
    
    # Facteur d'amélioration
    improvement = (f_min - f_max) / 0.00007
    print(f"\nAmélioration: {improvement:.0f}x plus de sensibilité") 