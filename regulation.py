"""
regulation.py - Fonctions de régulation adaptative FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Les formes de G(x), envₙ(x,t) sont à adapter selon l'expérience.
Ne jamais considérer la version présente comme définitive.
---------------------------------------------------------------

Ce module implémente les fonctions de régulation du système FPS :
- Archétypes de régulation G(x) : tanh, sinc, resonance, adaptive
- Réponse locale Gₙ(x,t) avec enveloppes adaptatives
- Version temporelle G(x,t) avec modulation contextuelle
- Enveloppes gaussiennes/sigmoïdes avec modes statique/dynamique

La régulation est le cœur de l'auto-organisation FPS : elle transforme
l'écart entre attendu et observé en correction douce et spiralée.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
from typing import Dict, Union, Optional, Any, Tuple
import warnings


# ============== ARCHÉTYPES DE RÉGULATION G(x) ==============

def compute_G(x: Union[float, np.ndarray], archetype: str = "tanh", 
              params: Optional[Dict[str, float]] = None) -> Union[float, np.ndarray]:
    """
    Calcule la fonction de régulation selon l'archétype choisi.
    
    La régulation transforme l'erreur (Eₙ - Oₙ) en signal de correction.
    Chaque archétype a ses propriétés : saturation, oscillation, résonance...
    
    Args:
        x: valeur(s) d'entrée (typiquement Eₙ - Oₙ)
        archetype: type de fonction parmi ["tanh", "sinc", "resonance", "spiral_log", "adaptive"]
        params: paramètres spécifiques à chaque archétype
    
    Returns:
        Valeur(s) de régulation G(x)
    
    Archétypes:
        - "tanh": tanh(λx) - Saturation douce, transition continue
        - "sinc": sin(x)/x - Oscillations amorties, passage par zéro
        - "resonance": sin(βx)·exp(-αx²) - Résonance localisée
        - "spiral_log": sign(x)·log(1+α|x|)·sin(βx) - Spirale logarithmique (FPS Paper)
        - "adaptive": Forme à définir selon contexte
    """
    if params is None:
        params = {}
    
    if archetype == "tanh":
        # Tangente hyperbolique : saturation douce aux extrêmes
        lambda_val = params.get("lambda", 1.0)
        return np.tanh(lambda_val * x)
    
    elif archetype == "sinc":
        # Sinus cardinal : oscillations qui s'amortissent
        # Protection contre division par zéro
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(x != 0, np.sin(x) / x, 1.0)
        return result
    
    elif archetype == "resonance":
        # Résonance gaussienne modulée : pic local avec décroissance
        alpha = params.get("alpha", 1.0)  # Largeur de la gaussienne
        beta = params.get("beta", 2.0)    # Fréquence d'oscillation
        return np.sin(beta * x) * np.exp(-alpha * x**2)
    
    elif archetype == "spiral_log":
        # Spirale logarithmique selon FPS Paper
        # G(x) = sign(x) · log(1 + α·|x|) · sin(β·x)
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 2.0)
        
        # Calcul par composants pour gérer les arrays
        sign_x = np.sign(x)
        abs_x = np.abs(x)
        
        log_component = np.log(1 + alpha * abs_x)
        sin_component = np.sin(beta * x)
        
        return sign_x * log_component * sin_component
    
    elif archetype == "adaptive":
        # Combinaison adaptative de tanh et spiral_log
        lambda_val = params.get("lambda", 1.0)
        alpha = params.get("alpha", 0.5)
        
        # Calcul des deux composantes
        tanh_part = np.tanh(lambda_val * x)
        spiral_part = compute_G(x, "spiral_log", params)
        
        # Mélange pondéré
        return alpha * tanh_part + (1 - alpha) * spiral_part
    
    elif archetype == "adaptive_aware":
        # Mode adaptatif conscient - utilise dynamics.compute_G_adaptive_aware
        # Pour éviter une dépendance circulaire, on retourne simplement tanh ici
        # La vraie logique adaptive_aware est dans dynamics.py
        lambda_val = params.get("lambda", 1.0)
        return np.tanh(lambda_val * x)
    
    else:
        # Archétype non reconnu - fallback sur tanh
        warnings.warn(f"Archétype '{archetype}' non reconnu. Utilisation de 'tanh' par défaut.")
        return compute_G(x, "tanh", params)


def adapt_params_for_archetype(G_arch: str, gamma: float, error_magnitude: float) -> Dict[str, float]:
    """
    Adapte les paramètres selon l'archétype, γ et l'erreur.
    Utilisé par compute_G_adaptive_aware pour ajuster dynamiquement les paramètres.
    
    Args:
        G_arch: archétype de régulation
        gamma: valeur actuelle de gamma
        error_magnitude: magnitude de l'erreur
        
    Returns:
        dict avec paramètres adaptés pour l'archétype
    """
    if G_arch == "tanh":
        return {"lambda": 0.5 + gamma * 0.5}
    
    elif G_arch == "resonance":
        return {
            "alpha": 1.0 - 0.3 * error_magnitude,
            "beta": 1.5 * gamma + 0.5
        }
    
    elif G_arch == "spiral_log":
        return {
            "alpha": gamma + 0.2 * error_magnitude,
            "beta": 3.0 - gamma
        }
    
    elif G_arch == "adaptive":
        return {
            "lambda": gamma,
            "alpha": 0.5 + 0.5 * (1 - error_magnitude)
        }
    
    else:
        return {"lambda": 1.0}


# ============== ENVELOPPES ADAPTATIVES ==============

def compute_sigma_n(t: float, mode: str = "static", T: Optional[float] = None,
                    sigma_n_static: float = 0.1, sigma_n_dynamic: Optional[Dict] = None) -> float:
    """
    Calcule l'écart-type de l'enveloppe.
    
    σₙ(t) contrôle la largeur de l'enveloppe gaussienne ou sigmoïde.
    
    Args:
        t: temps actuel
        mode: "static" ou "dynamic"
        T: période totale (pour mode dynamic)
        sigma_n_static: valeur statique par défaut
        sigma_n_dynamic: paramètres dynamiques {amp, freq, offset}
    
    Returns:
        float: écart-type σₙ(t)
    """
    if mode == "static":
        return sigma_n_static
    
    elif mode == "dynamic" and T is not None and sigma_n_dynamic is not None:
        # Modulation sinusoïdale de l'écart-type
        amp = sigma_n_dynamic.get("amp", 0.05)
        freq = sigma_n_dynamic.get("freq", 1.0)
        offset = sigma_n_dynamic.get("offset", 0.1)
        
        # σₙ(t) = offset + amp·sin(2π·freq·t/T)
        # GARANTIR que σn > 0 avec un minimum absolu
        sigma_min = 0.01  # Écart-type minimum pour éviter division par 0
        raw_sigma = offset + amp * np.sin(2 * np.pi * freq * t / T)
        return max(raw_sigma, sigma_min)
    
    else:
        # Fallback sur statique
        return sigma_n_static


def compute_mu_n(t: float, mode: str = "static", mu_n_static: float = 0.0,
                 mu_n_dynamic: Optional[Dict] = None) -> float:
    """
    Calcule le centre de l'enveloppe.
    
    μₙ(t) déplace le centre de régulation, permettant un focus adaptatif.
    
    Args:
        t: temps actuel
        mode: "static" ou "dynamic"
        mu_n_static: valeur statique
        mu_n_dynamic: paramètres dynamiques (à définir phase 2)
    
    Returns:
        float: centre μₙ(t)
    """
    if mode == "static":
        return mu_n_static
    
    elif mode == "dynamic" and mu_n_dynamic is not None:
        # Mode dynamique - À définir en phase 2
        # DÉSACTIVÉ : drift_rate causait des dérives non désirées
        # drift_rate = mu_n_dynamic.get("drift_rate", 0.01)
        # max_drift = mu_n_dynamic.get("max_drift", 1.0)
        # 
        # # Dérive bornée
        # drift = drift_rate * t
        # return np.clip(drift, -max_drift, max_drift)
        
        # Pour l'instant, retourner la valeur statique même en mode dynamique
        return mu_n_static
    
    else:
        return mu_n_static


def compute_env_n(x: Union[float, np.ndarray], t: float, mode: str = "static",
                  sigma_n: float = 0.1, mu_n: float = 0.0, T: Optional[float] = None,
                  env_type: str = "gaussienne") -> Union[float, np.ndarray]:
    """
    Calcule l'enveloppe adaptative.
    
    L'enveloppe localise la régulation autour de μₙ avec une largeur σₙ.
    
    Args:
        x: valeur(s) d'entrée
        t: temps actuel
        mode: "static" ou "dynamic"
        sigma_n: écart-type
        mu_n: centre
        T: période totale (pour mode dynamic)
        env_type: "gaussienne" ou "sigmoide"
    
    Returns:
        Valeur(s) d'enveloppe entre 0 et 1
    """
    if env_type == "gaussienne":
        # Enveloppe gaussienne : exp(-(x-μₙ)²/(2σₙ²))
        if sigma_n > 0:
            return np.exp(-0.5 * ((x - mu_n) / sigma_n) ** 2)
        else:
            # Protection contre σₙ = 0
            return np.where(x == mu_n, 1.0, 0.0)
    
    elif env_type == "sigmoide":
        # Enveloppe sigmoïde : transition douce
        # Utilise σₙ comme paramètre de pente
        k = 1.0 / (sigma_n + 1e-10)  # Protection division par zéro
        return 1.0 / (1.0 + np.exp(-k * (x - mu_n)))
    
    else:
        # Type non reconnu - fallback gaussienne
        return compute_env_n(x, t, mode, sigma_n, mu_n, T, "gaussienne")


# ============== RÉPONSE LOCALE Gₙ(x,t) ==============

def compute_Gn(x: Union[float, np.ndarray], t: float, An_t: float, fn_t: float,
               mu_n_t: float, env_n: Union[float, np.ndarray],
               config: Optional[Dict] = None) -> Union[float, np.ndarray]:
    """
    Calcule la réponse harmonique locale d'une strate.
    
    Gₙ(x,t) = Aₙ(t)·sinc[fₙ(t)·(x-μₙ(t))]·envₙ(x,t)
    
    Cette fonction combine l'amplitude adaptative, la résonance fréquentielle
    et la localisation spatiale pour créer une régulation harmonique.
    
    Args:
        x: erreur ou signal d'entrée (typiquement Eₙ - Oₙ)
        t: temps actuel
        An_t: amplitude de la strate
        fn_t: fréquence de la strate
        mu_n_t: centre de régulation
        env_n: enveloppe pré-calculée
        config: configuration optionnelle
    
    Returns:
        Réponse locale Gₙ(x,t)
    
    Note:
        Dans simulate.py, cette fonction est appelée avec x = On_t - En_t
        pour calculer le feedback de régulation.
    """
    # Calcul du sinc décalé et modulé en fréquence
    arg = fn_t * (x - mu_n_t)
    
    # Protection contre division par zéro dans sinc
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_val = np.where(arg != 0, np.sin(arg) / arg, 1.0)
    
    # Réponse complète : amplitude × sinc × enveloppe
    return An_t * sinc_val * env_n

def compute_eta(t: float, config: Dict) -> float:
    """
    Amplitude contextuelle pour la régulation temporelle.
    Phase 1 : patterns simples pour tester
    """
    temporal_config = config.get('temporal_regulation', {})
    mode = temporal_config.get('eta_mode', 'constant')
    
    if mode == 'constant':
        return temporal_config.get('eta_value', 1.0)
    
    elif mode == 'oscillating':
        # Oscillation lente
        T = config['system']['T']
        freq = temporal_config.get('eta_freq', 0.1)
        amp = temporal_config.get('eta_amp', 0.5)
        offset = temporal_config.get('eta_offset', 0.5)
        return offset + amp * np.sin(2 * np.pi * freq * t / T)
    
    elif mode == 'adaptive':
        # S'adapte selon l'historique (phase 2)
        history = config.get('history', [])
        if len(history) > 10:
            # Exemple : augmente si l'erreur moyenne est élevée
            recent_errors = [h.get('mean_abs_error', 0) for h in history[-10:]]
            mean_error = np.mean(recent_errors)
            return 1.0 + 0.5 * np.tanh(mean_error - 0.5)
        return 1.0

def compute_theta(t: float, config: Dict) -> float:
    """
    Fréquence adaptative pour la régulation temporelle.
    """
    temporal_config = config.get('temporal_regulation', {})
    mode = temporal_config.get('theta_mode', 'constant')
    
    if mode == 'constant':
        return temporal_config.get('theta_value', 1.0)
    
    elif mode == 'ramping':
        # Augmentation progressive
        T = config['system']['T']
        rate = temporal_config.get('theta_rate', 0.01)
        return 1.0 + rate * t / T
    
    elif mode == 'resonant':
        # Cherche la résonance avec le système
        history = config.get('history', [])
        if len(history) > 0:
            # Utilise la fréquence moyenne du système
            last_state = history[-1]
            if 'f_mean' in last_state:
                return last_state['f_mean']
        return 1.0

# ============== VERSION TEMPORELLE G(x,t) ==============

def compute_G_temporal(x: Union[float, np.ndarray], t: float, 
                       eta_t: float, theta_t: float) -> Union[float, np.ndarray]:
    """
    Calcule la version temporelle de la régulation.
    
    G(x,t) = η(t)·sin(θ(t)·x)
    
    Cette forme permet une modulation temporelle de la régulation,
    avec amplitude η(t) et fréquence θ(t) variables.
    
    Args:
        x: valeur(s) d'entrée
        t: temps actuel
        eta_t: amplitude contextuelle
        theta_t: fréquence adaptative
    
    Returns:
        Régulation temporelle G(x,t)
    
    Note:
        η(t) et θ(t) sont des paramètres exploratoires phase 2.
        Pour phase 1, on peut utiliser des constantes.
    """
    return eta_t * np.sin(theta_t * x)


# ============== FONCTION INTÉGRÉE POUR SIMULATE.PY ==============
'''
def compute_Gn(error: Union[float, np.ndarray], t: float, An_t: Union[float, np.ndarray], 
               fn_t: Union[float, np.ndarray], config: Dict) -> Union[float, np.ndarray]:
    """
    Interface principale pour simulate.py - calcule la régulation complète.
    
    Cette fonction orchestre tous les calculs de régulation en utilisant
    la configuration pour déterminer les modes et paramètres.
    
    Args:
        error: erreur Oₙ(t) - Eₙ(t) (peut être scalaire ou array)
        t: temps actuel
        An_t: amplitude(s) actuelle(s)
        fn_t: fréquence(s) actuelle(s)
        config: configuration complète
    
    Returns:
        Feedback de régulation (même forme que error)
    """
    # Extraction des paramètres de configuration
    regulation_config = config.get('regulation', {})
    enveloppe_config = config.get('enveloppe', {})
    T = config.get('system', {}).get('T', 100)
    
    # Archétype de régulation
    G_arch = regulation_config.get('G_arch', 'tanh')
    G_params = {
        'lambda': regulation_config.get('lambda', 1.0),
        'alpha': regulation_config.get('alpha', 1.0),
        'beta': regulation_config.get('beta', 2.0)
    }
    
    # Mode enveloppe
    env_mode = enveloppe_config.get('env_mode', 'static')
    env_type = config.get('to_calibrate', {}).get('env_n', 'gaussienne')
    
    # Si error est un scalaire, on traite une seule strate
    if np.isscalar(error):
        # Calcul des paramètres d'enveloppe
        sigma_n = compute_sigma_n(
            t, env_mode, T,
            enveloppe_config.get('sigma_n_static', 0.1),
            enveloppe_config.get('sigma_n_dynamic')
        )
        mu_n = compute_mu_n(
            t, env_mode,
            enveloppe_config.get('mu_n', 0.0),
            enveloppe_config.get('mu_n_dynamic')
        )
        
        # Calcul de l'enveloppe
        env_n = compute_env_n(error, t, env_mode, sigma_n, mu_n, T, env_type)
        
        # Régulation de base
        if regulation_config.get('dynamic_G', False):
            # Mode dynamique avec Gₙ complet
            return compute_Gn(error, t, An_t, fn_t, mu_n, env_n)
        else:
            # Mode statique avec archétype simple
            return compute_G(error, G_arch, G_params)
    
    # Si error est un array, on traite toutes les strates
    else:
        N = len(error)
        result = np.zeros_like(error)
        
        # Vérifier que An_t et fn_t sont aussi des arrays
        if np.isscalar(An_t):
            An_t = np.full(N, An_t)
        if np.isscalar(fn_t):
            fn_t = np.full(N, fn_t)
        
        for n in range(N):
            # Calcul par strate
            sigma_n = compute_sigma_n(
                t, env_mode, T,
                enveloppe_config.get('sigma_n_static', 0.1),
                enveloppe_config.get('sigma_n_dynamic')
            )
            mu_n = compute_mu_n(
                t, env_mode,
                enveloppe_config.get('mu_n', 0.0),
                enveloppe_config.get('mu_n_dynamic')
            )
            
            env_n = compute_env_n(error[n], t, env_mode, sigma_n, mu_n, T, env_type)
            
            if regulation_config.get('dynamic_G', False):
                result[n] = compute_Gn(error[n], t, An_t[n], fn_t[n], mu_n, env_n)
            else:
                result[n] = compute_G(error[n], G_arch, G_params)
        
        return result
'''

# ============== FONCTIONS EXPLORATOIRES PHASE 2 ==============

def compute_feedback_regulation(error: Union[float, np.ndarray], t: float, 
                                An_t: Union[float, np.ndarray], fn_t: Union[float, np.ndarray], 
                                config: Dict) -> Union[float, np.ndarray]:
    """
    Calcule la régulation G pour le feedback.
    
    Args:
        error: erreur Oₙ(t) - Eₙ(t)
        t: temps actuel
        An_t: amplitude(s)
        fn_t: fréquence(s)
        config: configuration
    
    Returns:
        G(error) selon l'archétype configuré
    """
    regulation_config = config.get('regulation', {})
    G_arch = regulation_config.get('G_arch', 'tanh')
    G_params = {
        'lambda': regulation_config.get('lambda', 1.0),
        'alpha': regulation_config.get('alpha', 1.0),
        'beta': regulation_config.get('beta', 2.0)
    }
    #print(f"🌀🌀 REGULATION CALLED")
    # Pour phase 1, on utilise simplement G(x) sans la complexité Gn
    return compute_G(error, G_arch, G_params)


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module regulation.py
    """
    print("=== Tests du module regulation.py ===\n")
    
    # Test 1: Archétypes de régulation
    print("Test 1 - Archétypes G(x):")
    x_test = np.linspace(-3, 3, 7)
    
    for arch in ["tanh", "sinc", "resonance", "spiral_log", "adaptive"]:
        g_vals = compute_G(x_test, arch)
        print(f"  {arch}: G(0) = {compute_G(0, arch):.4f}")
    
    # Test 2: Enveloppes
    print("\nTest 2 - Enveloppes:")
    x_env = np.linspace(-2, 2, 5)
    env_gauss = compute_env_n(x_env, t=0, mode="static", sigma_n=0.5, mu_n=0, env_type="gaussienne")
    env_sig = compute_env_n(x_env, t=0, mode="static", sigma_n=0.5, mu_n=0, env_type="sigmoide")
    print(f"  Gaussienne en 0: {compute_env_n(0, 0, 'static', 0.5, 0, None, 'gaussienne'):.4f}")
    print(f"  Sigmoïde en 0: {compute_env_n(0, 0, 'static', 0.5, 0, None, 'sigmoide'):.4f}")
    
    # Test 3: Sigma dynamique
    print("\nTest 3 - Écart-type dynamique:")
    T = 100
    for t in [0, 25, 50, 75, 100]:
        sigma_dyn = compute_sigma_n(t, "dynamic", T, 0.1, 
                                    {"amp": 0.05, "freq": 1, "offset": 0.1})
        print(f"  σₙ(t={t}) = {sigma_dyn:.4f}")
    
    # Test 4: Régulation intégrée
    print("\nTest 4 - Régulation complète:")
    config_test = {
        'regulation': {'G_arch': 'tanh', 'lambda': 2.0},
        'enveloppe': {'env_mode': 'static', 'sigma_n_static': 0.1},
        'system': {'T': 100}
    }
    
    error_test = np.array([0.5, -0.3, 0.1])
    An_test = np.array([1.0, 0.8, 1.2])
    fn_test = np.array([1.0, 1.1, 0.9])
    
    feedback = compute_Gn(error_test, t=50, An_t=An_test, fn_t=fn_test, config=config_test)
    print(f"  Feedback shape: {feedback.shape}")
    print(f"  Feedback values: {feedback}")
    
    print("\n✅ Module regulation.py prêt pour l'harmonie spiralée")
