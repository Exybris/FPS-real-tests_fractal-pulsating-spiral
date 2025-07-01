"""
dynamics.py - Calculs des termes FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
La définition actuelle de [Sᵢ(t)]/[Eₙ(t)]/[Oₙ(t)] (ainsi que de
φₙ(t), θ(t), η(t), μₙ(t) et les latences) est une hypothèse de phase 1,
appelée à être falsifiée/raffinée selon la feuille de route FPS.
---------------------------------------------------------------

Ce module implémente TOUS les calculs dynamiques du système FPS :
- Input contextuel avec modes multiples
- Calculs adaptatifs (amplitude, fréquence, phase)
- Signaux inter-strates et feedback
- Régulation spiralée
- Métriques globales

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any
import regulation


# ============== FONCTIONS D'INPUT CONTEXTUEL ==============

def compute_In(t: float, perturbation_config: Dict[str, Any], N: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calcule l'input contextuel pour toutes les strates.
    
    Args:
        t: temps actuel
        perturbation_config: configuration de perturbation depuis config.json
        N: nombre de strates (optionnel, pour retourner un array)
    
    Returns:
        float ou np.ndarray: valeur(s) d'input contextuel
    
    Modes supportés:
        - "constant": valeur fixe
        - "choc": impulsion à t0
        - "rampe": augmentation linéaire
        - "sinus": oscillation périodique
        - "uniform": U[0,1] aléatoire
        - "none": pas de perturbation (0.0)
    """
    mode = perturbation_config.get('type', 'none')
    amplitude = perturbation_config.get('amplitude', 1.0)
    t0 = perturbation_config.get('t0', 0.0)
    
    # Calcul de la valeur de base selon le mode
    if mode == "constant":
        value = amplitude
    
    elif mode == "choc":
        # Impulsion brève à t0
        dt = perturbation_config.get('dt', 0.05)  # durée du pic
        if abs(t - t0) < dt:
            value = amplitude
        else:
            value = 0.0
    
    elif mode == "rampe":
        # Augmentation linéaire de 0 à amplitude
        duration = perturbation_config.get('duration', 10.0)
        if t < t0:
            value = 0.0
        elif t < t0 + duration:
            value = amplitude * (t - t0) / duration
        else:
            value = amplitude
    
    elif mode == "sinus":
        # Oscillation périodique
        freq = perturbation_config.get('freq', 0.1)
        if t >= t0:
            value = amplitude * np.sin(2 * np.pi * freq * (t - t0))
        else:
            value = 0.0
    
    elif mode == "uniform":
        # Bruit uniforme U[0,1] * amplitude
        value = amplitude * np.random.uniform(0, 1)
    
    else:  # "none" ou mode inconnu
        value = 0.0
    
    # Retourner un array si N est spécifié
    if N is not None:
        return np.full(N, value)
    return value


# ============== FONCTIONS D'ADAPTATION ==============

def compute_sigma(x: Union[float, np.ndarray], k: float, x0: float) -> Union[float, np.ndarray]:
    """
    Fonction sigmoïde d'adaptation douce.
    
    σ(x) = 1 / (1 + exp(-k(x - x0)))
    
    Args:
        x: valeur(s) d'entrée
        k: sensibilité (pente)
        x0: seuil de basculement
    
    Returns:
        Valeur(s) sigmoïde entre 0 et 1
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def compute_An(t: float, state: List[Dict], In_t: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calcule l'amplitude adaptative pour chaque strate selon FPS Paper.
    
    Aₙ(t) = A₀ · σ(Iₙ(t)) · envₙ(x,t)  [si mode dynamique]
    Aₙ(t) = A₀ · σ(Iₙ(t))              [si mode statique]
    
    où x = Eₙ(t) - Oₙ(t) pour l'enveloppe
    
    Args:
        t: temps actuel
        state: état complet des strates
        In_t: input contextuel pour chaque strate
        config: configuration complète
    
    Returns:
        np.ndarray: amplitudes adaptatives
    """
    N = len(state)
    An_t = np.zeros(N)
    
    # Validation des entrées
    if isinstance(In_t, (int, float)):
        In_t = np.full(N, In_t)  # Convertir scalar en array
    elif len(In_t) != N:
        print(f"⚠️ Taille In_t ({len(In_t)}) != N ({N}), ajustement automatique")
        In_t = np.resize(In_t, N)
    
    # Vérifier le mode enveloppe dynamique
    enveloppe_config = config.get('enveloppe', {})
    env_mode = enveloppe_config.get('env_mode', 'static')
    T = config.get('system', {}).get('T', 100)
    
    # Pour le mode dynamique, on a besoin de En et On
    if env_mode == "dynamic":
        # Calculer En et On pour l'enveloppe
        history = config.get('history', [])
        En_t = compute_En(t, state, history, config)
        
        # Pour On, on a besoin des valeurs actuelles (problème de circularité)
        # Solution : utiliser les valeurs de l'itération précédente
        if len(history) > 0 and 'O' in history[-1]:
            On_t_prev = history[-1]['O']
        else:
            On_t_prev = np.zeros(N)
    
    for n in range(N):
        A0 = state[n]['A0']
        k = state[n]['k']
        x0 = state[n]['x0']
        
        # Amplitude de base via sigmoïde
        base_amplitude = A0 * compute_sigma(In_t[n], k, x0)
        
        if env_mode == "dynamic":
            # Application enveloppe dynamique selon FPS Paper
            try:
                import regulation
                # Paramètres d'enveloppe dynamique
                sigma_n_t = regulation.compute_sigma_n(
                    t, env_mode, T,
                    enveloppe_config.get('sigma_n_static', 0.1),
                    enveloppe_config.get('sigma_n_dynamic')
                )
                mu_n_t = regulation.compute_mu_n(
                    t, env_mode,
                    enveloppe_config.get('mu_n', 0.0),
                    enveloppe_config.get('mu_n_dynamic')
                )
                
                # Utiliser l'erreur Eₙ - Oₙ selon FPS Paper
                error_n = En_t[n] - On_t_prev[n] if n < len(On_t_prev) else 0.0
                env_type = enveloppe_config.get('env_type', 'gaussienne')
                
                # Calculer l'enveloppe avec l'erreur
                env_factor = regulation.compute_env_n(error_n, t, env_mode, 
                                                     sigma_n_t, mu_n_t, T, env_type)
                
                # Amplitude finale avec enveloppe SANS G(error)
                # An = A0 * σ(In) * env(error)
                # G(error) sera appliqué dans S(t) en mode extended
                An_t[n] = base_amplitude * env_factor
                
            except Exception as e:
                print(f"⚠️ Erreur enveloppe dynamique strate {n} à t={t}: {e}")
                An_t[n] = base_amplitude  # Fallback sur mode statique
        else:
            # Mode statique classique
            An_t[n] = base_amplitude
    
    return An_t


# ============== CALCUL DU SIGNAL INTER-STRATES ==============

def compute_S_i(t: float, n: int, history: List[Dict], state: List[Dict]) -> float:
    """
    Calcule le signal provenant des autres strates selon FPS Paper.
    
    S_i(t) = Σ(j≠n) Oj(t) * w_ji
    où w_ji sont les poids de connexion de la strate j vers la strate i.
    
    Args:
        t: temps actuel
        n: indice de la strate courante
        history: historique complet du système
        state: état actuel des strates (pour accéder aux poids)
    
    Returns:
        float: signal pondéré des autres strates
    """
    if t == 0 or len(history) == 0:
        return 0.0
    
    # Récupérer le dernier état avec les sorties observées
    last_state = history[-1]
    On_prev = last_state.get('O', None)
    
    if On_prev is None or not isinstance(On_prev, np.ndarray):
        return 0.0
    
    # Récupérer les poids de la strate n
    if n < len(state) and 'w' in state[n]:
        w_n = state[n]['w']
    else:
        return 0.0
    
    N = len(On_prev)
    S_i = 0.0
    
    # Calculer la somme pondérée selon FPS Paper
    for j in range(N):
        if j != n and j < len(w_n):  # Exclure la strate courante
            # w_n[j] est le poids de j vers n
            S_i += On_prev[j] * w_n[j]
    
    return S_i


# ============== MODULATION DE FRÉQUENCE ==============

def compute_delta_fn(t: float, alpha_n: float, S_i: float) -> float:
    """
    Calcule la modulation de fréquence selon FPS Paper.
    
    Δfₙ(t) = αₙ · S_i(t)
    
    où S_i(t) = Σ(j≠n) w_nj · Oj(t) est déjà calculé
    
    Args:
        t: temps actuel
        alpha_n: souplesse d'adaptation de la strate
        S_i: signal agrégé des autres strates
    
    Returns:
        float: modulation de fréquence
    """
    return alpha_n * S_i


def compute_fn(t: float, state: List[Dict], An_t: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calcule la fréquence modulée pour chaque strate selon FPS Paper.
    
    fₙ(t) = f₀ₙ + Δfₙ(t) · βₙ(t)  [si mode dynamique]
    fₙ(t) = f₀ₙ + Δfₙ(t)          [si mode statique]
    
    Avec contrainte spiralée : fₙ₊₁(t) ≈ r(t) · fₙ(t)
    
    Args:
        t: temps actuel
        state: état des strates
        An_t: amplitudes actuelles
        config: configuration
    
    Returns:
        np.ndarray: fréquences modulées
    """
    N = len(state)
    fn_t = np.zeros(N)
    history = config.get('history', [])
    
    # Vérifier le mode plasticité dynamique
    dynamic_params = config.get('dynamic_parameters', {})
    dynamic_beta = dynamic_params.get('dynamic_beta', False)
    T = config.get('system', {}).get('T', 100)
    
    # Calculer le ratio spiralé r(t) selon FPS_Paper
    if dynamic_params.get('dynamic_phi', False):
        spiral_config = config.get('spiral', {})
        phi = spiral_config.get('phi', 1.618)
        epsilon = spiral_config.get('epsilon', 0.05)
        omega = spiral_config.get('omega', 0.1)
        theta = spiral_config.get('theta', 0.0)
        r_t = compute_r(t, phi, epsilon, omega, theta)
    else:
        r_t = None
    
    # Calculer d'abord toutes les modulations de base
    delta_fn_array = np.zeros(N)
    for n in range(N):
        f0n = state[n]['f0']
        alpha_n = state[n]['alpha']
        beta_n = state[n]['beta']
        
        # Calcul du signal des autres strates
        S_i = compute_S_i(t, n, history, state)
        
        # Modulation de fréquence de base
        delta_fn = compute_delta_fn(t, alpha_n, S_i)
        delta_fn_array[n] = delta_fn
        
        if dynamic_beta:
            # Plasticité βₙ(t) adaptative
            try:
                # Facteur de plasticité basé sur l'amplitude et le temps
                A_factor = An_t[n] / state[n]['A0'] if state[n]['A0'] > 0 else 1.0
                t_factor = 1.0 + 0.5 * np.sin(2 * np.pi * t / T)  # Oscillation temporelle
                
                # Moduler βₙ selon le contexte
                # DÉSACTIVÉ : effort_factor causait des chutes à 0 non désirées
                # effort_factor = 1.0
                # if len(history) > 0:
                #     recent_effort = history[-1].get('effort(t)', 0.0)
                #     # Plus d'effort → moins de plasticité (stabilisation)
                #     effort_factor = 1.0 / (1.0 + 0.1 * recent_effort)
                
                # beta_n_t = beta_n * A_factor * t_factor * effort_factor
                beta_n_t = beta_n * A_factor * t_factor  # Sans effort_factor
                
                # Fréquence de base avec plasticité dynamique
                fn_t[n] = f0n + delta_fn * beta_n_t
                
            except Exception as e:
                print(f"⚠️ Erreur plasticité dynamique strate {n} à t={t}: {e}")
                fn_t[n] = f0n + delta_fn * beta_n  # Fallback sur mode statique
        else:
            # Mode statique classique
            fn_t[n] = f0n + delta_fn * beta_n
    
    # Appliquer la contrainte spiralée si r(t) est défini
    if r_t is not None and N > 1:
        # Ajustement progressif pour respecter fₙ₊₁ ≈ r(t) · fₙ
        # On utilise une approche de relaxation pour éviter les changements brusques
        relaxation_factor = 0.1  # Facteur d'ajustement doux
        
        for n in range(N - 1):
            # Ratio actuel entre fréquences adjacentes
            if fn_t[n] > 0:
                current_ratio = fn_t[n + 1] / fn_t[n]
                # Ajustement vers le ratio cible
                target_fn = r_t * fn_t[n]
                fn_t[n + 1] = fn_t[n + 1] * (1 - relaxation_factor) + target_fn * relaxation_factor
    
    return fn_t


# ============== PHASE ==============

def compute_phi_n(t: float, state: List[Dict], config: Dict) -> np.ndarray:
    """
    Calcule la phase pour chaque strate.
    
    Args:
        t: temps actuel
        state: état des strates
        config: configuration
    
    Returns:
        np.ndarray: phases
    
    Modes:
        - "static": φₙ constant (depuis config)
        - "dynamic": évolution à définir après phase 1
    """
    N = len(state)
    phi_n_t = np.zeros(N)
    
    # Récupération du mode depuis config
    dynamic_params = config.get('dynamic_parameters', {})
    dynamic_phi = dynamic_params.get('dynamic_phi', False)
    
    if dynamic_phi:
        # Mode dynamique selon FPS_Paper Section II : "Modulation spiralée"
        phi_golden = config.get('spiral', {}).get('phi', 1.618)
        epsilon = config.get('spiral', {}).get('epsilon', 0.05)
        omega = config.get('spiral', {}).get('omega', 0.1)
        theta = config.get('spiral', {}).get('theta', 0.0)
        
        # Calcul du ratio spiralé r(t) selon FPS_Paper
        # r(t) = φ + ε · sin(2π·ω·t + θ)
        r_t = phi_golden + epsilon * np.sin(2 * np.pi * omega * t + theta)
        
        for n in range(N):
            # Phase de base de la strate
            phi_base = state[n].get('phi', 0.0)
            
            # Évolution spiralée selon FPS_Paper :
            # Les phases évoluent selon la "règle de dilatation harmonique"
            # φₙ(t) = φₙ(0) + intégration de la modulation spiralée
            
            # Modulation spiralée adaptée par strate
            # Chaque strate suit sa propre évolution mais liée au ratio global r(t)
            spiral_phase_increment = r_t * epsilon * np.sin(2 * np.pi * omega * t + n * 2 * np.pi / N)
            
            # Interaction inter-strates légère (éviter convergence forcée)
            inter_strata_influence = 0.0
            for j in range(N):
                if j != n:
                    w_nj = state[n].get('w', [0.0]*N)[j] if len(state[n].get('w', [])) > j else 0.0
                    # Influence modérée pour permettre émergence sans convergence forcée
                    phase_diff = state[j].get('phi', 0.0) - phi_base
                    inter_strata_influence += 0.1 * w_nj * np.sin(phase_diff)
            
            # Phase résultante selon modulation spiralée FPS
            total_phase = phi_base + spiral_phase_increment + inter_strata_influence
            
            # Garder phases spiralantes SANS modulo pour permettre vraies excursions
            phi_n_t[n] = total_phase  # Phases peuvent dépasser 2π pour vraie spiralisation
    else:
        # Mode statique
        for n in range(N):
            phi_n_t[n] = state[n].get('phi', 0.0)
    
    return phi_n_t


# ============== LATENCE EXPRESSIVE ==============

def compute_gamma(t: float, mode: str = "static", T: Optional[float] = None, 
                  k: Optional[float] = None, t0: Optional[float] = None) -> float:
    """
    Calcule la latence expressive globale.
    
    Args:
        t: temps actuel
        mode: "static", "dynamic", "sigmoid_up", "sigmoid_down", "sigmoid_adaptive", "sigmoid_oscillating", "sinusoidal"
        T: durée totale (pour modes non statiques)
        k: paramètre de pente (optionnel, défaut selon mode) ou fréquence pour sinusoidal
        t0: temps de transition (optionnel, défaut = T/2) ou phase initiale pour sinusoidal
    
    Returns:
        float: latence entre 0 et 1
    
    Formes:
        - static: γ(t) = 1.0
        - dynamic: γ(t) = 1/(1 + exp(-k(t - t0)))
        - sigmoid_up: activation progressive
        - sigmoid_down: désactivation progressive
        - sigmoid_adaptive: varie entre 0.3 et 1.0
        - sigmoid_oscillating: sigmoïde + oscillation sinusoïdale mise à l'échelle
        - sinusoidal: oscillation sinusoïdale pure entre 0.1 et 0.9
    """
    if mode == "static":
        return 1.0
    elif mode == "dynamic" and T is not None:
        # Sigmoïde centrée à t0 (par défaut T/2)
        k_val = k if k is not None else 2.0
        t0_val = t0 if t0 is not None else T / 2
        return 1.0 / (1.0 + np.exp(-k_val * (t - t0_val)))
    elif mode == "sigmoid_up" and T is not None:
        # Activation progressive
        k_val = k if k is not None else 4.0 / T
        t0_val = t0 if t0 is not None else T / 2
        return 1.0 / (1.0 + np.exp(-k_val * (t - t0_val)))
    elif mode == "sigmoid_down" and T is not None:
        # Désactivation progressive
        k_val = k if k is not None else 4.0 / T
        t0_val = t0 if t0 is not None else T / 2
        return 1.0 / (1.0 + np.exp(k_val * (t - t0_val)))
    elif mode == "sigmoid_adaptive" and T is not None:
        # Varie entre 0.3 et 1.0
        k_val = k if k is not None else 4.0 / T
        t0_val = t0 if t0 is not None else T / 2
        return 0.3 + 0.7 / (1.0 + np.exp(-k_val * (t - t0_val)))
    elif mode == "sigmoid_oscillating" and T is not None:
        # Sigmoïde avec oscillation sinusoïdale
        k_val = k if k is not None else 4.0 / T
        t0_val = t0 if t0 is not None else T / 2
        
        # Calcul de la sigmoïde de base (entre 0 et 1)
        base_sigmoid = 1.0 / (1.0 + np.exp(-k_val * (t - t0_val)))
        
        # Oscillation avec fréquence adaptée
        oscillation_freq = 2.0  # Nombre d'oscillations sur la durée T
        oscillation_phase = 2 * np.pi * oscillation_freq / T * t
        
        # Mise à l'échelle pour préserver les oscillations complètes
        # La sigmoïde varie de 0 à 1, on la transforme pour varier de 0.1 à 0.9
        # puis on ajoute une oscillation de ±0.1 autour
        sigmoid_scaled = 0.1 + 0.8 * base_sigmoid
        oscillation_amplitude = 0.1
        
        # Résultat final : sigmoïde mise à l'échelle + oscillation
        # Cela garantit que γ reste dans [0.0, 1.0] sans saturation
        gamma = sigmoid_scaled + oscillation_amplitude * np.sin(oscillation_phase)
        
        # Assurer que gamma reste dans les bornes [0.1, 1.0] par sécurité
        # mais sans écrêtage brutal
        return max(0.1, min(1.0, gamma))
    elif mode == "sinusoidal" and T is not None:
        # Oscillation sinusoïdale pure sans transition sigmoïde
        # k représente le nombre d'oscillations sur la durée T (défaut: 2)
        # t0 représente la phase initiale en radians (défaut: 0)
        freq = k if k is not None else 2.0  # Nombre d'oscillations sur T
        phase_init = t0 if t0 is not None else 0.0  # Phase initiale
        
        # Oscillation entre 0.1 et 0.9 pour rester dans une plage utile
        # γ(t) = 0.5 + 0.4 * sin(2π * freq * t/T + phase_init)
        oscillation = np.sin(2 * np.pi * freq * t / T + phase_init)
        gamma = 0.5 + 0.4 * oscillation
        
        # Assurer que gamma reste dans [0.1, 0.9]
        return max(0.1, min(0.9, gamma))
    else:
        return 1.0


def compute_gamma_n(t: float, state: List[Dict], config: Dict) -> np.ndarray:
    """
    Calcule la latence expressive par strate.
    
    Args:
        t: temps actuel
        state: état des strates
        config: configuration
    
    Returns:
        np.ndarray: latences par strate
    """
    N = len(state)
    gamma_n_t = np.zeros(N)
    
    # Configuration de latence
    latence_config = config.get('latence', {})
    # IMPORTANT: Utiliser gamma_mode (pas gamma_n_mode) pour cohérence
    gamma_mode = latence_config.get('gamma_mode', 'static')
    T = config.get('system', {}).get('T', 100)
    
    # Récupérer les paramètres k et t0 depuis gamma_dynamic
    gamma_dynamic = latence_config.get('gamma_dynamic', {})
    k = gamma_dynamic.get('k', None)
    t0 = gamma_dynamic.get('t0', None)
    
    if gamma_mode in ["static", "dynamic", "sigmoid_up", "sigmoid_down", "sigmoid_adaptive", "sigmoid_oscillating", "sinusoidal"]:
        # Utiliser compute_gamma pour tous les modes avec k et t0
        gamma_global = compute_gamma(t, gamma_mode, T, k, t0)
        
        # Option 1: Toutes les strates utilisent le même gamma
        gamma_n_t[:] = gamma_global
        
        # Option 2: Décalage progressif entre strates (si activé dans config)
        if latence_config.get('strata_delay', False):
            for n in range(N):
                t_shifted = t - n * T / (2 * N)  # Décalage temporel progressif
                gamma_n_t[n] = compute_gamma(t_shifted, gamma_mode, T, k, t0)
        else:
            gamma_n_t[:] = gamma_global
    else:
        # Mode non reconnu, utiliser static par défaut
        gamma_n_t[:] = 1.0
    
    return gamma_n_t


# ============== SORTIES OBSERVÉE ET ATTENDUE ==============

def compute_On(t: float, state: List[Dict], An_t: np.ndarray, fn_t: np.ndarray, 
               phi_n_t: np.ndarray, gamma_n_t: np.ndarray) -> np.ndarray:
    """
    Calcule la sortie observée pour chaque strate.
    
    Oₙ(t) = Aₙ(t) · sin(2π·fₙ(t)·t + φₙ(t)) · γₙ(t)
    
    Args:
        t: temps actuel
        state: état des strates
        An_t: amplitudes
        fn_t: fréquences
        phi_n_t: phases
        gamma_n_t: latences
    
    Returns:
        np.ndarray: sorties observées

    Cette formule exploratoire est temporaire
    """
    N = len(state)
    On_t = np.zeros(N)
    
    for n in range(N):
        # Contribution de la strate n au signal global
        On_t[n] = An_t[n] * np.sin(2 * np.pi * fn_t[n] * t + phi_n_t[n]) * gamma_n_t[n]
    
    return On_t


def compute_En(t: float, state: List[Dict], history: List[Dict], config: Dict, 
               history_align: List[float] = None) -> np.ndarray:
    """
    Calcule la sortie attendue (harmonique cible) pour chaque strate.
    
    NOUVEAU S1: Attracteur inertiel avec lambda_E adaptatif
    Eₙ(t) = (1-λ) * Eₙ(t-dt) + λ * φ * Oₙ(t-τ)
    
    où λ peut être modulé par k_spacing selon le nombre d'alignements
    
    Args:
        t: temps actuel
        state: état des strates
        history: historique
        config: configuration
        history_align: historique des alignements En≈On (nouveau S1)
    
    Returns:
        np.ndarray: sorties attendues
    """
    N = len(state)
    En_t = np.zeros(N)
    
    # Paramètres de l'attracteur inertiel
    lambda_E = config.get('regulation', {}).get('lambda_E', 0.05)
    k_spacing = config.get('regulation', {}).get('k_spacing', 0.0)
    phi = config.get('spiral', {}).get('phi', 1.618)
    dt = config.get('system', {}).get('dt', 0.1)
    
    # Adapter lambda selon le nombre d'alignements (spacing effect)
    if history_align is not None and k_spacing > 0:
        n_alignments = len(history_align)
        lambda_dyn = lambda_E / (1 + k_spacing * n_alignments)
    else:
        lambda_dyn = lambda_E
    
    if len(history) > 0:
        # Récupérer les valeurs précédentes
        last_En = history[-1].get('E', np.zeros(N))
        last_On = history[-1].get('O', np.zeros(N))
        
        # S'assurer que les arrays ont la bonne taille
        if not isinstance(last_En, np.ndarray) or len(last_En) != N:
            last_En = np.zeros(N)
            for n in range(N):
                last_En[n] = state[n]['A0']
        
        if not isinstance(last_On, np.ndarray) or len(last_On) != N:
            last_On = np.zeros(N)
        
        # Attracteur inertiel : Eₙ(t) = (1-λ)*Eₙ(t-dt) + λ*φ*Oₙ(t-τ)
        # τ = dt pour l'instant (peut être ajusté)
        for n in range(N):
            En_t[n] = (1 - lambda_dyn) * last_En[n] + lambda_dyn * phi * last_On[n]
    else:
        # Valeur initiale = amplitude de base
        for n in range(N):
            En_t[n] = state[n]['A0']
    
    return En_t


# ============== SPIRALISATION ==============

def compute_r(t: float, phi: float, epsilon: float, omega: float, theta: float) -> float:
    """
    Calcule le ratio spiralé.
    
    r(t) = φ + ε · sin(2π·ω·t + θ)
    
    Args:
        t: temps actuel
        phi: nombre d'or
        epsilon: amplitude de variation
        omega: fréquence de modulation
        theta: phase initiale
    
    Returns:
        float: ratio spiralé
    """
    return phi + epsilon * np.sin(2 * np.pi * omega * t + theta)


def compute_C(t: float, phi_n_array: np.ndarray) -> float:
    """
    Calcule le coefficient d'accord spiralé selon FPS_Paper.
    
    C(t) = (1/N) · Σ cos(φₙ₊₁ - φₙ)
    
    Pour phases spiralantes (sans modulo), normalise les différences
    pour mesurer la cohérence locale plutôt que l'alignement absolu.
    
    Args:
        t: temps actuel
        phi_n_array: phases de toutes les strates (peuvent dépasser 2π)
    
    Returns:
        float: coefficient d'accord entre -1 et 1
    """
    N = len(phi_n_array)
    if N <= 1:
        return 1.0
    
    # Somme des cosinus entre phases adjacentes  
    cos_sum = 0.0
    for n in range(N - 1):
        # Différence de phase brute (peut être > 2π)
        phase_diff = phi_n_array[n + 1] - phi_n_array[n]
        
        # Pour phases spiralantes : mesurer cohérence locale
        # en normalisant la différence modulo 2π
        normalized_diff = ((phase_diff + np.pi) % (2 * np.pi)) - np.pi
        
        cos_sum += np.cos(normalized_diff)
    
    return cos_sum / (N - 1)


def compute_A(t: float, delta_fn_array: np.ndarray) -> float:
    """
    Calcule la modulation moyenne selon FPS Paper.
    
    A(t) = (1/N) · Σ Δfₙ(t)
    
    Args:
        t: temps actuel
        delta_fn_array: modulations de fréquence
    
    Returns:
        float: modulation moyenne
    """
    if len(delta_fn_array) == 0:
        return 0.0
    return np.mean(delta_fn_array)


def compute_A_spiral(t: float, C_t: float, A_t: float) -> float:
    """
    Calcule l'amplitude harmonisée.
    
    A_spiral(t) = C(t) · A(t)
    
    Args:
        t: temps actuel
        C_t: coefficient d'accord
        A_t: modulation moyenne
    
    Returns:
        float: amplitude spiralée
    """
    return C_t * A_t


# ============== FEEDBACK ==============

def compute_Fn(t: float, beta_n: float, On_t: float, En_t: float, gamma_t: float, 
               An_t: float, fn_t: float, config: dict) -> float:
    """
    Calcule le feedback pour une strate.
    
    Fₙ(t) = βₙ · G(Oₙ(t) - Eₙ(t)) · γ(t)
    où G peut être :
    - Identité (pas de régulation)
    - Archétype simple (tanh, sinc, resonance, adaptive)
    - Gn complet avec sinc et enveloppe
    
    Args:
        t: temps actuel
        beta_n: plasticité de la strate
        On_t: sortie observée
        En_t: sortie attendue
        gamma_t: latence globale
        An_t: amplitude actuelle
        fn_t: fréquence actuelle
        config: configuration
    
    Returns:
        float: valeur de feedback
    """
    error = On_t - En_t
    
    # Récupérer le mode de feedback depuis config
    feedback_mode = config.get('regulation', {}).get('feedback_mode', 'simple')
    
    if feedback_mode == 'simple':
        # Formule de base sans régulation G
        return beta_n * error * gamma_t
    
    elif feedback_mode == 'archetype':
        # Utiliser un archétype G simple
        G_arch = config.get('regulation', {}).get('G_arch', 'tanh')
        G_params = {
            'lambda': config.get('regulation', {}).get('lambda', 1.0),
            'alpha': config.get('regulation', {}).get('alpha', 1.0),
            'beta': config.get('regulation', {}).get('beta', 2.0)
        }
        G_feedback = regulation.compute_G(error, G_arch, G_params)
        return beta_n * G_feedback * gamma_t
    
    elif feedback_mode == 'gn_full':
        # Utiliser Gn complet avec sinc et enveloppe
        env_config = config.get('enveloppe', {})
        T = config['system']['T']
        
        # Centre et largeur de l'enveloppe
        mu_n_t = regulation.compute_mu_n(t, env_config.get('env_mode', 'static'), 
                                        env_config.get('mu_n', 0.0))
        sigma_n_t = regulation.compute_sigma_n(t, env_config.get('env_mode', 'static'), T,
                                              env_config.get('sigma_n_static', 0.1))
        
        # Type d'enveloppe (gaussienne ou sigmoïde)
        env_type = env_config.get('env_type', 'gaussienne')
        
        # Calcul de l'enveloppe
        env_n = regulation.compute_env_n(error, t, env_config.get('env_mode', 'static'),
                                        sigma_n_t, mu_n_t, T, env_type)
        
        # Régulation complète avec Gn
        G_feedback = regulation.compute_Gn(error, t, An_t, fn_t, mu_n_t, env_n)
        return beta_n * G_feedback * gamma_t
    
    else:
        # Mode non reconnu, fallback sur simple
        print(f"⚠️ Mode de feedback '{feedback_mode}' non reconnu, utilisation du mode simple")
        return beta_n * error * gamma_t


# ============== SIGNAL GLOBAL ==============

def compute_S(t: float, An_array: np.ndarray, fn_array: np.ndarray, 
              phi_n_array: np.ndarray, config: Dict) -> float:
    """
    Calcule le signal global du système selon FPS Paper.
    
    Args:
        t: temps actuel
        An_array: amplitudes
        fn_array: fréquences
        phi_n_array: phases
        config: configuration (pour modes avancés)
    
    Returns:
        float: signal global S(t)
    
    Modes:
        - "simple": Σₙ Aₙ(t)·sin(2π·fₙ(t)·t + φₙ(t))
        - "extended": S(t) = Σₙ [Aₙ(t)·sin(2π·fₙ(t)·t + φₙ(t))·γₙ(t)]·G(Eₙ(t) - Oₙ(t))
    """
    mode = config.get('system', {}).get('signal_mode', 'simple')
    N = len(An_array)
    
    if mode == "simple":
        # Somme simple des contributions
        S_t = 0.0
        for n in range(N):
            S_t += An_array[n] * np.sin(2 * np.pi * fn_array[n] * t + phi_n_array[n])
        return S_t
    
    elif mode == "extended":
        # Version complète selon FPS Paper Chapitre 4
        # S(t) = Σₙ [Aₙ(t)·sin(2π·fₙ(t)·t + φₙ(t))·γₙ(t)]·G(Eₙ(t) - Oₙ(t))
        
        state = config.get('state', [])
        history = config.get('history', [])
        
        # Vérifier que state est valide
        if not state or len(state) != N:
            # Fallback sur mode simple si pas d'état complet
            return compute_S(t, An_array, fn_array, phi_n_array, {'system': {'signal_mode': 'simple'}})
        
        # Calculer les composants nécessaires
        gamma_n_t = compute_gamma_n(t, state, config)
        En_t = compute_En(t, state, history, config)
        On_t = compute_On(t, state, An_array, fn_array, phi_n_array, gamma_n_t)
        
        S_t = 0.0
        for n in range(N):
            # Contribution de base avec latence selon FPS Paper
            sin_component = np.sin(2 * np.pi * fn_array[n] * t + phi_n_array[n])
            base_contribution = An_array[n] * sin_component * gamma_n_t[n]
            
            # Calcul de G(Eₙ - Oₙ) selon FPS Paper
            error = En_t[n] - On_t[n]
            
            # Paramètres pour la fonction G
            G_arch = config.get('regulation', {}).get('G_arch', 'tanh')
            G_params = {
                'lambda': config.get('regulation', {}).get('lambda', 1.0),
                'alpha': config.get('regulation', {}).get('alpha', 1.0),
                'beta': config.get('regulation', {}).get('beta', 2.0)
            }
            
            # Calculer G(error)
            import regulation
            G_value = regulation.compute_G(error, G_arch, G_params)
            
            # Contribution finale selon FPS Paper : chaque terme est multiplié par G
            S_t += base_contribution * G_value
        
        return S_t
    
    else:
        # Par défaut, mode simple
        return compute_S(t, An_array, fn_array, phi_n_array, {'system': {'signal_mode': 'simple'}})


# ============== MÉTRIQUES GLOBALES ==============

def compute_E(t: float, signal_array: Union[np.ndarray, List[float]]) -> float:
    """
    Calcule l'énergie totale du système.
    
    E(t) = sqrt(Σₙ Aₙ²(t)) / sqrt(N)
    
    Représente l'énergie totale distribuée dans le système,
    utilisée pour évaluer la capacité du système à maintenir
    une activité cohérente.
    
    Args:
        t: temps actuel
        signal_array: amplitudes Aₙ(t) de chaque strate
    
    Returns:
        float: énergie totale normalisée
    """
    if len(signal_array) == 0:
        return 0.0
    
    # Convertir en array numpy si nécessaire
    amplitudes = np.asarray(signal_array)
    
    # Énergie comme norme L2 des amplitudes
    energy = np.sqrt(np.sum(np.square(amplitudes)))
    
    # Normaliser par sqrt(N) pour avoir une mesure comparable
    # indépendamment du nombre de strates
    N = len(amplitudes)
    if N > 0:
        energy = energy / np.sqrt(N)
    
    return energy


def compute_L(t: float, An_history: List[np.ndarray], dt: float = 0.1) -> int:
    """
    Calcule L(t) selon FPS_Paper.md : argmaxₙ |dAₙ(t)/dt|
    
    Retourne l'indice de la strate avec la variation d'amplitude maximale.
    "Latence maximale de variation d'une strate" - quelle strate change le plus vite.
    
    Args:
        t: temps actuel (pour compatibilité, pas utilisé dans cette version)
        An_history: historique des amplitudes [An(t-dt), An(t), ...]
        dt: pas de temps pour calcul dérivée
    
    Returns:
        int: indice de la strate avec |dAₙ/dt| maximal
    """
    if len(An_history) < 2:
        # Pas assez d'historique pour dérivée
        return 0
    
    # Derniers états pour calcul dérivée
    An_current = np.asarray(An_history[-1])  # An(t)
    An_previous = np.asarray(An_history[-2])  # An(t-dt)
    
    if len(An_current) == 0 or len(An_previous) == 0:
        return 0
    
    # Calcul dérivées : dAₙ/dt ≈ (An(t) - An(t-dt)) / dt
    derivatives = np.abs((An_current - An_previous) / dt)
    
    # Retourner indice de variation maximale
    return int(np.argmax(derivatives))


def compute_L_legacy(t: float, signal_array: Union[np.ndarray, List[float]]) -> int:
    """
    Version legacy de compute_L (pour compatibilité si besoin).
    
    Args:
        t: temps actuel
        signal_array: signaux ou amplitudes
    
    Returns:
        int: indice de la strate avec amplitude max
    """
    if len(signal_array) == 0:
        return 0
    
    # Indice de la strate dominante
    return int(np.argmax(np.abs(signal_array)))


# ============== FONCTIONS UTILITAIRES ==============

def update_state(state: List[Dict], An_t: np.ndarray, fn_t: np.ndarray, 
                 phi_n_t: np.ndarray, gamma_n_t: np.ndarray, F_n_t: np.ndarray) -> List[Dict]:
    """
    Met à jour l'état complet du système.
    
    Args:
        state: état actuel
        An_t: amplitudes calculées
        fn_t: fréquences calculées
        phi_n_t: phases calculées
        gamma_n_t: latences calculées
        F_n_t: feedbacks calculés
    
    Returns:
        État mis à jour
    """
    N = len(state)
    
    for n in range(N):
        # Mise à jour des valeurs courantes
        state[n]['current_An'] = An_t[n] if n < len(An_t) else state[n].get('A0', 1.0)
        state[n]['current_fn'] = fn_t[n] if n < len(fn_t) else state[n].get('f0', 1.0)
        state[n]['current_phi'] = phi_n_t[n] if n < len(phi_n_t) else state[n].get('phi', 0.0)
        state[n]['current_gamma'] = gamma_n_t[n] if n < len(gamma_n_t) else 1.0
        state[n]['current_Fn'] = F_n_t[n] if n < len(F_n_t) else 0.0
        
        # NOUVEAU : Mise à jour des valeurs de base pour la prochaine itération
        # Ceci permet l'évolution temporelle du système
        # if 'current_An' in state[n] and state[n]['current_An'] != 0:
            # Evolution progressive de A0 vers la valeur courante
            # Taux réduit pour éviter l'extinction du signal
            # adaptation_rate = 0.01  # Réduit de 0.1 à 0.01
            # Conserver une amplitude minimale pour éviter l'extinction
            # min_amplitude = 0.1
            # new_A0 = state[n]['A0'] * (1 - adaptation_rate) + state[n]['current_An'] * adaptation_rate
            # state[n]['A0'] = max(min_amplitude, new_A0)
        
        # if 'current_fn' in state[n]:
            # Evolution progressive de f0 vers la valeur courante
            # adaptation_rate = 0.005  # Réduit de 0.05 à 0.005
            # state[n]['f0'] = state[n]['f0'] * (1 - adaptation_rate) + state[n]['current_fn'] * adaptation_rate
        
        # NOUVEAU : Mise à jour des phases si mode dynamique
        # if 'current_phi' in state[n]:
            # state[n]['phi'] = state[n]['current_phi']
    
    return state


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests basiques pour valider les fonctions.
    """
    print("=== Tests du module dynamics.py ===\n")
    
    # Test 1: Fonction sigmoïde
    print("Test 1 - Sigmoïde:")
    x_test = np.linspace(-5, 5, 11)
    sigma_test = compute_sigma(x_test, k=2.0, x0=0.0)
    print(f"  σ(0) = {compute_sigma(0, 2.0, 0.0):.4f} (attendu: 0.5)")
    print(f"  σ(-∞) → {compute_sigma(-10, 2.0, 0.0):.4f} (attendu: ~0)")
    print(f"  σ(+∞) → {compute_sigma(10, 2.0, 0.0):.4f} (attendu: ~1)")
    
    # Test 2: Input contextuel
    print("\nTest 2 - Input contextuel:")
    pert_config = {'type': 'choc', 't0': 5.0, 'amplitude': 2.0}
    print(f"  Choc à t=5: {compute_In(5.0, pert_config)}")
    print(f"  Choc à t=6: {compute_In(6.0, pert_config)}")
    
    # Test 3: Latence
    print("\nTest 3 - Latence:")
    print(f"  γ(t) statique = {compute_gamma(50, mode='static')}")
    print(f"  γ(t=50) dynamique = {compute_gamma(50, mode='dynamic', T=100):.4f}")
    print(f"  γ(t=0) dynamique = {compute_gamma(0, mode='dynamic', T=100):.4f}")
    
    # Test 4: Ratio spiralé
    print("\nTest 4 - Ratio spiralé:")
    r_test = compute_r(0, phi=1.618, epsilon=0.05, omega=0.1, theta=0)
    print(f"  r(0) = {r_test:.4f}")
    
    print("\n✅ Module dynamics.py prêt à l'emploi!")
