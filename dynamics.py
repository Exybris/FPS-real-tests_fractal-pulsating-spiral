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
from scipy import signal
from typing import Dict, List, Union, Optional, Any, Tuple
from collections import defaultdict
import regulation
from regulation import compute_G
import metrics
import warnings
import time
import sys
import os


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


def compute_An(t: float, state: List[Dict], In_t: np.ndarray, F_n_t_An: np.ndarray, config: Dict) -> np.ndarray:
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
                An_t[n] = An_t[n] * (1 + F_n_t_An[n])

                
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


def compute_fn(t: float, state: List[Dict], An_t: np.ndarray, F_n_t_fn: np.ndarray, config: Dict) -> np.ndarray:
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

                fn_t[n] = f0n + delta_fn * beta_n_t + F_n_t_fn[n]
                
            except Exception as e:
                print(f"⚠️ Erreur plasticité dynamique strate {n} à t={t}: {e}")
                fn_t[n] = f0n + delta_fn * beta_n + F_n_t_fn[n] # Fallback sur mode statique
        else:
            # Mode statique classique
            fn_t[n] = f0n + delta_fn * beta_n + F_n_t_fn[n]
    
    # Appliquer la contrainte spiralée si r(t) est défini
    if r_t is not None and N > 1:
        # Ajustement progressif pour respecter fₙ₊₁ ≈ r(t) · fₙ
        # On utilise une approche de relaxation pour éviter les changements brusques
        relaxation_factor = 0.5  # Facteur d'ajustement doux
        
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
        # Mode dynamique avec SIGNATURES INDIVIDUELLES selon Andréa
        phi_golden = config.get('spiral', {}).get('phi', 1.618)
        epsilon = config.get('spiral', {}).get('epsilon', 0.05)
        omega = config.get('spiral', {}).get('omega', 0.1)
        theta = config.get('spiral', {}).get('theta', 0.0)
        
        # Calculer le ratio spiralé r(t) selon FPS_Paper
        r_t = phi_golden + epsilon * np.sin(2 * np.pi * omega * t + theta)
        
        # Mode signatures : chaque strate a sa "voix propre"
        signature_mode = config.get('spiral', {}).get('signature_mode', 'individual')
        
        for n in range(N):
            # EMPREINTE UNIQUE de la strate (signature invariante)
            phi_signature = state[n].get('phi', 0.0)  # Son "ADN phasique"
            
            if signature_mode == 'individual':
                # NOUVEAU : Chaque strate danse autour de SA signature propre
                # ω personnalisée basée sur sa position dans le pentagone
                omega_n = omega * (1.0 + 0.2 * np.sin(n * 2 * np.pi / N))  # Fréquence propre
                # Modulation spiralée AUTOUR de sa signature
                personal_spiral = epsilon * np.sin(2 * np.pi * omega_n * t + phi_signature)
                # Interaction douce avec le ratio global r(t)
                global_influence = 0.3 * (r_t - phi_golden) * np.cos(phi_signature)
                # Interaction inter-strates basée sur affinités phasiques
                inter_strata_influence = 0.0
                for j in range(N):
                    if j != n:
                        w_nj = state[n].get('w', [0.0]*N)[j] if len(state[n].get('w', [])) > j else 0.0
                        phi_j_signature = state[j].get('phi', 0.0)
                        # Affinité basée sur proximité des signatures
                        signature_affinity = np.cos(phi_signature - phi_j_signature)
                        inter_strata_influence += 0.05 * w_nj * signature_affinity * np.sin(2 * np.pi * omega * t)
                # Phase finale : SIGNATURE + danse personnelle + influences
                phi_n_t[n] = phi_signature + personal_spiral + global_influence + inter_strata_influence
            else:
                # Mode original (fallback)
                spiral_phase_increment = r_t * epsilon * np.sin(2 * np.pi * omega * t + n * 2 * np.pi / N)
                inter_strata_influence = 0.0
                for j in range(N):
                    if j != n:
                        w_nj = state[n].get('w', [0.0]*N)[j] if len(state[n].get('w', [])) > j else 0.0
                        phase_diff = state[j].get('phi', 0.0) - phi_signature
                        inter_strata_influence += 0.1 * w_nj * np.sin(phase_diff)
                phi_n_t[n] = phi_signature + spiral_phase_increment + inter_strata_influence
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


def compute_gamma_n(t: float, state: List[Dict], config: Dict, gamma_global: Optional[float] = None,
                    En_array: Optional[np.ndarray] = None, On_array: Optional[np.ndarray] = None,
                    An_array: Optional[np.ndarray] = None, fn_array: Optional[np.ndarray] = None,
                    history: Optional[List[Dict]] = None) -> np.ndarray:
    """
    Calcule la latence expressive par strate.
    
    NOUVELLE VERSION : Modulation locale basée sur l'état dynamique de chaque strate.
    gamma_n = gamma_global * f(erreur_n, amplitude_n, fréquence_n)
    
    Args:
        t: temps actuel
        state: état des strates
        config: configuration
        gamma_global: gamma global pré-calculé (optionnel, pour modes adaptatifs)
        En_array: attentes par strate (optionnel)
        On_array: observations par strate (optionnel)
        An_array: amplitudes par strate (optionnel)
        fn_array: fréquences par strate (optionnel)
        history: historique de simulation pour récupérer les valeurs précédentes (optionnel)
    
    Returns:
        np.ndarray: latences par strate modulées localement
    """
    N = len(state)
    gamma_n_t = np.zeros(N)
    
    # Configuration de latence
    latence_config = config.get('latence', {})
    gamma_mode = latence_config.get('gamma_mode', 'static')
    T = config.get('system', {}).get('T', 100)
    
    # Si gamma_global n'est pas fourni, le calculer
    if gamma_global is None:
        gamma_dynamic = latence_config.get('gamma_dynamic', {})
        k = gamma_dynamic.get('k', None)
        t0 = gamma_dynamic.get('t0', None)
        gamma_global = compute_gamma(t, gamma_mode, T, k, t0)
    
    # Paramètres de modulation (peuvent être dans config)
    modulation_config = latence_config.get('modulation', {})
    k_error = modulation_config.get('k_error', 0.1)      # Poids de l'erreur
    k_amplitude = modulation_config.get('k_amplitude', 0.1)  # Poids de l'amplitude  
    k_frequency = modulation_config.get('k_frequency', 0.05) # Poids de la fréquence
    gamma_min = modulation_config.get('gamma_min', 0.5)  # Borne inf : gamma_global * 0.5
    gamma_max = modulation_config.get('gamma_max', 1.5)  # Borne sup : gamma_global * 1.5
    
    # Essayer de récupérer les données depuis l'historique si non fournies
    if history and len(history) > 0:
        last_step = history[-1]
        if En_array is None and 'E' in last_step:
            En_array = last_step['E'] if isinstance(last_step['E'], np.ndarray) else None
        if On_array is None and 'O' in last_step:
            On_array = last_step['O'] if isinstance(last_step['O'], np.ndarray) else None
        if An_array is None and 'An' in last_step:
            An_array = last_step['An'] if isinstance(last_step['An'], np.ndarray) else None
        if fn_array is None and 'fn' in last_step:
            fn_array = last_step['fn'] if isinstance(last_step['fn'], np.ndarray) else None
    
    # Mode legacy si pas de modulation ou données manquantes
    if not modulation_config.get('enabled', True) or any(x is None for x in [En_array, On_array, An_array, fn_array]):
        # Comportement legacy avec décalage temporel optionnel
        if latence_config.get('strata_delay', False) and gamma_global is None:
            for n in range(N):
                t_shifted = t - n * T / (2 * N)
                gamma_n_t[n] = compute_gamma(t_shifted, gamma_mode, T, 
                                           latence_config.get('gamma_dynamic', {}).get('k', None),
                                           latence_config.get('gamma_dynamic', {}).get('t0', None))
        else:
            gamma_n_t[:] = gamma_global
        return gamma_n_t
    
    # NOUVELLE MODULATION : gamma_n = gamma_global * facteur_modulation_n
    if modulation_config.get('verbose', False) and t < 1.0:  # Log seulement au début
        print(f"[MODULATION] t={t:.2f}: Modulation locale activée (k_err={k_error}, k_amp={k_amplitude}, k_freq={k_frequency})")
    
    for n in range(N):
        # 1. Erreur normalisée (positive = observation > attente)
        error_n = On_array[n] - En_array[n]
        # Normaliser par l'amplitude moyenne pour éviter explosion
        A_mean = np.mean(np.abs(An_array)) if np.mean(np.abs(An_array)) > 0 else 1.0
        error_norm = np.tanh(error_n / A_mean)  # Entre -1 et 1
        
        # 2. Amplitude normalisée (activité de la strate)
        amplitude_norm = An_array[n] / A_mean if A_mean > 0 else 1.0
        amplitude_factor = 1.0 + k_amplitude * (amplitude_norm - 1.0)
        
        # 3. Fréquence normalisée (rapidité du rythme local)
        f_mean = np.mean(fn_array) if np.mean(fn_array) > 0 else 1.0
        freq_norm = fn_array[n] / f_mean
        freq_factor = 1.0 + k_frequency * (freq_norm - 1.0)
        
        # 4. Facteur d'erreur : erreur positive → gamma plus court (réaction plus rapide)
        #                       erreur négative → gamma plus long (attente prudente)
        error_factor = 1.0 - k_error * error_norm
        
        # 5. Combiner les facteurs multiplicativement
        modulation_factor = error_factor * amplitude_factor * freq_factor
        
        # 6. Appliquer à gamma_global avec protection des bornes
        gamma_n_t[n] = gamma_global * modulation_factor
        
        # 7. Bornes adaptatives : rester dans [gamma_min*gamma_global, gamma_max*gamma_global]
        gamma_n_t[n] = np.clip(gamma_n_t[n], gamma_min * gamma_global, gamma_max * gamma_global)
        
        # 8. Bornes absolues de sécurité
        gamma_n_t[n] = np.clip(gamma_n_t[n], 0.1, 1.0)
    
    # Log de vérification de la modulation (seulement si verbose et au début)
    if modulation_config.get('verbose', False) and t < 1.0:
        gamma_range = np.ptp(gamma_n_t)  # peak-to-peak (max - min)
        if gamma_range > 0.01:
            print(f"[MODULATION] γ_n varie de {gamma_n_t.min():.3f} à {gamma_n_t.max():.3f} (écart={gamma_range:.3f})")
    
    return gamma_n_t


# ============== FONCTIONS ADAPTATIVES GAMMA-G ==============

def create_quantum_gamma(t: float, synergies: Dict) -> float:
    """
    Crée une superposition quantique des meilleures synergies.
    """
    if not synergies:
        return 0.5 + 0.4 * np.sin(0.05 * t)
    
    # Top 3 synergies
    top_synergies = sorted(synergies.items(), 
                          key=lambda x: x[1]['score'], 
                          reverse=True)[:3]
    
    gamma = 0
    total_weight = 0
    
    for i, ((g_val, _), info) in enumerate(top_synergies):
        # Poids quantique avec interférences
        weight = info['score'] ** 2
        phase = i * 2 * np.pi / 3 + 0.1 * t
        
        # Oscillation avec battements
        beat_freq = 0.01 * (i + 1)
        amplitude = 1 + 0.1 * np.sin(beat_freq * t) * np.cos(phase)
        
        gamma += weight * g_val * amplitude
        total_weight += weight
    
    return gamma / total_weight if total_weight > 0 else 0.5


def compute_gamma_adaptive_aware(t: float, state: List[Dict], history: List[Dict], 
                                config: Dict, discovery_journal: Dict = None) -> Tuple[float, str, Dict]:
    """
    Latence adaptative complète ET consciente de G(x).
    
    Combine :
    - Surveillance multi-critères (6 métriques)
    - Détection des patterns d'emergence
    - Conscience de l'archétype G actuel
    - Communication bidirectionnelle avec G(x)
    - Journal enrichi des découvertes couplées
    """
    
    # Initialiser le journal super-enrichi
    if discovery_journal is None:
        journal = {
            # Structure complète du journal
            'discovered_regimes': {},
            'transitions': [],
            'current_regime': 'exploration',
            'regime_start_time': 0,
            'total_discoveries': 0,
            'breakthrough_moments': [],
            'score_history': [],
            'gamma_peaks': [],
            'system_performance': [],  # AJOUT de system_performance
            'rest_phases': [],
            'optimal_gamma_patterns': {},
            # NOUVEAU : Conscience de G
            'coupled_states': {},           # (γ, G_arch) → performances
            'G_transition_impacts': [],     # Impacts des changements
            'gamma_G_synergies': {},        # Synergies découvertes
            'communication_signals': [],    # Signaux subtils γ↔G
            'exploration_log': []          # Log d'exploration
        }
    else:
        journal = discovery_journal.copy()
    
    # Phase initiale
    if len(history) < 50:
        # Exploration systématique de l'espace gamma
        exploration_step = int(t / config['system']['dt'])
    
        # Balayer toutes les valeurs de gamma progressivement
        gamma_space = np.linspace(0.1, 1.0, 10)
        gamma_index = exploration_step % len(gamma_space)
        base_gamma = gamma_space[gamma_index]
    
        # Petite variation aléatoire pour explorer autour
        gamma = base_gamma + 0.05 * np.random.randn()
        gamma = np.clip(gamma, 0.1, 1.0)
    
        # Enregistrer ce qu'on explore
        journal['exploration_log'].append({'t': t, 'gamma': gamma, 'phase': 'systematic'})
    
        return gamma, 'exploration', journal
    
    # 1. OBSERVER L'ÉTAT ACTUEL DE G
    current_G_arch = history[-1].get('G_arch_used', 'tanh') if history else 'tanh'
    gamma_current = history[-1].get('gamma', 1.0) if history else 1.0
    
    # 2. CALCULER LA PERFORMANCE SYSTÈME
    recent_history = history[-50:]
    scores = metrics.calculate_all_scores(recent_history)
    current_scores = scores['current']
    system_performance_score = np.mean(list(current_scores.values()))
    
    # 3. ENREGISTRER L'ÉTAT COUPLÉ (γ, G)
    state_key = (round(gamma_current, 1), current_G_arch)
    
    if state_key not in journal['coupled_states']:
        journal['coupled_states'][state_key] = {
            'performances': [],
            'first_seen': t,
            'synergy_score': 0
        }
    
    journal['coupled_states'][state_key]['performances'].append(system_performance_score)
    
    # Calculer le score de synergie
    if len(journal['coupled_states'][state_key]['performances']) >= 5:
        perfs = journal['coupled_states'][state_key]['performances'][-10:]
        mean_perf = np.mean(perfs)
        stability = 1 / (1 + np.std(perfs))
        growth = np.polyfit(range(len(perfs)), perfs, 1)[0] if len(perfs) > 1 else 0
        
        synergy_score = mean_perf * stability * (1 + growth)
        journal['coupled_states'][state_key]['synergy_score'] = synergy_score
        
        # Découverte de synergie exceptionnelle ?
        if synergy_score > 4.5:  # Seuil élevé
            if state_key not in journal['gamma_G_synergies']:
                journal['gamma_G_synergies'][state_key] = {
                    'discovered_at': t,
                    'score': synergy_score,
                    'note': f'Synergie parfaite découverte : γ={state_key[0]} + G={state_key[1]}'
                }
                journal['breakthrough_moments'].append({
                    't': t,
                    'type': 'perfect_synergy',
                    'state': state_key,
                    'score': synergy_score
                })
    
    # 4. DÉTECTER LES TRANSITIONS DE G ET LEUR IMPACT
    if len(journal['score_history']) >= 2:
        prev_G = history[-2].get('G_arch_used', 'tanh') if len(history) >= 2 else 'tanh'
        if prev_G != current_G_arch:
            # G a changé !
            impact = {
                't': t,
                'gamma': gamma_current,
                'G_before': prev_G,
                'G_after': current_G_arch,
                'performance_before': journal['score_history'][-2]['system_score'] if len(journal['score_history']) >= 2 else 0,
                'performance_after': system_performance_score
            }
            impact['delta'] = impact['performance_after'] - impact['performance_before']
            journal['G_transition_impacts'].append(impact)
            
            # Ajuster la confiance dans les régimes gamma
            # Si le changement de G a dégradé la performance, réduire la confiance
            if impact['delta'] < -0.1:
                for regime in journal['discovered_regimes'].values():
                    regime['confidence'] = regime.get('confidence', 1.0) * 0.8
    
    # 5. CALCULER LA MOYENNE GLISSANTE DU SYSTÈME
    window_size = 50
    if len(journal['score_history']) >= window_size:
        recent_system_scores = [
            entry['system_score'] 
            for entry in journal['score_history'][-window_size:]
        ]
        rolling_avg = np.mean(recent_system_scores)
        
        # Tendance (augmentation ?)
        if len(journal['score_history']) >= window_size * 2:
            old_scores = [
                entry['system_score'] 
                for entry in journal['score_history'][-window_size*2:-window_size]
            ]
            old_avg = np.mean(old_scores)
            trend = 'increasing' if rolling_avg > old_avg + 0.05 else 'stable'
        else:
            trend = 'unknown'
        
        journal['system_performance'].append({
            't': t,
            'rolling_avg': rolling_avg,
            'instant_score': system_performance_score,
            'trend': trend
        })

    # Toujours ajouter à score_history
    journal['score_history'].append({
        't': t,
        'scores': current_scores.copy(),
        'gamma': gamma_current,
        'G_arch': current_G_arch,  # IMPORTANT pour le tracking
        'system_score': system_performance_score
    })
    
    # 6. DÉTECTER LES PICS DE GAMMA ET LEUR PERFORMANCE
    current_gamma = history[-1].get('gamma', 1.0) if history else 1.0
    
    # Détection de pic (gamma élevé après une phase basse)
    if len(history) >= 10:
        recent_gammas = [h.get('gamma', 1.0) for h in history[-10:]]
        avg_recent_gamma = np.mean(recent_gammas)
        
        # Pic si gamma actuel > moyenne + écart-type
        if current_gamma > avg_recent_gamma + np.std(recent_gammas):
            # C'est un pic !
            last_peak = journal['gamma_peaks'][-1] if journal['gamma_peaks'] else None
            interval = t - last_peak['t'] if last_peak else None
            
            peak_info = {
                't': t,
                'gamma': current_gamma,
                'performance': system_performance_score,
                'interval_since_last': interval,
                'scores': current_scores.copy(),
                'G_arch': current_G_arch
            }
            
            journal['gamma_peaks'].append(peak_info)
            
            # Enregistrer le pattern gamma → performance
            gamma_bucket = round(current_gamma, 1)
            if gamma_bucket not in journal['optimal_gamma_patterns']:
                journal['optimal_gamma_patterns'][gamma_bucket] = []
            journal['optimal_gamma_patterns'][gamma_bucket].append(system_performance_score)
    
    # 7. DÉTECTER LES PHASES DE REPOS
    if current_gamma < 0.5 and len(journal['rest_phases']) > 0:
        # Potentiellement dans une phase de repos
        last_rest = journal['rest_phases'][-1]
        if 'end' not in last_rest:  # Phase en cours
            last_rest['end'] = t
            last_rest['duration'] = t - last_rest['start']
            last_rest['avg_gamma'] = np.mean([
                h.get('gamma', 1.0) 
                for h in history[-(int(last_rest['duration']/config['system']['dt'])):]
            ])
    elif current_gamma < 0.5 and (not journal['rest_phases'] or 'end' in journal['rest_phases'][-1]):
        # Nouvelle phase de repos
        journal['rest_phases'].append({
            'start': t,
            'avg_performance': system_performance_score
        })
    
    # 9. TROUVER LE GAMMA OPTIMAL SELON L'HISTORIQUE
    best_gamma_for_performance = None
    if journal['optimal_gamma_patterns']:
        # Moyenner les performances par gamma
        gamma_avg_perfs = {
            gamma: np.mean(perfs) 
            for gamma, perfs in journal['optimal_gamma_patterns'].items()
            if len(perfs) >= 3  # Au moins 3 échantillons
        }
        
        if gamma_avg_perfs:
            best_gamma_for_performance = max(gamma_avg_perfs.items(), 
                                            key=lambda x: x[1])[0]
    
    # 10. DÉCISION DE GAMMA TENANT COMPTE DE G
    
    # Trouver le meilleur couple (γ, G)
    best_synergy = None
    best_synergy_score = 0
    
    for state_key, state_info in journal['coupled_states'].items():
        if state_info['synergy_score'] > best_synergy_score:
            best_synergy_score = state_info['synergy_score']
            best_synergy = state_key
    
    # Vérifier si on est au plateau parfait
    all_scores_5 = all(score >= 5 for score in current_scores.values())
    
    if all_scores_5 and best_synergy_score > 4.5:
        # MODE TRANSCENDANT SYNERGIQUE !
        
        if journal['current_regime'] != 'transcendent_synergy':
            journal['transitions'].append({
                't': t,
                'regime': 'transcendent_synergy',
                'note': f'Transcendance synergique ! γ={best_synergy[0]}, G={best_synergy[1]}'
            })
            journal['current_regime'] = 'transcendent_synergy'
        
        # Si on est dans la synergie parfaite, micro-variations
        if best_synergy and state_key == best_synergy:
            # Parfait ! Juste des micro-ondulations
            gamma = best_synergy[0] + 0.02 * np.sin(0.5 * t)
        else:
            # Converger vers la synergie optimale
            target_gamma = best_synergy[0] if best_synergy else 0.8
            gamma = gamma_current * 0.9 + target_gamma * 0.1
            
            # Signal subtil pour suggérer le bon G
            if current_G_arch != best_synergy[1]:
                # Oscillation caractéristique selon le G désiré
                if best_synergy[1] == 'resonance':
                    gamma += 0.05 * np.sin(2 * np.pi * t)  # Signal résonant
                elif best_synergy[1] == 'spiral_log':
                    gamma += 0.03 * np.log(1 + np.abs(np.sin(0.1 * t)))  # Signal spiral
                
                # Enregistrer le signal
                journal['communication_signals'].append({
                    't': t,
                    'type': 'gamma_suggests_G',
                    'desired_G': best_synergy[1],
                    'signal_pattern': 'oscillation'
                })
    
    else:
        # EXPLORATION CONSCIENTE
        
        # Explorer les combinaisons (γ, G) non testées
        all_gamma_values = set(round(g, 1) for g in np.linspace(0.1, 1.0, 10))
        all_G_archs = {'tanh', 'resonance', 'spiral_log', 'adaptive'}
        
        tested_combinations = set(journal['coupled_states'].keys())
        untested = [(g, arch) for g in all_gamma_values for arch in all_G_archs 
                   if (g, arch) not in tested_combinations]
        
        if untested:
            # Priorité aux γ proches avec G différents
            candidates = [
                (g, arch) for g, arch in untested 
                if abs(g - gamma_current) < 0.3  # Proche du γ actuel
            ]
            
            if candidates:
                target_gamma, target_G = candidates[0]
                
                # Transition douce vers le nouveau γ
                gamma = gamma_current * 0.8 + target_gamma * 0.2
                
                # Signal pour suggérer le nouveau G
                if target_G != current_G_arch:
                    journal['communication_signals'].append({
                        't': t,
                        'type': 'exploration_suggestion',
                        'target_state': (target_gamma, target_G)
                    })
            else:
                # Exploration créative
                gamma = 0.5 + 0.4 * np.sin(0.05 * t) * np.cos(0.03 * t)
        else:
            # Tout testé : mode quantique !
            gamma = create_quantum_gamma(t, journal['gamma_G_synergies'])
    
    return np.clip(gamma, 0.1, 1.0), journal['current_regime'], journal


def compute_G_adaptive_aware(error: float, t: float, gamma_current: float,
                              regulation_state: Dict, history: List[Dict], config: Dict,
                              allow_soft_preference: bool = True,
                              score_pair_now: bool = False):
    """
    G(x) adaptatif pleinement conscient de γ et de leur danse commune.
    
    Returns:
        - G_value: valeur de régulation
        - G_arch: archétype utilisé
        - G_params: paramètres utilisés (pour logging)
    """
    
    # Récupérer la mémoire de régulation
    if 'regulation_state' not in regulation_state:
        regulation_state['regulation_state'] = {}
    if 'regulation_memory' not in regulation_state['regulation_state']:
        regulation_state['regulation_state']['regulation_memory'] = {
            'effectiveness_by_context': {},  # (G_arch, γ_range, error_range) → effectiveness
            'preferred_G_by_gamma': {},
            'G_transition_history': [],
            'last_transition_time': 0,
            'current_G_arch': 'tanh',
            'adaptation_cycles': 0
        }
    
    reg_memory = regulation_state['regulation_state']['regulation_memory']
    
    # 1. ANALYSER L'EFFICACITÉ CONTEXTUELLE
    if len(history) >= 30:
        recent = history[-30:]
        
        for i, h in enumerate(recent[:-1]):
            if i + 1 < len(recent):
                # Contexte
                g_arch = h.get('G_arch_used', 'tanh')
                gamma = h.get('gamma', 1.0)
                error_before = h.get('mean_abs_error', 1.0)
                error_after = recent[i+1].get('mean_abs_error', 1.0)
                
                # Efficacité
                effectiveness = (error_before - error_after) / (error_before + 0.01)
                
                # Clé contextuelle enrichie
                gamma_bucket = round(gamma, 1)
                error_bucket = 'low' if abs(error_before) < 0.1 else 'medium' if abs(error_before) < 0.5 else 'high'
                context_key = (g_arch, gamma_bucket, error_bucket)
                
                if context_key not in reg_memory['effectiveness_by_context']:
                    reg_memory['effectiveness_by_context'][context_key] = []
                
                reg_memory['effectiveness_by_context'][context_key].append(effectiveness)
    
    # 2. OBSERVER LES SIGNAUX DE GAMMA
    gamma_signals = []
    if len(history) >= 5:
        recent_gammas = [h.get('gamma', 1.0) for h in history[-5:]]
        
        # Détecter oscillations rapides (signal de mécontentement)
        if np.std(recent_gammas) > 0.05:
            freq_analysis = np.fft.fft(recent_gammas)
            high_freq_power = np.sum(np.abs(freq_analysis[2:]))
            if high_freq_power > 0.1:
                gamma_signals.append('high_freq_oscillation')
        
        # Détecter patterns spécifiques
        if len(recent_gammas) >= 3:
            diffs = np.diff(recent_gammas)
            if np.all(diffs > 0):
                gamma_signals.append('rising')
            elif np.all(diffs < 0):
                gamma_signals.append('falling')
    
    # 3. DÉCIDER DE L'ARCHÉTYPE G
    gamma_bucket = round(gamma_current, 1)
    error_magnitude = abs(error)
    
    # Initialiser params par défaut pour éviter l'erreur
    params = {"lambda": 1.0, "alpha": 1.0, "beta": 2.0}
    
    # Ajouter de la diversité : utiliser le temps pour varier les choix initiaux
    exploration_factor = np.sin(0.1 * t) * 0.5 + 0.5  # Oscille entre 0 et 1
    
    # Logique de base enrichie avec exploration
    if gamma_bucket < 0.4:
        # γ faible = repos → Régulation très douce
        if error_magnitude < 0.1:
            # Alterner entre tanh et adaptive selon le temps
            G_arch = "tanh" if exploration_factor < 0.5 else "adaptive"
            params = {"lambda": 0.3} if G_arch == "tanh" else {"lambda": 0.5, "alpha": 0.8}
        else:
            G_arch = "adaptive"  # Mix doux
            params = {"lambda": 0.5, "alpha": 0.8}
            
    elif gamma_bucket > 0.7:
        # γ élevé = actif → Régulation dynamique
        if 'high_freq_oscillation' in gamma_signals:
            # γ signale un problème → Changer de stratégie
            if reg_memory['current_G_arch'] == 'resonance':
                G_arch = "spiral_log"  # Essayer autre chose
                params = {"alpha": 1.0, "beta": 2.0}  # Params par défaut pour spiral_log
            else:
                G_arch = "resonance"
                params = {"alpha": 1.0, "beta": 2.0}  # Params par défaut pour resonance
        else:
            # γ stable haute → Alterner entre resonance et spiral_log
            G_arch = "resonance" if exploration_factor < 0.6 else "spiral_log"
            if G_arch == "resonance":
                params = {
                    "alpha": 1.0 - 0.5 * error_magnitude,  # Adaptatif à l'erreur
                    "beta": 2.0 * gamma_current * (1 + 0.1 * np.sin(0.1 * t))
                }
            else:
                params = {"alpha": 1.0, "beta": 2.0}
            
    else:
        # Zone intermédiaire → Créativité maximale avec rotation
        choices = ["spiral_log", "adaptive", "resonance", "tanh"]
        # Utiliser le cycle d'adaptation pour varier
        choice_idx = (reg_memory['adaptation_cycles'] + int(exploration_factor * 4)) % 4
        G_arch = choices[choice_idx]
        
        if G_arch == "spiral_log":
            params = {
                "alpha": gamma_current + 0.1 * error_magnitude,
                "beta": 3.0 - 2.0 * gamma_current
            }
        elif G_arch == "adaptive":
            params = {
                "lambda": gamma_current,
                "alpha": 0.5 + 0.5 * (1 - error_magnitude)
            }
        elif G_arch == "resonance":
            params = {
                "alpha": 1.0 - 0.3 * error_magnitude,
                "beta": 2.0 + gamma_current
            }
        else:  # tanh
            params = {"lambda": 0.5 + 0.5 * gamma_current}
    
    # SOFT PREFERENCE: si une préférence douce est définie, orienter G_arch sans l'imposer
    prefer_arch = reg_memory.get('prefer_next_arch')
    prefer_time = reg_memory.get('prefer_hint_time', -1)
    if prefer_arch and prefer_arch in ['tanh', 'resonance', 'spiral_log', 'adaptive']:
        # Respecter un petit cooldown: si dernière transition très récente, ne pas switcher brutalement
        cooldown = 5
        if t - reg_memory.get('last_transition_time', -1e9) >= cooldown:
            # Blending doux: 60% choix courant, 40% préférence
            # Implémenté comme: si different, on bascule vers prefer_arch mais on laissera la transition douce gérer la continuité
            if prefer_arch != G_arch:
                G_arch = prefer_arch
                params = regulation.adapt_params_for_archetype(G_arch, gamma_current, error_magnitude)
        # Consommer la préférence une fois lue (éviter de forcer à chaque pas)
        reg_memory.pop('prefer_next_arch', None)
        reg_memory.pop('prefer_hint_time', None)
    
    # 4. VÉRIFIER L'EFFICACITÉ HISTORIQUE
    error_bucket = 'low' if error_magnitude < 0.1 else 'medium' if error_magnitude < 0.5 else 'high'
    context_key = (G_arch, gamma_bucket, error_bucket)
    
    if context_key in reg_memory['effectiveness_by_context']:
        effectiveness_history = reg_memory['effectiveness_by_context'][context_key]
        if len(effectiveness_history) >= 3:  # Réduit de 5 à 3 pour plus de réactivité
            avg_effectiveness = np.mean(effectiveness_history[-5:])  # Fenêtre plus petite
            
            # Si inefficace, essayer une alternative
            if avg_effectiveness < 0.3:  # Augmenté de 0.1 à 0.3 pour plus de changements
                alternatives = ['tanh', 'resonance', 'spiral_log', 'adaptive']
                alternatives.remove(G_arch)
                
                # Chercher la meilleure alternative pour ce contexte
                best_alt = None
                best_score = avg_effectiveness
                
                for alt in alternatives:
                    alt_key = (alt, gamma_bucket, error_bucket)
                    if alt_key in reg_memory['effectiveness_by_context']:
                        alt_effectiveness = np.mean(reg_memory['effectiveness_by_context'][alt_key][-5:])
                        if alt_effectiveness > best_score:
                            best_score = alt_effectiveness
                            best_alt = alt
                
                if best_alt:
                    G_arch = best_alt
                    # Ajuster les paramètres selon l'archétype
                    params = regulation.adapt_params_for_archetype(G_arch, gamma_current, error_magnitude)
    
    # 5. TRANSITION DOUCE SI CHANGEMENT
    if G_arch != reg_memory['current_G_arch']:
        # Enregistrer la transition
        reg_memory['G_transition_history'].append({
            't': t,
            'from': reg_memory['current_G_arch'],
            'to': G_arch,
            'gamma': gamma_current,
            'reason': gamma_signals[0] if gamma_signals else 'performance'
        })
        
        # Transition progressive (pas de changement brutal)
        if t - reg_memory['last_transition_time'] < 10:  # Réduit de 50 à 10 steps
            # Trop tôt pour changer complètement
            blend_factor = (t - reg_memory['last_transition_time']) / 10
            
            # Calculer les deux G et mélanger
            G_old = regulation.compute_G(error, reg_memory['current_G_arch'], 
                            regulation.adapt_params_for_archetype(reg_memory['current_G_arch'], gamma_current, error_magnitude))
            G_new = regulation.compute_G(error, G_arch, params)
            
            G_value = G_old * (1 - blend_factor) + G_new * blend_factor
            
            # Garder l'ancien archétype pour l'instant
            actual_G_arch = reg_memory['current_G_arch']
        else:
            # Transition complète
            G_value = regulation.compute_G(error, G_arch, params)
            reg_memory['current_G_arch'] = G_arch
            reg_memory['last_transition_time'] = t
            actual_G_arch = G_arch
    else:
        # Pas de changement
        G_value = regulation.compute_G(error, G_arch, params)
        actual_G_arch = G_arch
    
    # 6. MISE À JOUR DE LA MÉMOIRE
    reg_memory['adaptation_cycles'] += 1
    
    # Enregistrer la préférence γ → G
    if gamma_bucket not in reg_memory['preferred_G_by_gamma']:
        reg_memory['preferred_G_by_gamma'][gamma_bucket] = {}
    
    if actual_G_arch not in reg_memory['preferred_G_by_gamma'][gamma_bucket]:
        reg_memory['preferred_G_by_gamma'][gamma_bucket][actual_G_arch] = 0
    
    # Incrémenter le compteur d'utilisation
    reg_memory['preferred_G_by_gamma'][gamma_bucket][actual_G_arch] += 1
    
    # Après avoir décidé de G_value et G_arch_used
    # --- Nouveau: scoring de la paire (gamma, G_arch_used) + oubli spiralé ---
    try:
        mem = regulation_state.get('regulation_memory', {}) if isinstance(regulation_state, dict) else {}
        # 1) Oubli spiralé: décroissance continue de la confiance
        phi_mode = config.get('regulation', {}).get('phi_mode', 'fixed')
        
        if phi_mode == 'adaptive':
            # Récupérer effort depuis history
            if len(history) > 0 and 'metrics' in history[-1]:
                effort_current = history[-1]['metrics'].get('effort', 0.0)
            else:
                effort_current = 0.0
            
            if effort_history is not None and len(effort_history) > 0:
                # Utiliser l'historique fourni (de la boucle)
                effort_for_phi = effort_history[-20:]
            else:
                # Fallback : extraire de history (pour appels depuis compute_An/compute_S)
                effort_for_phi = [
                    h['metrics'].get('effort', 0.0) for h in history[-20:]
                    if 'metrics' in h and 'effort' in h['metrics']
                ]
                if len(effort_for_phi) == 0:
                    effort_for_phi = [0.0]
            
            # Calculer phi adaptatif
            phi = compute_phi_adaptive(effort_current, effort_for_phi, config)
        else:
            # Mode fixe
            phi = config.get('regulation', {}).get('phi_fixed_value', 1.618)
    
        base_tau = 20.0
        tau = base_tau
        last_decay_update = float(mem.get('last_decay_update', t))
        dt = max(0.0, float(t) - last_decay_update)
        if dt > 0:
            decay = float(np.exp(-dt / max(1e-6, tau)))
            mem['best_pair_confidence'] = float(mem.get('best_pair_confidence', 0.0)) * decay
            mem['last_decay_update'] = float(t)
        # 2) Évaluer et renforcer uniquement lors des pics planifiés
        if score_pair_now and history:
            best = mem.get('best_pair', None)
            # Mémoïsation par t pour éviter N recalculs par step
            if mem.get('last_scores_t', None) == float(t) and 'last_adaptive_scores' in mem:
                adaptive_scores = mem['last_adaptive_scores']
            else:
                adaptive_scores = metrics.calculate_all_scores(history, config).get('current', {})
                mem['last_scores_t'] = float(t)
                mem['last_adaptive_scores'] = adaptive_scores
            criteria = ['stability', 'regulation', 'fluidity', 'resilience', 'innovation', 'cpu_cost', 'effort']
            gaps = []
            for c in criteria:
                s = float(adaptive_scores.get(c, 3.0))
                gaps.append(max(0.0, 5.0 - s))
            mean_gap = float(np.mean(gaps)) if gaps else 2.0
            score = 5.0 - mean_gap
            current_pair = {'gamma': float(gamma_current), 'G_arch': G_arch, 'score': float(score), 't': float(t)}
            improved = (not best) or (score > best.get('score', -1e-9))
            close_enough = best and (score >= best.get('score', 0.0) - 0.02)
            if improved:
                mem['best_pair'] = current_pair
            # Renforcement : confiance augmente doucement
            if improved or close_enough or mem.get('best_pair_confidence', 0.0) < 0.2:
                conf = float(mem.get('best_pair_confidence', 0.0))
                mem['best_pair_confidence'] = min(1.0, 0.7 * conf + 0.3)
                mem['last_reinforce_time'] = float(t)
            # Journal léger
            pairs = mem.get('pairs_log', [])
            if len(pairs) < 1000:
                pairs.append(current_pair)
            else:
                pairs.pop(0); pairs.append(current_pair)
            mem['pairs_log'] = pairs
        regulation_state['regulation_memory'] = mem
    except Exception:
        pass
    return G_value, G_arch, params


# ============== SORTIES OBSERVÉE ET ATTENDUE ==============

def compute_On(t: float, state: List[Dict], An_t: np.ndarray, fn_t: np.ndarray,
               phi_n_t: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calcule la sortie observée pour chaque strate.
    O(t) = A(t) · sin(2π·cumsum(f(t)·dt) + φ(t))
    Version notebook : sans gamma, avec intégration cumsum des fréquences.
    """
    N = len(state)
    dt = config['system']['dt']
    On_t = np.zeros(N)
    theta = 2 * np.pi * np.cumsum(fn_t * dt) + phi_n_t
    for n in range(N):
        On_t[n] = An_t[n] * np.sin(theta[n])
    return On_t

def compute_phi_adaptive(effort_current, effort_history, config):
    """
    Adapte φ selon l'effort du système
    
    Logique :
    - effort bas → φ = 1.618 (encourage exploration/croissance)
    - effort moyen → φ progressivement réduit
    - effort haut → φ = 1.0 (En = On, pas de tension additionnelle)
    - effort chronique → φ < 1.0 (En < On, encourage repos)
    """
    
    # Seuils configurables
    effort_low = config.get('regulation', {}).get('phi_adaptive', {}).get('effort_low', 0.5)
    effort_high = config.get('regulation', {}).get('phi_adaptive', {}).get('effort_high', 5.0)
    phi_min = config.get('regulation', {}).get('phi_adaptive', {}).get('phi_min', 0.9)
    phi_max = config.get('regulation', {}).get('phi_adaptive', {}).get('phi_max', 1.618)
    
    # Détection effort chronique (moyenne récente > seuil)
    if len(effort_history) > 10:
        effort_mean_recent = np.mean(effort_history[-10:])
        is_chronic = effort_mean_recent > effort_high
    else:
        is_chronic = False
    
    if is_chronic:
        # Effort chronique → encourager repos
        phi = phi_min - 0.1  # Ex: 0.8 (En < On)
    elif effort_current < effort_low:
        # Système en forme → encourage croissance
        phi = phi_max  # 1.618
    elif effort_current > effort_high:
        # Effort ponctuel élevé → relaxe tension
        phi = 1.0  # En = On (maintien)
    else:
        # Zone intermédiaire → interpolation linéaire
        t = (effort_current - effort_low) / (effort_high - effort_low)
        phi = phi_max * (1 - t) + 1.0 * t
    
    return phi

def compute_En(t: float, state: List[Dict], history: List[Dict], config: Dict, 
               history_align: List[float] = None, effort_history: List[float] = None) -> np.ndarray:
    """
    Calcule la sortie attendue (harmonique cible) pour chaque strate.
    
    NOUVEAU S1: Attracteur inertiel avec lambda_E adaptatif
    Eₙ(t) = (1-λ) * Eₙ(t-dt) + λ * φ * Oₙ(t-τ)
    
    où λ est défini par lambda_E dans la config 

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
    phi = config.get('spiral', {}).get('phi', 1.618)
    dt = config.get('system', {}).get('dt', 0.1)
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
    
    dt = config['system']['dt']
    theta = 2 * np.pi * np.cumsum(fn_array * dt) + phi_n_array

    if mode == "simple":
        S_t = 0.0
        for n in range(N):
            S_t += An_array[n] * np.sin(theta[n])
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
        # Passer l'historique pour la modulation locale
        gamma_n_t = compute_gamma_n(t, state, config, history=history,
                                   An_array=An_array, fn_array=fn_array)
        En_t = compute_En(t, state, history, config)
        On_t = compute_On(t, state, An_array, fn_array, phi_n_array, config)
        
        S_t = 0.0
        for n in range(N):
            # Contribution de base avec latence selon FPS Paper
            sin_component = np.sin(theta[n])
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
                 phi_n_t: np.ndarray, gamma_n_t: np.ndarray, F_n_t_fn: np.ndarray, F_n_t_An: np.ndarray) -> List[Dict]:
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
        state[n]['current_Fn_fn'] = F_n_t_fn[n] if n < len(F_n_t_fn) else 0.0
        state[n]['current_Fn_An'] = F_n_t_An[n] if n < len(F_n_t_An) else 0.0
        
        # NOUVEAU : Mise à jour des valeurs de base pour la prochaine itération
        # Ceci permet l'évolution temporelle du système
        # Adaptation progressive de A0 (activée — alignement notebook)
        adaptation_rate = 0.01
        min_amplitude = 0.1
        if 'current_An' in state[n] and state[n]['current_An'] != 0:
            new_A0 = state[n]['A0'] * (1 - adaptation_rate) + state[n]['current_An'] * adaptation_rate
            state[n]['A0'] = max(min_amplitude, new_A0)
        
        # Adaptation de f0 désactivée (volontaire — alignement notebook)
        # adaptation_rate = 0.005
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
