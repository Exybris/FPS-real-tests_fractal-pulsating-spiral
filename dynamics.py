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
    Calcule l'amplitude adaptative pour chaque strate.
    
    Aₙ(t) = A₀ · σ(Iₙ(t))
    
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
    
    for n in range(N):
        A0 = state[n]['A0']
        k = state[n]['k']
        x0 = state[n]['x0']
        
        # Amplitude adaptative via sigmoïde
        An_t[n] = A0 * compute_sigma(In_t[n], k, x0)
    
    return An_t

# ============== CALCUL DU SIGNAL INTER-STRATES ==============

def compute_S_i(t: float, n: int, history: List[Dict]) -> float:
    """
    Calcule le signal provenant des autres strates.
    
    Formule FPS correcte : S_i(t) = Σ(j≠n) Oj(t) * w_ji
    où w_ji sont les poids de connexion de la strate j vers la strate i.
    
    Args:
        t: temps actuel
        n: indice de la strate courante
        history: historique complet du système
    
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
    
    # Calculer la somme pondérée des signaux des autres strates
    # Note: dans la config actuelle, w_ni est défini pour chaque strate
    # mais nous avons besoin de w_ji (connexions entrantes)
    # Pour l'instant, on utilise une matrice de connexion symétrique
    
    S_i = 0.0
    N = len(On_prev)
    
    # Pour chaque autre strate j
    for j in range(N):
        if j != n:  # Exclure la strate courante
            # Utiliser le signal observé de la strate j
            Oj = On_prev[j]
            
            # Poids de connexion basé sur la distance, variation exploratoire valide
            # Formule FPS : connexions plus fortes entre strates proches
            distance = abs(j - n)
            # Connexion cyclique (la strate 0 est proche de la strate N-1)
            distance = min(distance, N - distance)
            
            # Poids décroissant avec la distance selon une gaussienne
            sigma_connexion = N / 4.0  # Portée des connexions
            w_ji = np.exp(-distance**2 / (2 * sigma_connexion**2))
            
            S_i += Oj * w_ji
    
    # Normaliser par la somme des poids pour garder l'échelle
    total_weight = 0.0
    for j in range(N):
        if j != n:
            distance = min(abs(j - n), N - abs(j - n))
            total_weight += np.exp(-distance**2 / (2 * (N/4.0)**2))
    
    if total_weight > 0:
        S_i = S_i / total_weight
    
    return S_i


# ============== MODULATION DE FRÉQUENCE ==============

def compute_delta_fn(t: float, alpha_n: float, w_ni: List[float], S_i: float) -> float:
    """
    Calcule la modulation de fréquence.
    
    Δfₙ(t) = αₙ · Σᵢ w_{ni} · Sᵢ(t)
    
    Args:
        t: temps actuel
        alpha_n: souplesse d'adaptation
        w_ni: poids de connexion
        S_i: signal des autres strates
    
    Returns:
        float: modulation de fréquence
    """
    # Pour phase 1, on simplifie en utilisant S_i comme signal moyen pondéré
    # La modulation ne doit pas utiliser sum(w_ni) car cela donne 0
    # À la place, on utilise la valeur absolue moyenne des poids non-nuls
    non_zero_weights = [abs(w) for w in w_ni if w != 0]
    if non_zero_weights:
        avg_weight = np.mean(non_zero_weights)
        modulation = alpha_n * avg_weight * S_i
    else:
        modulation = 0.0
    return modulation


def compute_fn(t: float, state: List[Dict], An_t: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calcule la fréquence modulée pour chaque strate.
    
    fₙ(t) = f₀ₙ + Δfₙ(t)
    
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
    
    for n in range(N):
        f0n = state[n]['f0']
        alpha_n = state[n]['alpha']
        w_ni = state[n]['w']
        
        # Calcul du signal des autres strates
        S_i = compute_S_i(t, n, history)
        
        # Modulation de fréquence
        delta_fn = compute_delta_fn(t, alpha_n, w_ni, S_i)
        
        # Fréquence finale
        fn_t[n] = f0n + delta_fn
    
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
    
    for n in range(N):
        if dynamic_phi:
            # Mode dynamique : évolution spiralée des phases
            # φₙ(t) = φₙ(0) + 2π * (n/N) * φ * t/T
            phi_golden = config.get('spiral', {}).get('phi', 1.618)
            T = config.get('system', {}).get('T', 100)
            phi_n_t[n] = state[n].get('phi', 0.0) + 2 * np.pi * (n/N) * phi_golden * (t/T)
        else:
            # Mode statique
            phi_n_t[n] = state[n].get('phi', 0.0)
    
    return phi_n_t


# ============== LATENCE EXPRESSIVE ==============

def compute_gamma(t: float, mode: str = "static", T: Optional[float] = None) -> float:
    """
    Calcule la latence expressive globale.
    
    Args:
        t: temps actuel
        mode: "static" ou "dynamic"
        T: durée totale (pour mode dynamic)
    
    Returns:
        float: latence entre 0 et 1
    
    Formes:
        - static: γ(t) = 1.0
        - dynamic: γ(t) = 1/(1 + exp(-2(t - T/2)))
    """
    if mode == "static":
        return 1.0
    elif mode == "dynamic" and T is not None:
        # Sigmoïde centrée à T/2
        k = 2.0  # Paramètre de pente fixé pour phase 1
        t0 = T / 2
        return 1.0 / (1.0 + np.exp(-k * (t - t0)))
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
    gamma_n_mode = latence_config.get('gamma_n_mode', 'static')
    T = config.get('system', {}).get('T', 100)
    
    if gamma_n_mode == "static":
        # Mode statique : toutes les strates à 1.0
        gamma_n_t[:] = 1.0
    elif gamma_n_mode == "dynamic":
        # Mode dynamique avec paramètres par défaut ou depuis config
        gamma_n_dynamic = latence_config.get('gamma_n_dynamic', {})
        k_n = gamma_n_dynamic.get('k_n', 2.0)
        t0_n = gamma_n_dynamic.get('t0_n', T / 2)
        
        # Sigmoïde pour chaque strate
        for n in range(N):
            gamma_n_t[n] = 1.0 / (1.0 + np.exp(-k_n * (t - t0_n)))
    
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


def compute_En(t: float, state: List[Dict], history: List[Dict], config: Dict) -> np.ndarray:
    """
    Calcule la sortie attendue (harmonique cible) pour chaque strate.
    
    Hypothèse exploratoire phase 1:
    Eₙ(t) = φ · Oₙ(t-1) où φ est le nombre d'or
    
    Args:
        t: temps actuel
        state: état des strates
        history: historique
        config: configuration
    
    Returns:
        np.ndarray: sorties attendues

    Cette formule exploratoire est temporaire
    """
    N = len(state)
    En_t = np.zeros(N)
    
    # Nombre d'or
    phi = config.get('spiral', {}).get('phi', 1.618)
    
    if len(history) > 0:
        # Attracteur basé sur le nombre d'or
        last_On = history[-1].get('O', np.zeros(N))
        if isinstance(last_On, np.ndarray) and len(last_On) == N:
            En_t = phi * last_On
        else:
            # Valeur par défaut si historique incomplet
            for n in range(N):
                En_t[n] = state[n]['A0']
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
    Calcule le coefficient d'accord spiralé.
    
    C(t) = (1/N) · Σ cos(φₙ₊₁ - φₙ)
    
    Args:
        t: temps actuel
        phi_n_array: phases de toutes les strates
    
    Returns:
        float: coefficient d'accord entre -1 et 1
    """
    N = len(phi_n_array)
    if N <= 1:
        return 1.0
    
    # Somme des cosinus entre phases adjacentes
    cos_sum = 0.0
    for n in range(N - 1):
        cos_sum += np.cos(phi_n_array[n + 1] - phi_n_array[n])
    
    return cos_sum / (N - 1)


def compute_A(t: float, delta_fn_array: np.ndarray) -> float:
    """
    Calcule la modulation moyenne.
    
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
    Calcule le signal global du système.
    
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
        - "extended": avec γₙ(t) et G(Eₙ(t) - Oₙ(t))
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
        # Version étendue avec latence et régulation
        # S(t) = Σₙ Aₙ(t)·sin(2π·fₙ(t)·t + φₙ(t))·γₙ(t)·[1 + G(Eₙ(t) - Oₙ(t))]
        
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
            # Contribution de base avec latence
            contribution = An_array[n] * np.sin(2 * np.pi * fn_array[n] * t + phi_n_array[n]) * gamma_n_t[n]
            
            # Facteur de régulation [1 + G(En - On)]
            error = En_t[n] - On_t[n]
            
            # Paramètres pour la fonction G
            G_arch = config.get('regulation', {}).get('G_arch', 'tanh')
            G_params = {
                'lambda': config.get('regulation', {}).get('lambda', 1.0),
                'alpha': config.get('regulation', {}).get('alpha', 1.0),
                'beta': config.get('regulation', {}).get('beta', 2.0)
            }
            
            # Calculer G(error)
            if hasattr(regulation, 'compute_G'):
                G_value = regulation.compute_G(error, G_arch, G_params)
            else:
                # Fallback sur tanh si regulation n'est pas disponible
                G_value = np.tanh(G_params['lambda'] * error)
            
            # Appliquer le facteur de modulation
            # Le facteur 0.1 limite l'impact de la régulation pour éviter l'instabilité
            modulation_factor = 1.0 + 0.1 * G_value
            
            S_t += contribution * modulation_factor
        
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


def compute_L(t: float, signal_array: Union[np.ndarray, List[float]]) -> int:
    """
    Calcule l'indice de latence maximale ou le lag optimal.
    
    Pour phase 1 : retourne l'indice de la strate dominante.
    Pour phase 2 : pourrait calculer un lag temporel optimal.
    
    Args:
        t: temps actuel
        signal_array: signaux ou amplitudes
    
    Returns:
        int: indice de la strate avec amplitude max ou lag optimal
    """
    if len(signal_array) == 0:
        return 0
    
    # Mode 1 : Indice de la strate dominante
    if isinstance(signal_array, np.ndarray) and signal_array.ndim == 1:
        # C'est un array d'amplitudes par strate
        return int(np.argmax(np.abs(signal_array)))
    
    # Mode 2 : Si c'est une série temporelle, calculer le lag optimal
    if len(signal_array) > 20:
        try:
            # Convertir en array numpy
            signal = np.array(signal_array)
            
            # Autocorrélation simple
            signal_norm = signal - np.mean(signal)
            autocorr = np.correlate(signal_norm, signal_norm, mode='same')
            mid = len(autocorr) // 2
            autocorr_positive = autocorr[mid:mid+20]  # Regarder les 20 premiers lags
            
            # Chercher le premier pic après lag 0
            for lag in range(1, len(autocorr_positive)-1):
                if (autocorr_positive[lag] > autocorr_positive[lag-1] and 
                    autocorr_positive[lag] > autocorr_positive[lag+1]):
                    return lag
        except:
            pass
    
    # Par défaut
    return int(np.argmax(np.abs(signal_array))) if len(signal_array) > 0 else 0


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
