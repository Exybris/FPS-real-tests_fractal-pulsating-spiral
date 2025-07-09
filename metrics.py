"""
metrics.py - Calcul des métriques FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Les métriques (effort, entropy, etc.) sont ajustables : toute
modification ou alternative testée doit être documentée et laissée
ouverte dans la config.
---------------------------------------------------------------

Ce module quantifie tous les aspects du système FPS :
- Performance computationnelle (CPU, mémoire)
- Effort d'adaptation (interne, transitoire, chronique)
- Qualité dynamique (fluidité, stabilité, innovation)
- Résilience et régulation
- Détection d'états (stable, transitoire, chronique, chaotique)

Les métriques sont le miroir empirique du système, permettant
la falsification et le raffinement continu.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
from scipy import signal
import time
import csv
import h5py
import os
from typing import Dict, List, Union, Optional, Tuple, Any
import warnings
from utils import deep_convert


# ============== MÉTRIQUES DE PERFORMANCE ==============

def compute_cpu_step(start_time: float, end_time: float, N: int) -> float:
    """
    Calcule le temps CPU normalisé par pas et par strate.
    
    cpu_step = (end_time - start_time) / N
    
    Args:
        start_time: temps début (time.perf_counter())
        end_time: temps fin
        N: nombre de strates
    
    Returns:
        float: temps CPU moyen par strate en secondes
    """
    if N <= 0:
        return 0.0
    return (end_time - start_time) / N


# ============== MÉTRIQUES D'EFFORT ==============

def compute_effort(delta_An_array: np.ndarray, delta_fn_array: np.ndarray, 
                   delta_gamma_n_array: np.ndarray, An_ref: float, fn_ref: float, 
                   gamma_ref: float) -> float:
    """
    Calcule l'effort d'adaptation interne du système.
    
    Version corrigée qui évite l'explosion quand les valeurs de référence → 0 :
    effort = Σₙ [|ΔAₙ|/(|Aₙ|+ε) + |Δfₙ|/(|fₙ|+ε) + |Δγₙ|/(|γₙ|+ε)]
    
    Args:
        delta_An_array: variations d'amplitude
        delta_fn_array: variations de fréquence
        delta_gamma_n_array: variations de latence
        An_ref: valeur de référence pour An (mean ou max)
        fn_ref: valeur de référence pour fn
        gamma_ref: valeur de référence pour gamma
    
    Returns:
        float: effort total normalisé
    
    Note:
        L'effort mesure maintenant le changement RELATIF par rapport
        aux valeurs actuelles, pas par rapport au maximum global.
    """
    effort = 0.0
    
    # Protection contre division par zéro avec floor minimum
    # Pour éviter overflow quand les valeurs de référence sont très petites
    # Utilisation d'un epsilon adaptatif basé sur l'échelle des variations
    epsilon = 0.01
    
    # Calcul de l'effort comme changement relatif
    # Si la référence est trop petite, on utilise epsilon
    An_denom = max(abs(An_ref), epsilon)
    fn_denom = max(abs(fn_ref), epsilon) 
    gamma_denom = max(abs(gamma_ref), epsilon)
    
    # Somme des efforts relatifs pour chaque composante
    effort_An = np.sum(np.abs(delta_An_array)) / An_denom
    effort_fn = np.sum(np.abs(delta_fn_array)) / fn_denom
    effort_gamma = np.sum(np.abs(delta_gamma_n_array)) / gamma_denom
    
    effort = effort_An + effort_fn + effort_gamma
    
    # Vérification finale contre NaN/Inf et saturation
    if np.isnan(effort) or np.isinf(effort):
        # Si malgré tout on a un problème, retourner une valeur par défaut
        effort = 0.0
    
    # NOUVEAU : Saturation pour éviter les valeurs extrêmes
    MAX_EFFORT = 100.0  # Limite raisonnable pour l'effort
    effort = min(effort, MAX_EFFORT)
    
    return effort


def compute_effort_status(effort_t: float, effort_history: List[float], 
                          config: Dict) -> str:
    """
    Détermine le statut de l'effort : stable, transitoire ou chronique.
    
    Args:
        effort_t: effort actuel
        effort_history: historique des efforts
        config: configuration (seuils)
    
    Returns:
        str: "stable", "transitoire" ou "chronique"
    
    Logique:
        - stable: effort dans la norme
        - transitoire: pic temporaire (adaptation ponctuelle)
        - chronique: effort élevé persistant (système en lutte)
    """
    if len(effort_history) < 10:
        # Pas assez d'historique - on considère stable par défaut
        return "stable"
    
    # Calcul des statistiques sur les 10 derniers pas
    recent_efforts = effort_history[-10:]
    mean_recent = np.mean(recent_efforts)
    
    # Seuils adaptatifs basés sur l'historique complet
    if len(effort_history) >= 50:
        # Moyenne et écart-type sur une fenêtre plus large
        long_term = effort_history[-50:]
        mean_long = np.mean(long_term)
        std_long = np.std(long_term)
        
        # Détection transitoire : pic > 2σ
        if effort_t > mean_long + 2 * std_long:
            return "transitoire"
        
        # Détection chronique : moyenne récente élevée
        # Si std_long est très petit (effort constant), utiliser un seuil absolu
        if std_long < 0.01:
            # Effort constant - utiliser les seuils de config
            thresholds = config.get('to_calibrate', {})
            if mean_recent > thresholds.get('effort_chronique_threshold', 1.5):
                return "chronique"
        else:
            # Effort variable - utiliser la logique adaptative
            if mean_recent > mean_long + std_long:
                # Vérifier la persistance
                high_count = sum(1 for e in recent_efforts if e > mean_long + std_long)
                if high_count >= 7:  # 70% du temps récent
                    return "chronique"
    
    # Sinon, utiliser des seuils fixes depuis config
    thresholds = config.get('to_calibrate', {})
    
    # Seuil transitoire
    if effort_t > thresholds.get('effort_transitoire_threshold', 2.0):
        return "transitoire"
    
    # Seuil chronique sur la moyenne
    if mean_recent > thresholds.get('effort_chronique_threshold', 1.5):
        return "chronique"
    
    return "stable"


def compute_mean_high_effort(effort_history: List[float], percentile: int = 80) -> float:
    """
    Calcule la moyenne haute de l'effort (percentile élevé).
    
    Mesure l'effort chronique en regardant les valeurs hautes.
    
    Args:
        effort_history: historique complet des efforts
        percentile: percentile à considérer (80 par défaut)
    
    Returns:
        float: moyenne des efforts au-dessus du percentile
    """
    if len(effort_history) == 0:
        return 0.0
    
    if len(effort_history) < 10:
        # Pas assez de données - retourner la moyenne simple
        return np.mean(effort_history)
    
    # Filtrer les valeurs aberrantes (NaN, Inf, et valeurs énormes)
    MAX_REASONABLE_EFFORT = 1e6
    filtered_efforts = []
    
    for effort in effort_history:
        if np.isfinite(effort) and abs(effort) < MAX_REASONABLE_EFFORT:
            filtered_efforts.append(effort)
    
    if len(filtered_efforts) == 0:
        return 0.0
    
    # Calculer le seuil du percentile sur les valeurs filtrées
    threshold = np.percentile(filtered_efforts, percentile)
    
    # Moyenner les valeurs au-dessus
    high_efforts = [e for e in filtered_efforts if e >= threshold]
    
    if len(high_efforts) == 0:
        return threshold
    
    return np.mean(high_efforts)


def compute_d_effort_dt(effort_history: List[float], dt: float) -> float:
    """
    Calcule la dérivée temporelle de l'effort.
    
    Mesure les variations brusques d'effort (transitoires).
    
    Args:
        effort_history: historique des efforts
        dt: pas de temps
    
    Returns:
        float: dérivée d_effort/dt
    """
    if len(effort_history) < 2:
        return 0.0
    
    # Dérivée simple entre les deux derniers points
    d_effort = effort_history[-1] - effort_history[-2]
    d_effort_dt = d_effort / dt
    
    # Protection contre les valeurs aberrantes
    # Si la dérivée est énorme, c'est probablement dû à un overflow
    MAX_DERIVATIVE = 1e6  # Limite raisonnable pour une dérivée
    
    if np.isnan(d_effort_dt) or np.isinf(d_effort_dt):
        return 0.0
    elif abs(d_effort_dt) > MAX_DERIVATIVE:
        # Clipper à la valeur max avec le bon signe
        return MAX_DERIVATIVE if d_effort_dt > 0 else -MAX_DERIVATIVE
    
    return d_effort_dt


# ============== MÉTRIQUES DE QUALITÉ DYNAMIQUE ==============

def compute_variance_d2S(S_history: List[float], dt: float) -> float:
    """
    Calcule la variance de la dérivée seconde de S(t).
    
    Mesure la fluidité : une faible variance indique des transitions douces.
    
    Args:
        S_history: historique du signal global
        dt: pas de temps
    
    Returns:
        float: variance de d²S/dt²
    """
    if len(S_history) < 3:
        return 0.0
    
    # Conversion en array pour calculs
    S_array = np.array(S_history)
    
    # Première dérivée
    dS_dt = np.gradient(S_array, dt)
    
    # Seconde dérivée
    d2S_dt2 = np.gradient(dS_dt, dt)
    
    # Variance
    return np.var(d2S_dt2)


def compute_fluidity(variance_d2S: float, reference_variance: float = 175.0) -> float:
    """
    Calcule la fluidité du système basée sur la variance de d²S/dt².
    
    Une faible variance indique des transitions douces (haute fluidité).
    Utilise une sigmoïde inversée pour une sensibilité optimale.
    
    Args:
        variance_d2S: variance de la dérivée seconde du signal
        reference_variance: variance de référence (défaut: médiane empirique)
    
    Returns:
        float: fluidité entre 0 (saccadé) et 1 (très fluide)
    """
    if variance_d2S <= 0:
        return 1.0  # Variance nulle = parfaitement fluide
    
    x = variance_d2S / reference_variance
    k = 5.0  # Sensibilité de la transition
    
    return 1 / (1 + np.exp(k * (x - 1)))


def compute_entropy_S(S_t: Union[float, List[float], np.ndarray], 
                      sampling_rate: float) -> float:
    """
    Calcule l'entropie spectrale du signal S(t).
    
    Mesure l'innovation : haute entropie = riche en fréquences diverses.
    Utilise l'entropie de Shannon sur le spectre de puissance normalisé.
    
    Args:
        S_t: signal (peut être juste la valeur actuelle ou une fenêtre)
        sampling_rate: fréquence d'échantillonnage (1/dt)
    
    Returns:
        float: entropie spectrale entre 0 et 1
    """
    # Si on n'a qu'une valeur scalaire, on utilise une approximation basée sur la magnitude
    if np.isscalar(S_t):
        # Approximation : mapper la magnitude dans [0,1] pour l'entropie
        # Plus la valeur est proche de 0, moins il y a d'information
        magnitude = abs(S_t)
        if magnitude < 0.1:
            return 0.1  # Très peu d'information
        elif magnitude > 10:
            return 0.9  # Beaucoup d'information
        else:
            # Fonction sigmoïde pour mapper [0.1, 10] -> [0.1, 0.9]
            return 0.1 + 0.8 / (1 + np.exp(-0.5 * (magnitude - 5)))
    
    # Si on a moins de 10 points, calculer une entropie approximative
    if len(S_t) < 10:
        # Entropie basée sur la variance du signal court
        variance = np.var(S_t)
        # Normaliser la variance pour obtenir une entropie entre 0 et 1
        # Variance élevée = plus d'information = entropie plus haute
        return min(0.9, 0.1 + 0.8 * np.tanh(variance))
    
    try:
        # Calcul du spectre de puissance
        freqs, psd = signal.periodogram(S_t, sampling_rate)
        
        # Normalisation pour obtenir une distribution de probabilité
        psd_norm = psd / np.sum(psd)
        
        # Éviter log(0)
        psd_norm = psd_norm + 1e-15
        
        # Entropie de Shannon
        entropy = -np.sum(psd_norm * np.log(psd_norm))
        
        # Normalisation par l'entropie maximale
        max_entropy = np.log(len(psd_norm))
        if max_entropy > 0:
            entropy_normalized = entropy / max_entropy
        else:
            entropy_normalized = 0.5
        
        return np.clip(entropy_normalized, 0.0, 1.0)
        
    except Exception as e:
        warnings.warn(f"Erreur dans compute_entropy_S: {e}")
        return 0.5


def compute_max_median_ratio(S_history: List[float]) -> float:
    """
    Calcule le ratio max/médiane du signal.
    
    Mesure la stabilité : un ratio élevé indique des pics extrêmes.
    
    Args:
        S_history: historique du signal
    
    Returns:
        float: ratio max(|S|) / median(|S|)
    """
    if len(S_history) < 10:
        return 1.0
    
    # Valeurs absolues
    S_abs = np.abs(S_history)
    
    # Protection contre médiane nulle
    median_val = np.median(S_abs)
    if median_val < 1e-10:
        median_val = 1e-10
    
    return np.max(S_abs) / median_val


# ============== MÉTRIQUES DE RÉGULATION ==============

def compute_mean_abs_error(En_array: np.ndarray, On_array: np.ndarray) -> float:
    """
    Calcule l'erreur absolue moyenne entre attendu et observé.
    
    mean(|Eₙ(t) - Oₙ(t)|)
    
    Mesure la qualité de la régulation : faible = bonne convergence.
    
    Args:
        En_array: sorties attendues
        On_array: sorties observées
    
    Returns:
        float: erreur absolue moyenne
    """
    if len(En_array) == 0 or len(On_array) == 0:
        return 0.0
    
    return np.mean(np.abs(En_array - On_array))


# ============== MÉTRIQUES DE RÉSILIENCE ==============

def compute_t_retour(S_history: List[float], t_choc: int, dt: float, 
                     threshold: float = 0.95) -> float:
    """
    Calcule le temps de retour à l'équilibre après perturbation.
    
    Temps pour revenir à 95% de l'état pré-choc.
    
    Args:
        S_history: historique du signal
        t_choc: indice temporel du choc
        dt: pas de temps
        threshold: seuil de retour (0.95 = 95%)
    
    Returns:
        float: temps de retour en unités de temps
    
    Note:
        État pré-choc = moyenne de |S(t)| sur fenêtre [t_choc-10*dt, t_choc]
    """
    if t_choc >= len(S_history) or t_choc < 10:
        return 0.0
    
    # État pré-choc : moyenne sur EXACTEMENT 10 pas avant le choc
    pre_shock_window = S_history[max(0, t_choc-10):t_choc]
    if len(pre_shock_window) == 0:
        return 0.0
    
    # Valeur de référence avant le choc
    etat_pre_choc = np.mean(np.abs(pre_shock_window))

    # Cas dégénéré : fenêtre quasi nulle (p. ex. S passe par 0)
    if etat_pre_choc < 1e-6:
        # Repli : prendre plutôt le max absolu de la même fenêtre
        etat_pre_choc = np.max(np.abs(pre_shock_window))

    # Si c'est toujours ≈0, on considère que le système n'est pas revenu ;
    # on renverra la durée totale restante (pénalité maximale)
    if etat_pre_choc < 1e-6:
        return (len(S_history) - t_choc) * dt

    # Chercher quand |S(t)| revient à ±5 % de l'état pré-choc
    tolerance = (1 - threshold) * etat_pre_choc
    
    for i in range(t_choc + 1, len(S_history)):
        # Valeur instantanée, pas de moyenne glissante
        current_value = abs(S_history[i])
        
        # Vérifier si on est revenu dans la tolérance
        if abs(current_value - etat_pre_choc) <= tolerance:
            return (i - t_choc) * dt
    
    # Pas encore revenu à l'équilibre
    return (len(S_history) - t_choc) * dt


def compute_continuous_resilience(C_history: List[float], S_history: List[float], 
                                 perturbation_active: bool = True) -> float:
    """
    Calcule la résilience sous perturbation continue.
    
    Pour une perturbation continue (ex: sinusoïdale), mesure la capacité
    du système à maintenir sa cohérence et stabilité.
    
    Args:
        C_history: historique de la cohérence C(t)
        S_history: historique du signal S(t)
        perturbation_active: si une perturbation est active
    
    Returns:
        float: score de résilience continue [0, 1]
        - 1.0 = excellente résilience (maintien de cohérence)
        - 0.0 = mauvaise résilience (perte de cohérence)
    
    Note:
        Métrique complémentaire à t_retour pour perturbations non-ponctuelles
    """
    if not perturbation_active or len(C_history) < 20:
        return 1.0  # Pas de perturbation ou pas assez de données
    
    # Utiliser une fenêtre récente
    window_size = min(100, len(C_history))
    C_recent = C_history[-window_size:]
    S_recent = S_history[-window_size:]
    
    # 1. Stabilité de la cohérence sous perturbation
    # Une bonne résilience = C(t) reste élevé malgré la perturbation
    C_mean = np.mean(C_recent)
    C_std = np.std(C_recent)
    
    # Score de stabilité de C(t)
    C_stability_score = C_mean / (1 + C_std) if C_mean > 0 else 0.0
    
    # 2. Stabilité ET expressivité de S(t)
    S_mean = np.mean(S_recent)
    S_std = np.std(S_recent)
    S_power = np.mean(np.abs(S_recent))
    
    if S_power > 0:
        # Coefficient de variation - mesure de stabilité relative
        S_cv = S_std / S_power
        
        # Score de stabilité de S(t) : CV faible = signal stable
        if S_cv < 0.5:  # Très stable
            S_stability_score = 1.0
        elif S_cv < 1.0:  # Stable
            S_stability_score = 0.9 - 0.2 * (S_cv - 0.5)
        elif S_cv < 2.0:  # Moyennement stable
            S_stability_score = 0.7 - 0.3 * (S_cv - 1.0)
        else:  # Instable
            S_stability_score = 0.4
        
        # Score d'expressivité : le signal garde son amplitude
        expressivity_score = np.tanh(2 * S_power)  # Saturation douce
        
        # Combiner stabilité et expressivité de S(t)
        S_combined_score = 0.6 * S_stability_score + 0.4 * expressivity_score
    else:
        # Signal effondré
        S_combined_score = 0.0
    
    # 3. Résilience combinée
    # Pondération : C(t) reste le plus important, mais S(t) compte significativement
    continuous_resilience = 0.6 * C_stability_score + 0.4 * S_combined_score
    
    return float(np.clip(continuous_resilience, 0.0, 1.0))


def compute_adaptive_resilience(config: Dict, metrics: Dict, 
                               C_history: List[float] = None, 
                               S_history: List[float] = None,
                               t_choc: int = None, dt: float = 0.05) -> Dict[str, Any]:
    """
    Calcule la résilience adaptative selon le type de perturbation.
    
    Cette fonction unifie t_retour et continuous_resilience en sélectionnant
    automatiquement la métrique appropriée selon le type de perturbation configuré.
    
    Args:
        config: configuration complète
        metrics: métriques déjà calculées (pour t_retour existant)
        C_history: historique de cohérence (pour continuous_resilience)
        S_history: historique du signal
        t_choc: indice temporel du choc (pour t_retour)
        dt: pas de temps
    
    Returns:
        Dict contenant:
        - 'type': 'punctual' ou 'continuous'
        - 'value': valeur de résilience [0, 1] ou temps
        - 'score': score normalisé [1-5]
        - 'metric_used': 't_retour' ou 'continuous_resilience'
    """
    # Déterminer le type de perturbation
    has_continuous_perturbation = False
    perturbation_type = 'none'
    
    # Nouvelle structure avec input.perturbations
    input_cfg = config.get('system', {}).get('input', {})
    perturbations = input_cfg.get('perturbations', [])
    
    for pert in perturbations:
        pert_type = pert.get('type', 'none')
        if pert_type in ['sinus', 'bruit', 'rampe']:
            has_continuous_perturbation = True
            perturbation_type = 'continuous'
            break
        elif pert_type == 'choc':
            perturbation_type = 'punctual'
    
    # Si pas trouvé dans la nouvelle structure, vérifier l'ancienne (compatibilité)
    if perturbation_type == 'none' and config:
        old_pert = config.get('system', {}).get('perturbation', {})
        old_type = old_pert.get('type', 'none')
        if old_type in ['sinus', 'bruit', 'rampe']:
            has_continuous_perturbation = True
            perturbation_type = 'continuous'
        elif old_type == 'choc':
            perturbation_type = 'punctual'
    
    result = {
        'type': perturbation_type,
        'metric_used': None,
        'value': None,
        'score': 3,  # Score par défaut
        'raw_value': None
    }
    
    if has_continuous_perturbation:
        # Utiliser continuous_resilience
        result['metric_used'] = 'continuous_resilience'
        
        # Préférer la valeur moyenne si disponible
        cont_resilience = metrics.get('continuous_resilience_mean', 
                                     metrics.get('continuous_resilience', None))
        
        # Si pas disponible, calculer
        if cont_resilience is None and C_history is not None and S_history is not None:
            cont_resilience = compute_continuous_resilience(C_history, S_history, True)
        
        if cont_resilience is not None:
            result['value'] = cont_resilience
            result['raw_value'] = cont_resilience
            
            # Calculer le score 1-5
            if cont_resilience >= 0.90:
                result['score'] = 5  # Excellence
            elif cont_resilience >= 0.75:
                result['score'] = 4  # Très bon
            elif cont_resilience >= 0.60:
                result['score'] = 3  # Bon
            elif cont_resilience >= 0.40:
                result['score'] = 2  # Acceptable
            else:
                result['score'] = 1  # Faible
    
    else:
        # Utiliser t_retour pour perturbation ponctuelle ou absence
        result['metric_used'] = 't_retour'
        
        # Récupérer t_retour existant
        t_retour = metrics.get('resilience_t_retour', 
                               metrics.get('t_retour', None))
        
        # Si pas disponible et qu'on a les données nécessaires, calculer
        if t_retour is None and S_history is not None and t_choc is not None:
            t_retour = compute_t_retour(S_history, t_choc, dt)
        
        if t_retour is not None:
            result['raw_value'] = t_retour
            # Normaliser t_retour en score [0, 1] pour cohérence
            # Plus t_retour est petit, meilleure est la résilience
            if t_retour == 0:
                result['value'] = 1.0
            else:
                result['value'] = 1.0 / (1.0 + t_retour)
            
            # Calculer le score 1-5
            if t_retour < 1.0:
                result['score'] = 5  # Récupération très rapide
            elif t_retour < 2.0:
                result['score'] = 4  # Récupération rapide
            elif t_retour < 5.0:
                result['score'] = 3  # Récupération modérée
            elif t_retour < 10.0:
                result['score'] = 2  # Récupération lente
            else:
                result['score'] = 1  # Très lente
    
    return result


# ============== VÉRIFICATION DES SEUILS ==============

def check_thresholds(metrics_dict: Dict[str, float], 
                     thresholds_dict: Dict[str, float]) -> Dict[str, bool]:
    """
    Vérifie le franchissement des seuils pour chaque métrique.
    
    Args:
        metrics_dict: dictionnaire des métriques calculées
        thresholds_dict: dictionnaire des seuils (depuis config)
    
    Returns:
        Dict[str, bool]: métrique -> dépassement True/False
    
    Note:
        Seuils initiaux théoriques, à ajuster après 5 runs de calibration
    """
    results = {}
    
    # Mapping des métriques aux seuils et conditions
    threshold_checks = {
        'variance_d2S': ('variance_d2S', lambda x, t: x > t),
        'max_median_ratio': ('stability_ratio', lambda x, t: x > t),
        't_retour': ('resilience', lambda x, t: x > t),
        'entropy_S': ('entropy_S', lambda x, t: x < t),
        'mean_high_effort': ('mean_high_effort', lambda x, t: x > t),
        'd_effort_dt': ('d_effort_dt', lambda x, t: x > t),
        'mean_abs_error': ('regulation_threshold', lambda x, t: x > t)
    }
    
    for metric_name, (threshold_key, check_func) in threshold_checks.items():
        if metric_name in metrics_dict and threshold_key in thresholds_dict:
            value = metrics_dict[metric_name]
            threshold = thresholds_dict[threshold_key]
            results[metric_name] = check_func(value, threshold)
        else:
            results[metric_name] = False
    
    return deep_convert(results)


# ============== EXPORT ET LOGGING ==============

def log_metrics(t: float, metrics_dict: Dict[str, Any], csv_writer: Any, 
                hdf5_file: Optional[h5py.File] = None) -> None:
    """
    Exporte les métriques dans les fichiers de log.
    
    Args:
        t: temps actuel
        metrics_dict: toutes les métriques à logger
        csv_writer: writer CSV (depuis simulate.py)
        hdf5_file: fichier HDF5 optionnel pour gros volumes
    
    Note:
        L'ordre des colonnes est défini dans config['system']['logging']['log_metrics']
    """
    # Pour CSV : on suppose que simulate.py gère déjà l'écriture
    # Cette fonction est un placeholder pour extensions futures
    
    # Si HDF5 est fourni (pour N > 10 ou T > 1000)
    if hdf5_file is not None:
        try:
            # Créer un groupe pour ce pas de temps
            time_group = hdf5_file.create_group(f"t_{int(t*1000)}")
            
            # Sauvegarder chaque métrique
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    time_group.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    time_group.create_dataset(key, data=value)
                elif isinstance(value, str):
                    time_group.attrs[key] = value
                    
        except Exception as e:
            warnings.warn(f"Erreur HDF5 à t={t}: {e}")


def summarize_metrics(metrics_history: Union[Dict[str, List], List[Dict]]) -> Dict[str, float]:
    """
    Calcule un résumé statistique des métriques sur tout le run.
    
    Args:
        metrics_history: historique complet des métriques
    
    Returns:
        Dict[str, float]: statistiques résumées
    """
    summary = {}
    
    # Convertir en format uniforme si nécessaire
    if isinstance(metrics_history, list) and len(metrics_history) > 0:
        # Liste de dicts -> dict de listes
        keys = metrics_history[0].keys()
        history_dict = {k: [m.get(k, 0) for m in metrics_history] for k in keys}
    else:
        history_dict = metrics_history
    
    # Calculer les statistiques pour chaque métrique numérique
    for key, values in history_dict.items():
        if len(values) > 0 and isinstance(values[0], (int, float)):
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_min"] = np.min(values)
            summary[f"{key}_max"] = np.max(values)
            summary[f"{key}_final"] = values[-1]
    
    return deep_convert(summary)


# ============== FONCTIONS SPÉCIALISÉES ==============

def detect_chaos_events(S_history: List[float], threshold_sigma: float = 3.0) -> List[Dict]:
    """
    Détecte les événements chaotiques dans le signal.
    
    Un événement chaotique est défini comme une déviation > threshold_sigma * σ.
    
    Args:
        S_history: historique du signal
        threshold_sigma: seuil en nombre d'écarts-types
    
    Returns:
        List[Dict]: liste des événements détectés
    """
    if len(S_history) < 100:
        return []
    
    events = []
    S_array = np.array(S_history)
    
    # Statistiques de référence sur une fenêtre glissante
    window_size = 50
    
    for i in range(window_size, len(S_array)):
        # Fenêtre de référence
        window = S_array[i-window_size:i]
        mean_window = np.mean(window)
        std_window = np.std(window)
        
        # Vérifier la valeur actuelle
        if std_window > 0:
            z_score = abs(S_array[i] - mean_window) / std_window
            
            if z_score > threshold_sigma:
                events.append({
                    'time_index': i,
                    'value': S_array[i],
                    'z_score': z_score,
                    'type': 'chaos'
                })
    
    return deep_convert(events)

def compute_correlation_effort_cpu(effort_history: List[float], 
                                   cpu_history: List[float]) -> float:
    """
    Calcule la corrélation entre l'effort et le coût CPU.
    
    Permet de détecter si l'effort interne se traduit en charge computationnelle.
    
    Args:
        effort_history: historique des efforts
        cpu_history: historique des temps CPU
    
    Returns:
        float: coefficient de corrélation [-1, 1]
    """
    if len(effort_history) < 10 or len(cpu_history) < 10:
        return 0.0
    
    # Aligner les longueurs
    min_len = min(len(effort_history), len(cpu_history))
    effort_aligned = effort_history[-min_len:]
    cpu_aligned = cpu_history[-min_len:]
    
    # Corrélation de Pearson
    correlation_matrix = np.corrcoef(effort_aligned, cpu_aligned)
    
    if correlation_matrix.shape == (2, 2):
        return correlation_matrix[0, 1]
    else:
        return 0.0


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module metrics.py
    """
    print("=== Tests du module metrics.py ===\n")
    
    # Test 1: CPU step
    print("Test 1 - CPU step:")
    start = time.perf_counter()
    time.sleep(0.1)  # Simuler du travail
    end = time.perf_counter()
    cpu = compute_cpu_step(start, end, 10)
    print(f"  CPU par strate: {cpu:.4f} secondes")
    
    # Test 2: Effort
    print("\nTest 2 - Effort:")
    delta_A = np.array([0.1, -0.05, 0.02])
    delta_f = np.array([0.01, 0.02, -0.01])
    delta_gamma = np.array([0.0, 0.0, 0.0])
    effort = compute_effort(delta_A, delta_f, delta_gamma, 1.0, 1.0, 1.0)
    print(f"  Effort total: {effort:.4f}")
    
    # Test 3: Variance d²S/dt²
    print("\nTest 3 - Fluidité:")
    # Signal sinusoïdal lisse
    t = np.linspace(0, 10, 100)
    S_smooth = np.sin(t)
    S_noisy = np.sin(t) + 0.1 * np.random.randn(100)
    
    var_smooth = compute_variance_d2S(S_smooth.tolist(), 0.1)
    var_noisy = compute_variance_d2S(S_noisy.tolist(), 0.1)
    print(f"  Variance lisse: {var_smooth:.6f}")
    print(f"  Variance bruitée: {var_noisy:.6f}")
    
    # Test 4: Entropie spectrale
    print("\nTest 4 - Entropie spectrale:")
    # Signal mono-fréquence vs multi-fréquence
    S_mono = np.sin(2 * np.pi * t)
    S_multi = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t) + 0.3 * np.sin(10 * np.pi * t)
    
    entropy_mono = compute_entropy_S(S_mono, 10.0)
    entropy_multi = compute_entropy_S(S_multi, 10.0)
    print(f"  Entropie mono-fréquence: {entropy_mono:.4f}")
    print(f"  Entropie multi-fréquence: {entropy_multi:.4f}")
    
    # Test 5: Temps de retour
    print("\nTest 5 - Résilience:")
    # Signal avec perturbation
    S_perturbed = np.ones(100)
    S_perturbed[50:55] = 5.0  # Perturbation
    S_perturbed[55:] = 1.0 + 0.1 * np.exp(-0.1 * np.arange(45))  # Retour progressif
    
    t_ret = compute_t_retour(S_perturbed.tolist(), 50, 0.1, 0.95)
    print(f"  Temps de retour: {t_ret:.2f}")
    
    # Test 6: Vérification des seuils
    print("\nTest 6 - Vérification seuils:")
    metrics = {
        'variance_d2S': 0.02,
        'entropy_S': 0.3,
        'mean_high_effort': 2.5
    }
    thresholds = {
        'variance_d2S': 0.01,
        'entropy_S': 0.5,
        'mean_high_effort': 2.0
    }
    
    checks = check_thresholds(metrics, thresholds)
    for metric, exceeded in checks.items():
        print(f"  {metric}: {'DÉPASSÉ' if exceeded else 'OK'}")
    
    print("\n✅ Module metrics.py prêt pour quantifier l'harmonie!")


# ============== MÉTRIQUES GLOBALES ET SCORES ==============

def calculate_all_scores(recent_history: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Calcule les scores normalisés (1-5) pour toutes les métriques FPS.
    Utilisé par compute_gamma_adaptive_aware pour évaluer la performance système.
    
    Args:
        recent_history: historique récent des métriques
        
    Returns:
        dict avec scores normalisés pour chaque métrique
    """
    if len(recent_history) < 5:
        # Pas assez d'historique, scores neutres
        return {
            'current': {
                'stability': 3,
                'regulation': 3,
                'fluidity': 3,
                'resilience': 3,
                'innovation': 3,
                'cpu_cost': 3,
                'effort': 3
            }
        }
    
    # Extraire les métriques récentes
    recent_S = [h.get('S(t)', 0) for h in recent_history]
    recent_errors = [h.get('mean_abs_error', 0) for h in recent_history]
    recent_efforts = [h.get('effort(t)', 0) for h in recent_history]
    recent_cpu = [h.get('cpu_step(t)', 0) for h in recent_history]
    recent_entropy = [h.get('entropy_S', 0.5) for h in recent_history]
    recent_fluidity = [h.get('fluidity', 1.0) for h in recent_history]
    recent_C = [h.get('C(t)', 1.0) for h in recent_history]
    
    # NOUVEAU: Utiliser adaptive_resilience si disponible
    recent_adaptive_resilience = [h.get('adaptive_resilience', None) for h in recent_history]
    recent_continuous_resilience = [h.get('continuous_resilience', None) for h in recent_history]
    
    # Calculer les scores (1-5)
    scores = {}
    
    # Stabilité : basée sur std(S) et variations de C(t)
    std_S = np.std(recent_S) if recent_S else 1.0
    stability_score = 5 if std_S < 0.5 else 4 if std_S < 1.0 else 3 if std_S < 2.0 else 2 if std_S < 3.0 else 1
    scores['stability'] = stability_score
    
    # Régulation : basée sur l'erreur moyenne
    mean_error = np.mean(recent_errors) if recent_errors else 1.0
    regulation_score = 5 if mean_error < 0.1 else 4 if mean_error < 0.3 else 3 if mean_error < 0.5 else 2 if mean_error < 1.0 else 1
    scores['regulation'] = regulation_score
    
    # Fluidité : directement depuis la métrique
    mean_fluidity = np.mean(recent_fluidity) if recent_fluidity else 0.5
    fluidity_score = 5 if mean_fluidity > 0.9 else 4 if mean_fluidity > 0.7 else 3 if mean_fluidity > 0.5 else 2 if mean_fluidity > 0.3 else 1
    scores['fluidity'] = fluidity_score
    
    # Résilience : utiliser adaptive_resilience en priorité
    resilience_score = 3  # Par défaut
    
    # D'abord essayer adaptive_resilience
    valid_adaptive_resilience = [r for r in recent_adaptive_resilience if r is not None]
    if valid_adaptive_resilience:
        mean_adaptive_resilience = np.mean(valid_adaptive_resilience)
        # adaptive_resilience est déjà normalisé [0-1]
        if mean_adaptive_resilience >= 0.90:
            resilience_score = 5
        elif mean_adaptive_resilience >= 0.75:
            resilience_score = 4
        elif mean_adaptive_resilience >= 0.60:
            resilience_score = 3
        elif mean_adaptive_resilience >= 0.40:
            resilience_score = 2
        else:
            resilience_score = 1
    # Sinon essayer continuous_resilience
    elif recent_continuous_resilience:
        valid_continuous_resilience = [r for r in recent_continuous_resilience if r is not None]
        if valid_continuous_resilience:
            mean_continuous_resilience = np.mean(valid_continuous_resilience)
            if mean_continuous_resilience >= 0.90:
                resilience_score = 5
            elif mean_continuous_resilience >= 0.75:
                resilience_score = 4
            elif mean_continuous_resilience >= 0.60:
                resilience_score = 3
            elif mean_continuous_resilience >= 0.40:
                resilience_score = 2
            else:
                resilience_score = 1
    # En dernier recours, utiliser C(t) comme proxy (ancienne méthode)
    else:
        C_recovery = np.mean(recent_C[-5:]) if len(recent_C) >= 5 else 0.5
        resilience_score = 5 if C_recovery > 0.9 else 4 if C_recovery > 0.7 else 3 if C_recovery > 0.5 else 2 if C_recovery > 0.3 else 1
    
    scores['resilience'] = resilience_score
    
    # Innovation : basée sur l'entropie
    mean_entropy = np.mean(recent_entropy) if recent_entropy else 0.5
    innovation_score = 5 if mean_entropy > 0.8 else 4 if mean_entropy > 0.6 else 3 if mean_entropy > 0.4 else 2 if mean_entropy > 0.2 else 1
    scores['innovation'] = innovation_score
    
    # Coût CPU
    mean_cpu = np.mean(recent_cpu) if recent_cpu else 0.01
    cpu_score = 5 if mean_cpu < 0.001 else 4 if mean_cpu < 0.01 else 3 if mean_cpu < 0.1 else 2 if mean_cpu < 1.0 else 1
    scores['cpu_cost'] = cpu_score
    
    # Effort interne
    mean_effort = np.mean(recent_efforts) if recent_efforts else 1.0
    effort_score = 5 if mean_effort < 0.5 else 4 if mean_effort < 1.0 else 3 if mean_effort < 2.0 else 2 if mean_effort < 3.0 else 1
    scores['effort'] = effort_score
    
    return {'current': scores}


# ============== EXPORT DES MÉTRIQUES ==============
