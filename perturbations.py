"""
perturbations.py - Gestion des perturbations et inputs contextuels
Version complète conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module gère tous les types de perturbations pour tester
la résilience et l'adaptabilité du système FPS :

- Choc : impulsion ponctuelle (stress soudain)
- Rampe : augmentation progressive (pression croissante)
- Sinus : oscillation périodique (environnement cyclique)
- Bruit : variation aléatoire (chaos ambiant)
- Combinaisons : séquences complexes de perturbations

Chaque perturbation raconte une histoire différente,
révélant comment la FPS danse avec l'adversité.

(c) 2025 Gepetto & Andréa Gadal & Claude (Anthropic) 🌀
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import json
import warnings
from dataclasses import dataclass
from enum import Enum
from utils import deep_convert


# ============== TYPES DE PERTURBATIONS ==============

class PerturbationType(Enum):
    """Énumération des types de perturbations disponibles."""
    NONE = "none"
    CHOC = "choc"
    RAMPE = "rampe"
    SINUS = "sinus"
    BRUIT = "bruit"
    COMPOSITE = "composite"  # Pour les séquences complexes


@dataclass
class PerturbationConfig:
    """Configuration structurée d'une perturbation."""
    type: str
    t0: float = 0.0
    amplitude: float = 1.0
    duration: Optional[float] = None
    freq: Optional[float] = None
    phase: Optional[float] = 0.0
    seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'PerturbationConfig':
        """Crée une configuration depuis un dictionnaire."""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


# ============== GÉNÉRATEURS DE PERTURBATIONS ==============

def generate_perturbation(t: float, config: Union[Dict, PerturbationConfig]) -> float:
    """
    Génère une perturbation selon la configuration.
    
    Args:
        t: temps actuel
        config: configuration de la perturbation
    
    Returns:
        float: valeur de la perturbation à l'instant t
    
    Types supportés:
        - "choc": impulsion à t0
        - "rampe": croissance linéaire
        - "sinus": oscillation périodique  
        - "bruit": variation aléatoire
        - "none": pas de perturbation
    """
    # Gestion des configurations vides ou invalides
    if isinstance(config, dict):
        if not config or 'type' not in config:
            return 0.0  # Pas de perturbation si configuration invalide
        config = PerturbationConfig.from_dict(config)
    
    pert_type = config.type.lower()
    
    if pert_type == PerturbationType.NONE.value:
        return 0.0
    
    elif pert_type == PerturbationType.CHOC.value:
        return generate_choc(t, config)
    
    elif pert_type == PerturbationType.RAMPE.value:
        return generate_rampe(t, config)
    
    elif pert_type == PerturbationType.SINUS.value:
        return generate_sinus(t, config)
    
    elif pert_type == PerturbationType.BRUIT.value:
        return generate_bruit(t, config)
    
    else:
        warnings.warn(f"Type de perturbation '{pert_type}' non reconnu. Retour à 0.")
        return 0.0


def generate_choc(t: float, config: PerturbationConfig) -> float:
    """
    Génère une perturbation de type choc (impulsion).
    
    Le choc peut avoir une durée configurable pour modéliser
    des impulsions brèves mais non instantanées.
    
    Args:
        t: temps actuel
        config: configuration
    
    Returns:
        float: amplitude si dans la fenêtre du choc, 0 sinon
    """
    # Durée du choc (par défaut très brève)
    duration = config.duration if config.duration else 0.1
    
    # Vérifier si on est dans la fenêtre du choc
    if config.t0 <= t < config.t0 + duration:
        # Option : profil du choc (rectangulaire par défaut)
        # On pourrait ajouter un profil gaussien ou triangulaire
        return config.amplitude
    else:
        return 0.0


def generate_rampe(t: float, config: PerturbationConfig) -> float:
    """
    Génère une perturbation de type rampe (croissance linéaire).
    
    La rampe peut être bornée ou non selon la durée spécifiée.
    
    Args:
        t: temps actuel
        config: configuration
    
    Returns:
        float: valeur de la rampe
    """
    if t < config.t0:
        return 0.0
    
    # Temps écoulé depuis le début
    elapsed = t - config.t0
    
    if config.duration is not None:
        # Rampe bornée
        if elapsed >= config.duration:
            return config.amplitude
        else:
            # Croissance linéaire
            return config.amplitude * (elapsed / config.duration)
    else:
        # Rampe non bornée (croissance infinie)
        # On peut ajouter un taux de croissance
        growth_rate = config.amplitude / 10.0  # Par défaut : amplitude/10 par unité de temps
        return growth_rate * elapsed


def generate_sinus(t: float, config: PerturbationConfig) -> float:
    """
    Génère une perturbation sinusoïdale.
    
    Permet de modéliser des environnements cycliques
    ou des influences périodiques.
    
    Args:
        t: temps actuel
        config: configuration
    
    Returns:
        float: valeur sinusoïdale
    """
    if t < config.t0:
        return 0.0
    
    # Fréquence par défaut
    freq = config.freq if config.freq is not None else 0.1
    
    # Phase initiale
    phase = config.phase if config.phase is not None else 0.0
    
    # Sinusoïde
    return config.amplitude * np.sin(2 * np.pi * freq * (t - config.t0) + phase)


def generate_bruit(t: float, config: PerturbationConfig) -> float:
    """
    Génère une perturbation de type bruit.
    
    Plusieurs types de bruit sont disponibles :
    - Uniforme : distribution uniforme
    - Gaussien : distribution normale
    - Rose/Brown : bruit coloré (à implémenter)
    
    Args:
        t: temps actuel
        config: configuration
    
    Returns:
        float: valeur aléatoire
    """
    # Seed pour reproductibilité si spécifiée — sur un RNG PRIVÉ (fix 14/07/2026) :
    # l'ancien np.random.seed(...) par pas réensemençait le flux GLOBAL, enchevêtrant
    # le tirage du bruit avec tous les autres consommateurs (dont le randn
    # d'exploration de gamma). Le RandomState privé produit des valeurs
    # bit-identiques à l'ancien comportement (même seed par t, premier tirage
    # du flux), sans plus jamais toucher au flux global.
    if config.seed is not None:
        rng = np.random.RandomState(int(config.seed + t * 1000) % 2**32)
    else:
        rng = np.random.RandomState(int(t * 1000) % 2**32)

    # Type de bruit (uniforme par défaut)
    # On pourrait étendre avec un paramètre noise_type
    return config.amplitude * rng.uniform(-1, 1)


# ============== APPLICATION DES PERTURBATIONS ==============

def apply_perturbation_to_In(In_array: np.ndarray, 
                             perturbation_value: float) -> np.ndarray:
    """
    Applique une perturbation aux inputs contextuels.
    
    Args:
        In_array: array des inputs pour chaque strate
        perturbation_value: valeur de la perturbation
    
    Returns:
        np.ndarray: inputs perturbés
    """
    # Application simple : addition
    # On pourrait avoir des modes plus sophistiqués
    return In_array + perturbation_value


def apply_perturbation_selective(In_array: np.ndarray, 
                                 perturbation_value: float,
                                 strata_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Applique une perturbation sélectivement à certaines strates.
    
    Args:
        In_array: array des inputs
        perturbation_value: valeur de la perturbation
        strata_mask: masque booléen pour sélectionner les strates
    
    Returns:
        np.ndarray: inputs perturbés sélectivement
    """
    result = In_array.copy()
    
    if strata_mask is not None:
        # Appliquer seulement aux strates sélectionnées
        result[strata_mask] += perturbation_value
    else:
        # Appliquer à toutes
        result += perturbation_value
    
    return result


# ============== SÉQUENCES DE PERTURBATIONS ==============

def generate_perturbation_sequence(T: float, dt: float, 
                                   perturbation_configs: List[Dict]) -> np.ndarray:
    """
    Génère une séquence complète de perturbations combinées.
    
    Permet de créer des scénarios complexes avec plusieurs
    perturbations qui se chevauchent ou se succèdent.
    
    Args:
        T: durée totale
        dt: pas de temps
        perturbation_configs: liste de configurations
    
    Returns:
        np.ndarray: séquence temporelle des perturbations
    """
    # Array temporel
    t_array = np.arange(0, T, dt)
    n_steps = len(t_array)
    
    # Initialiser la séquence
    sequence = np.zeros(n_steps)
    
    # Appliquer chaque perturbation
    for config in perturbation_configs:
        for i, t in enumerate(t_array):
            sequence[i] += generate_perturbation(t, config)
    
    return sequence


def create_scenario(scenario_name: str, T: float, base_amplitude: float = 1.0) -> List[Dict]:
    """
    Crée des scénarios de perturbations prédéfinis.
    
    Args:
        scenario_name: nom du scénario
        T: durée totale
        base_amplitude: amplitude de base
    
    Returns:
        List[Dict]: liste de configurations de perturbations
    
    Scénarios disponibles:
        - "stress_test": chocs répétés
        - "environnement_variable": sinus + bruit
        - "crise_recovery": choc fort puis rampe douce
        - "chaos": bruit intense avec pics aléatoires
    """
    scenarios = {
        "stress_test": [
            {"type": "choc", "t0": T*0.2, "amplitude": base_amplitude*2, "duration": 0.5},
            {"type": "choc", "t0": T*0.4, "amplitude": base_amplitude*1.5, "duration": 0.5},
            {"type": "choc", "t0": T*0.6, "amplitude": base_amplitude*2.5, "duration": 0.5},
            {"type": "choc", "t0": T*0.8, "amplitude": base_amplitude*3, "duration": 0.5}
        ],
        
        "environnement_variable": [
            {"type": "sinus", "t0": 0, "amplitude": base_amplitude*0.5, "freq": 0.1},
            {"type": "sinus", "t0": T*0.3, "amplitude": base_amplitude*0.3, "freq": 0.3, "phase": np.pi/4},
            {"type": "bruit", "t0": T*0.5, "amplitude": base_amplitude*0.2}
        ],
        
        "crise_recovery": [
            {"type": "choc", "t0": T*0.3, "amplitude": base_amplitude*5, "duration": 1.0},
            {"type": "rampe", "t0": T*0.5, "amplitude": -base_amplitude*2, "duration": T*0.3}
        ],
        
        "chaos": [
            {"type": "bruit", "t0": 0, "amplitude": base_amplitude*0.5},
            {"type": "choc", "t0": T*0.15, "amplitude": base_amplitude*3, "duration": 0.2},
            {"type": "choc", "t0": T*0.35, "amplitude": -base_amplitude*2, "duration": 0.3},
            {"type": "sinus", "t0": T*0.5, "amplitude": base_amplitude, "freq": 0.5},
            {"type": "choc", "t0": T*0.75, "amplitude": base_amplitude*4, "duration": 0.1}
        ]
    }
    
    if scenario_name in scenarios:
        return scenarios[scenario_name]
    else:
        warnings.warn(f"Scénario '{scenario_name}' non reconnu. Retour à une perturbation simple.")
        return [{"type": "choc", "t0": T/2, "amplitude": base_amplitude}]


# ============== ANALYSE DES PERTURBATIONS ==============

def analyze_perturbation_impact(S_history: np.ndarray, 
                                perturbation_sequence: np.ndarray,
                                dt: float) -> Dict[str, float]:
    """
    Analyse l'impact des perturbations sur le signal.
    
    Args:
        S_history: historique du signal S(t)
        perturbation_sequence: séquence des perturbations
        dt: pas de temps
    
    Returns:
        Dict avec métriques d'impact
    """
    # Corrélation perturbation-signal
    if len(S_history) == len(perturbation_sequence):
        correlation = np.corrcoef(S_history, perturbation_sequence)[0, 1]
    else:
        correlation = 0.0
    
    # Délai de réponse (cross-corrélation)
    if len(S_history) > 10 and len(perturbation_sequence) > 10:
        xcorr = np.correlate(S_history, perturbation_sequence, mode='same')
        delay_idx = np.argmax(np.abs(xcorr)) - len(xcorr)//2
        response_delay = delay_idx * dt
    else:
        response_delay = 0.0
    
    # Amplification/Atténuation
    pert_energy = np.sum(perturbation_sequence**2)
    signal_energy = np.sum(S_history**2)
    
    if pert_energy > 0:
        amplification = signal_energy / pert_energy
    else:
        amplification = 1.0
    
    # Persistance de l'effet
    # (combien de temps le signal reste perturbé après la fin de la perturbation)
    pert_end_idx = np.where(perturbation_sequence != 0)[0]
    if len(pert_end_idx) > 0:
        last_pert_idx = pert_end_idx[-1]
        if last_pert_idx < len(S_history) - 10:
            post_pert_std = np.std(S_history[last_pert_idx:])
            baseline_std = np.std(S_history[:min(100, last_pert_idx)])
            persistence = post_pert_std / (baseline_std + 1e-10)
        else:
            persistence = 1.0
    else:
        persistence = 0.0
    
    return deep_convert({
        'correlation': correlation,
        'response_delay': response_delay,
        'amplification': amplification,
        'persistence': persistence
    })


# ============== VISUALISATION DES PERTURBATIONS ==============

def plot_perturbation_profile(config: Union[Dict, PerturbationConfig], 
                              T: float, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère le profil temporel d'une perturbation pour visualisation.
    
    Args:
        config: configuration de la perturbation
        T: durée totale
        dt: pas de temps
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (temps, valeurs)
    """
    t_array = np.arange(0, T, dt)
    values = np.array([generate_perturbation(t, config) for t in t_array])
    
    return t_array, values


# ============== EXPORT/IMPORT DE SCÉNARIOS ==============

def save_scenario(scenario_configs: List[Dict], filename: str) -> None:
    """
    Sauvegarde un scénario de perturbations dans un fichier JSON.
    
    Args:
        scenario_configs: liste de configurations
        filename: nom du fichier
    """
    with open(filename, 'w') as f:
        json.dump(deep_convert({
            'version': '1.0',
            'timestamp': np.datetime64('now').astype(str),
            'perturbations': scenario_configs
        }), f, indent=2)
    
    print(f"Scénario sauvegardé : {filename}")


def load_scenario(filename: str) -> List[Dict]:
    """
    Charge un scénario depuis un fichier JSON.
    
    Args:
        filename: nom du fichier
    
    Returns:
        List[Dict]: liste de configurations
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data.get('perturbations', [])


# ============== NOUVELLE ARCHITECTURE In(t) ==============

def compute_In(t: float, input_cfg: Dict, state: Optional[Dict] = None, 
               history: Optional[List[Dict]] = None, dt: float = 0.05) -> float:
    """
    Calcule In(t) selon la nouvelle architecture modulaire.
    
    In(t) = Offset(t) + Gain(t) · tanh(In_raw(t) / scale)
    où In_raw(t) = Σᵢ wᵢ · Pertᵢ(t)
    
    Args:
        t: temps actuel
        input_cfg: configuration de l'input contextuel
        state: état du système (pour modes adaptatifs)
        history: historique pour calculs adaptatifs
        dt: pas de temps
    
    Returns:
        float: valeur de In(t) garantie > 0
    
    Structure de input_cfg:
        {
            "baseline": {
                "offset_mode": "static" | "adaptive",
                "offset": 0.1,  # pour mode static
                "offset_adaptive": {...},  # pour mode adaptive
                "gain_mode": "static" | "adaptive",
                "gain": 1.0,  # pour mode static
                "gain_adaptive": {...}  # pour mode adaptive
            },
            "scale": 1.0,
            "perturbations": [
                {"type": "sinus", "weight": 1.0, ...},
                {"type": "choc", "weight": 0.5, ...}
            ]
        }
    """
    # Configuration par défaut
    baseline = input_cfg.get('baseline', {})
    scale = input_cfg.get('scale', 1.0)
    perturbations = input_cfg.get('perturbations', [])
    
    # ----- 1. Calcul de l'Offset -----
    offset_mode = baseline.get('offset_mode', 'static')
    
    if offset_mode == 'static':
        offset = baseline.get('offset', 0.1)
    else:  # adaptive
        offset = compute_adaptive_offset(t, baseline.get('offset_adaptive', {}), 
                                         state, history, dt)
    
    # ----- 2. Calcul du Gain -----
    gain_mode = baseline.get('gain_mode', 'static')
    
    if gain_mode == 'static':
        gain = baseline.get('gain', 1.0)
    else:  # adaptive
        gain = compute_adaptive_gain(t, baseline.get('gain_adaptive', {}), 
                                     state, history, dt)
    
    # ----- 3. Superposition des perturbations -----
    In_raw = 0.0
    for pert_cfg in perturbations:
        weight = pert_cfg.get('weight', 1.0)
        pert_value = generate_perturbation(t, pert_cfg)
        In_raw += weight * pert_value
    
    # ----- 4. Normalisation douce avec tanh -----
    # tanh borne In_raw dans [-1, 1], donc In_final reste dans [offset-gain, offset+gain]
    In_normalized = np.tanh(In_raw / scale)
    In_final = offset + gain * In_normalized
    
    # Garantir In > 0
    epsilon = 0.001  # Valeur minimale absolue
    In_final = max(In_final, epsilon)
    
    return In_final


def compute_adaptive_offset(t: float, offset_cfg: Dict, state: Optional[Dict], 
                            history: Optional[List[Dict]], dt: float) -> float:
    """
    Calcule un offset adaptatif pour maintenir σ(In) dans une plage cible.
    
    Args:
        offset_cfg: configuration de l'offset adaptatif
            - "target_sigma": valeur cible pour σ(In) (défaut: 0.5)
            - "k_A": gain de l'intégrateur (défaut: 0.01)
            - "min": offset minimum (défaut: 0.05)
            - "max": offset maximum (défaut: 0.5)
            - "window": taille de fenêtre pour moyenner (défaut: 20)
    """
    # Paramètres
    target_sigma = offset_cfg.get('target_sigma', 0.5)
    k_A = offset_cfg.get('k_A', 0.01)
    offset_min = offset_cfg.get('min', 0.05)
    offset_max = offset_cfg.get('max', 0.5)
    window_size = offset_cfg.get('window', 20)
    
    # Récupérer l'offset actuel depuis l'état
    if state is None:
        state = {}
    
    if 'adaptive_offset' not in state:
        state['adaptive_offset'] = offset_cfg.get('initial', 0.1)
    
    current_offset = state['adaptive_offset']
    
    # Calculer σ moyen sur la fenêtre récente
    if history and len(history) >= window_size:
        # Récupérer les In récents et calculer leurs sigmoïdes
        recent_In = [h.get('In', 0.1) for h in history[-window_size:]]
        # Sigmoïde simple : 1/(1+exp(-x))
        recent_sigma = [1.0 / (1.0 + np.exp(-In)) for In in recent_In]
        mean_sigma = np.mean(recent_sigma)
        
        # Intégrateur : dA/dt = k_A * (target - actual)
        error = target_sigma - mean_sigma
        dA = k_A * error * dt
        
        # Mise à jour avec saturation
        new_offset = np.clip(current_offset + dA, offset_min, offset_max)
        state['adaptive_offset'] = new_offset
        
        return new_offset
    else:
        # Pas assez d'historique, retourner l'offset actuel
        return current_offset


def compute_adaptive_gain(t: float, gain_cfg: Dict, state: Optional[Dict], 
                          history: Optional[List[Dict]], dt: float) -> float:
    """
    Calcule un gain adaptatif pour éviter la saturation.
    
    Args:
        gain_cfg: configuration du gain adaptatif
            - "target_range": amplitude cible pour In_raw (défaut: 2.0)
            - "k_G": gain du contrôleur (défaut: 0.05)
            - "min": gain minimum (défaut: 0.1)
            - "max": gain maximum (défaut: 2.0)
            - "window": taille de fenêtre (défaut: 10)
    """
    # Paramètres
    target_range = gain_cfg.get('target_range', 2.0)
    k_G = gain_cfg.get('k_G', 0.05)
    gain_min = gain_cfg.get('min', 0.1)
    gain_max = gain_cfg.get('max', 2.0)
    window_size = gain_cfg.get('window', 10)
    
    # Récupérer le gain actuel depuis l'état
    if state is None:
        state = {}
    
    if 'adaptive_gain' not in state:
        state['adaptive_gain'] = gain_cfg.get('initial', 1.0)
    
    current_gain = state['adaptive_gain']
    
    # Calculer |In_raw| moyen récent
    if history and len(history) >= window_size:
        # Note: il faudrait stocker In_raw dans l'historique
        # Pour l'instant, on estime depuis les variations de In
        recent_In = [h.get('In', 0.1) for h in history[-window_size:]]
        In_variations = np.std(recent_In)
        
        # Si les variations sont trop grandes, réduire le gain
        error = In_variations - target_range
        dG = -k_G * error * dt
        
        # Mise à jour avec saturation
        new_gain = np.clip(current_gain + dG, gain_min, gain_max)
        state['adaptive_gain'] = new_gain
        
        return new_gain
    else:
        return current_gain


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module perturbations.py
    """
    print("=== Tests du module perturbations.py ===\n")
    
    # Test 1: Perturbations individuelles
    print("Test 1 - Types de perturbations:")
    
    test_configs = [
        {"type": "choc", "t0": 5.0, "amplitude": 2.0, "duration": 0.5},
        {"type": "rampe", "t0": 3.0, "amplitude": 1.5, "duration": 5.0},
        {"type": "sinus", "t0": 0.0, "amplitude": 1.0, "freq": 0.2},
        {"type": "bruit", "t0": 0.0, "amplitude": 0.5}
    ]
    
    for config in test_configs:
        print(f"\n  {config['type'].upper()}:")
        for t in [0, 3, 5, 5.5, 10]:
            value = generate_perturbation(t, config)
            print(f"    t={t}: {value:.3f}")
    
    # Test 2: Application aux inputs
    print("\nTest 2 - Application aux inputs:")
    In_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pert_value = 0.5
    
    In_perturbed = apply_perturbation_to_In(In_test, pert_value)
    print(f"  Input original: {In_test}")
    print(f"  Perturbation: {pert_value}")
    print(f"  Input perturbé: {In_perturbed}")
    
    # Test 3: Application sélective
    print("\nTest 3 - Perturbation sélective:")
    mask = np.array([True, False, True, False, True])
    In_selective = apply_perturbation_selective(In_test, pert_value, mask)
    print(f"  Masque: {mask}")
    print(f"  Résultat: {In_selective}")
    
    # Test 4: Séquence de perturbations
    print("\nTest 4 - Séquence combinée:")
    T = 20
    dt = 0.1
    
    sequence_configs = [
        {"type": "choc", "t0": 5.0, "amplitude": 2.0},
        {"type": "sinus", "t0": 0.0, "amplitude": 0.5, "freq": 0.5}
    ]
    
    sequence = generate_perturbation_sequence(T, dt, sequence_configs)
    print(f"  Longueur séquence: {len(sequence)}")
    print(f"  Min/Max: [{np.min(sequence):.3f}, {np.max(sequence):.3f}]")
    print(f"  Moyenne: {np.mean(sequence):.3f}")
    
    # Test 5: Scénarios prédéfinis
    print("\nTest 5 - Scénarios prédéfinis:")
    for scenario in ["stress_test", "environnement_variable", "chaos"]:
        configs = create_scenario(scenario, T=100, base_amplitude=1.0)
        print(f"  {scenario}: {len(configs)} perturbations")
    
    # Test 6: Analyse d'impact
    print("\nTest 6 - Analyse d'impact:")
    # Signal synthétique affecté par la perturbation
    t_array = np.arange(0, T, dt)
    S_test = np.sin(2 * np.pi * t_array / 5) + 0.5 * sequence
    
    impact = analyze_perturbation_impact(S_test, sequence, dt)
    print(f"  Corrélation: {impact['correlation']:.3f}")
    print(f"  Délai de réponse: {impact['response_delay']:.3f}")
    print(f"  Amplification: {impact['amplification']:.3f}")
    print(f"  Persistance: {impact['persistence']:.3f}")
    
    # Test 7: Visualisation
    print("\nTest 7 - Profils de perturbation:")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for ax, config in zip(axes.flat, test_configs):
        t_vis, values = plot_perturbation_profile(config, T=20, dt=0.1)
        ax.plot(t_vis, values, linewidth=2)
        ax.set_title(f"Perturbation {config['type']}")
        ax.set_xlabel('Temps')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_perturbations.png')
    print("  Graphiques sauvegardés : test_perturbations.png")
    
    print("\n✅ Module perturbations.py prêt à challenger la FPS")
