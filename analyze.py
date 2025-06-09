"""
analyze.py - Analyse et raffinement adaptatif FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Le processus de raffinements/ajustements de seuils, critères, et
dynamiques doit être itératif et réversible.
---------------------------------------------------------------

Ce module incarne l'apprentissage et l'évolution du système FPS :
- Analyse des runs par batch (typiquement 5)
- Détection des franchissements de seuils
- Raffinement automatique des paramètres
- Traçabilité complète via changelog
- Corrélations et analyses croisées

L'analyse est le miroir qui permet au système de se voir,
se comprendre et s'améliorer continuellement.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import json
import numpy as np
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
from utils import deep_convert

# Import des modules FPS pour cohérence
try:
    import metrics
    import validate_config
except ImportError:
    warnings.warn("Modules metrics ou validate_config non trouvés. Mode standalone.")


# ============== ANALYSE DE BATCH ==============

def analyze_criteria_and_refine(logs_batch: List[str], config: Dict) -> Dict[str, Any]:
    """
    Analyse un batch de runs et déclenche les raffinements nécessaires.
    
    Processus:
    1. Charge les logs de chaque run du batch
    2. Calcule les statistiques pour chaque critère
    3. Vérifie les franchissements de seuils
    4. Déclenche les raffinements si >50% des runs dépassent
    5. Met à jour la config et log les changements
    
    Args:
        logs_batch: liste des chemins vers les fichiers CSV des runs
        config: configuration actuelle
    
    Returns:
        Dict contenant:
        - 'refinements': liste des raffinements effectués
        - 'statistics': statistiques du batch
        - 'updated_config': config mise à jour
    """
    print(f"\n=== Analyse de batch: {len(logs_batch)} runs ===")
    
    # Validation des entrées
    if not isinstance(logs_batch, list):
        print(f"❌ logs_batch doit être une liste, reçu: {type(logs_batch)}")
        return deep_convert({'refinements': [], 'statistics': {}, 'updated_config': config})
    
    # Charger les données de chaque run
    batch_data = []
    for log_path in logs_batch:
        # Vérifier que c'est bien un chemin de fichier
        if not isinstance(log_path, str) or len(log_path) < 3:
            print(f"⚠️ Chemin invalide ignoré: {log_path}")
            continue
            
        try:
            if not os.path.exists(log_path):
                print(f"⚠️ Fichier non trouvé: {log_path}")
                continue
                
            run_data = load_run_data(log_path)
            batch_data.append(run_data)
            print(f"✓ Chargé: {os.path.basename(log_path)}")
        except Exception as e:
            print(f"✗ Erreur chargement {log_path}: {e}")
    
    if len(batch_data) == 0:
        return deep_convert({'refinements': [], 'statistics': {}, 'updated_config': config})
    
    # Analyser les critères
    criteria_stats = analyze_criteria_statistics(batch_data, config)
    
    # Déterminer les raffinements nécessaires
    refinements_needed = determine_refinements(criteria_stats, config)
    
    # Appliquer les raffinements
    updated_config = config.copy()
    refinements_applied = []
    
    for criterion, stats in refinements_needed.items():
        print(f"\nCritère '{criterion}' déclenché sur {stats['trigger_rate']*100:.1f}% des runs")
        
        # Appeler la fonction de raffinement appropriée
        refinement_func = REFINEMENT_FUNCTIONS.get(criterion)
        if refinement_func:
            changes = refinement_func(updated_config, stats)
            if changes:
                refinements_applied.append({
                    'criterion': criterion,
                    'changes': changes,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Logger le raffinement, déterminer le dossier de sortie depuis le premier log
                output_dir = os.path.dirname(logs_batch[0]) if logs_batch else "logs"
                changelog_path = os.path.join(output_dir, "changelog.txt")
                
                log_refinement(
                    changelog_path,
                    datetime.now(),
                    f"batch_{len(logs_batch)}",
                    criterion,
                    changes.get('old_value'),
                    changes.get('new_value'),
                    f"Dépassement sur {stats['trigger_rate']*100:.1f}% des runs"
                )
    
    # Exporter le journal des seuils
    # Exporter le journal des seuils
    if refinements_applied:
        output_dir = os.path.dirname(logs_batch[0]) if logs_batch else "logs"
        journal_path = os.path.join(output_dir, "threshold_journal.json")
        export_threshold_journal(criteria_stats, journal_path)
    
    return deep_convert({
        'refinements': refinements_applied,
        'statistics': criteria_stats,
        'updated_config': updated_config
    })


# ============== CHARGEMENT DES DONNÉES ==============

def load_run_data(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Charge les données d'un run depuis un fichier CSV.
    
    Args:
        csv_path: chemin vers le fichier CSV
    
    Returns:
        Dict avec les colonnes comme clés et arrays comme valeurs
    """
    data = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                try:
                    # Convertir en float si possible
                    if value and value.lower() not in ['stable', 'transitoire', 'chronique']:
                        data[key].append(float(value))
                    else:
                        data[key].append(value)
                except ValueError:
                    data[key].append(value)
    
    # Convertir en arrays numpy pour les colonnes numériques
    for key in data:
        if data[key] and isinstance(data[key][0], (int, float)):
            data[key] = np.array(data[key])
    
    return dict(data)


# ============== ANALYSE DES CRITÈRES ==============

def analyze_criteria_statistics(batch_data: List[Dict], config: Dict) -> Dict[str, Dict]:
    """
    Calcule les statistiques pour chaque critère sur le batch.
    
    Returns:
        Dict[criterion_name, statistics_dict]
    """
    stats = {}
    thresholds = config.get('to_calibrate', {})
    
    # Critères et leurs métriques associées
    criteria_metrics = {
        'fluidity': {
            'metric': 'variance_d2S',
            'threshold_key': 'variance_d2S',
            'condition': lambda x, t: x > t,
            'threshold_percent': 0.7  # 70% du run
        },
        'stability': {
            'metric': 'max_median_ratio',
            'threshold_key': 'stability_ratio',
            'condition': lambda x, t: x > t,
            'threshold_percent': 0.05  # 5% du run
        },
        'resilience': {
            'metric': 't_retour',
            'threshold_key': 'resilience',
            'condition': lambda x, t: x > t,
            'threshold_percent': None  # Valeur unique
        },
        'innovation': {
            'metric': 'entropy_S',
            'threshold_key': 'entropy_S',
            'condition': lambda x, t: x < t,
            'threshold_percent': 0.7
        },
        'regulation': {
            'metric': 'mean_abs_error',
            'threshold_key': 'mean_high_effort',  # Utilise ce seuil pour la régulation
            'condition': lambda x, t: x > 2 * t,  # 2x la médiane
            'threshold_percent': 0.5
        },
        'cpu_cost': {
            'metric': 'cpu_step(t)',
            'threshold_key': 'cpu_step_ctrl',
            'condition': lambda x, t: x > t,
            'threshold_percent': 0.8
        }
    }
    
    # Analyser chaque critère
    for criterion, metric_info in criteria_metrics.items():
        metric_name = metric_info['metric']
        threshold_key = metric_info['threshold_key']
        condition = metric_info['condition']
        threshold_percent = metric_info['threshold_percent']
        
        if threshold_key not in thresholds:
            continue
        
        threshold_value = thresholds[threshold_key]
        
        # Collecter les statistiques pour ce critère
        criterion_stats = {
            'threshold': threshold_value,
            'runs_triggered': 0,
            'values_per_run': [],
            'mean_values': [],
            'max_values': [],
            'trigger_rate': 0.0
        }
        
        # Analyser chaque run
        for run_data in batch_data:
            if metric_name not in run_data:
                continue
            
            values = run_data[metric_name]
            if len(values) == 0:
                continue
            
            # Calculer le pourcentage de dépassement
            if threshold_percent is not None:
                # Pourcentage du temps où le seuil est franchi
                triggers = [condition(v, threshold_value) for v in values if not np.isnan(v)]
                trigger_rate = sum(triggers) / len(triggers) if triggers else 0
                
                if trigger_rate >= threshold_percent:
                    criterion_stats['runs_triggered'] += 1
            else:
                # Valeur unique (comme t_retour)
                final_value = values[-1] if len(values) > 0 else 0
                if condition(final_value, threshold_value):
                    criterion_stats['runs_triggered'] += 1
            
            # Statistiques
            criterion_stats['values_per_run'].append(values)
            criterion_stats['mean_values'].append(np.mean(values))
            criterion_stats['max_values'].append(np.max(np.abs(values)))
        
        # Taux de déclenchement sur le batch
        if len(batch_data) > 0:
            criterion_stats['trigger_rate'] = criterion_stats['runs_triggered'] / len(batch_data)
        
        stats[criterion] = criterion_stats
    
    # Ajouter les critères d'effort
    stats.update(analyze_effort_criteria(batch_data, config))
    
    return stats


def analyze_effort_criteria(batch_data: List[Dict], config: Dict) -> Dict[str, Dict]:
    """
    Analyse spécifique pour les critères d'effort (chronique et transitoire).
    """
    stats = {}
    thresholds = config.get('to_calibrate', {})
    
    # Effort chronique
    if 'mean_high_effort' in thresholds:
        chronique_stats = {
            'threshold': thresholds['mean_high_effort'],
            'runs_triggered': 0,
            'values_per_run': [],
            'trigger_rate': 0.0
        }
        
        for run_data in batch_data:
            if 'mean_high_effort' in run_data:
                values = run_data['mean_high_effort']
                # Vérifier si > 2x médiane sur >80% du run
                if len(values) > 0:
                    median = np.median(values)
                    high_count = sum(1 for v in values if v > 2 * median)
                    if high_count / len(values) > 0.8:
                        chronique_stats['runs_triggered'] += 1
                chronique_stats['values_per_run'].append(values)
        
        if len(batch_data) > 0:
            chronique_stats['trigger_rate'] = chronique_stats['runs_triggered'] / len(batch_data)
        
        stats['effort_internal'] = chronique_stats
    
    # Effort transitoire
    if 'd_effort_dt' in thresholds:
        transitoire_stats = {
            'threshold': thresholds['d_effort_dt'],
            'runs_triggered': 0,
            'spike_counts': [],
            'trigger_rate': 0.0
        }
        
        for run_data in batch_data:
            if 'd_effort_dt' in run_data:
                values = run_data['d_effort_dt']
                if len(values) > 10:
                    std = np.std(values)
                    # Compter les pics > 5σ
                    spikes = sum(1 for v in values if abs(v) > 5 * std)
                    if spikes > 10:  # Plus de 10 pics
                        transitoire_stats['runs_triggered'] += 1
                    transitoire_stats['spike_counts'].append(spikes)
        
        if len(batch_data) > 0:
            transitoire_stats['trigger_rate'] = transitoire_stats['runs_triggered'] / len(batch_data)
        
        stats['effort_transient'] = transitoire_stats
    
    return stats


# ============== DÉTERMINATION DES RAFFINEMENTS ==============

def determine_refinements(criteria_stats: Dict[str, Dict], config: Dict) -> Dict[str, Dict]:
    """
    Détermine quels critères nécessitent un raffinement.
    Règle: si un critère dépasse son seuil sur >50% des runs du batch.
    """
    refinements_needed = {}
    
    for criterion, stats in criteria_stats.items():
        if stats.get('trigger_rate', 0) > 0.5:
            refinements_needed[criterion] = stats
    
    return refinements_needed

# ============== FONCTIONS DE RAFFINEMENT ==============

def refine_fluidity(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine les paramètres pour améliorer la fluidité.
    Actions: ajuster γₙ(t), envₙ(x,t)
    
    NOTE FPS: Conformément à la feuille de route V1.3, 
    le raffinement de fluidité agit sur la latence expressive
    et l'enveloppe adaptative pour adoucir les transitions.
    """
    changes = {}
    
    # Passer en mode dynamique pour la latence
    if config.get('latence', {}).get('gamma_n_mode') == 'static':
        old_mode = config['latence']['gamma_n_mode']
        config['latence']['gamma_n_mode'] = 'dynamic'
        changes['gamma_n_mode'] = {'old': old_mode, 'new': 'dynamic'}
        print("  → Latence γₙ: static → dynamic")
        
        # Mettre à jour les paramètres dynamiques selon la feuille de route
        config['latence']['gamma_n_dynamic'] = {
            'k_n': 2.0,  # Valeur de la feuille de route
            't0_n': config.get('system', {}).get('T', 100) / 2
        }
    
    # Ajuster les paramètres de l'enveloppe
    if config.get('enveloppe', {}).get('env_mode') == 'static':
        old_mode = config['enveloppe']['env_mode']
        config['enveloppe']['env_mode'] = 'dynamic'
        changes['env_mode'] = {'old': old_mode, 'new': 'dynamic'}
        print("  → Enveloppe: static → dynamic")
        
        # Paramètres dynamiques selon la feuille de route
        T = config.get('system', {}).get('T', 100)
        config['enveloppe']['sigma_n_dynamic'] = {
            'amp': 0.05,
            'freq': 1.0,
            'offset': 0.1,
            'T': T
        }
    
    # Augmenter sigma_n pour adoucir les transitions
    old_sigma = config['enveloppe'].get('sigma_n_static', 0.1)
    new_sigma = min(old_sigma * config.get('refinement_factors', {}).get('sigma_increase', 1.3), 0.3)
    config['enveloppe']['sigma_n_static'] = new_sigma
    changes['sigma_n'] = {'old': old_sigma, 'new': new_sigma}
    print(f"  → σₙ: {old_sigma:.3f} → {new_sigma:.3f}")
    
    return changes

def refine_stability(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine pour améliorer la stabilité.
    Actions: ajuster σ(x), αₙ
    
    NOTE FPS: Les facteurs de raffinement proviennent de
    config['refinement_factors'] conformément à la feuille de route.
    """
    changes = {}
    
    # Facteurs depuis la config
    k_reduction = config.get('refinement_factors', {}).get('k_reduction', 0.8)
    alpha_reduction = config.get('refinement_factors', {}).get('alpha_reduction', 0.7)
    
    # Réduire la sensibilité k pour toutes les strates
    for i, strate in enumerate(config.get('strates', [])):
        old_k = strate.get('k', 2.0)
        new_k = max(old_k * k_reduction, 1.0)  # Min 1.0 selon feuille de route
        strate['k'] = new_k
        if i == 0:  # Logger seulement la première
            changes['k'] = {'old': old_k, 'new': new_k}
            print(f"  → k (sensibilité): {old_k:.2f} → {new_k:.2f}")
    
    # Réduire alpha (souplesse d'adaptation)
    for i, strate in enumerate(config.get('strates', [])):
        old_alpha = strate.get('alpha', 0.5)
        new_alpha = max(old_alpha * alpha_reduction, 0.1)  # Min 0.1
        strate['alpha'] = new_alpha
        if i == 0:
            changes['alpha'] = {'old': old_alpha, 'new': new_alpha}
            print(f"  → αₙ: {old_alpha:.2f} → {new_alpha:.2f}")
    
    return changes


def refine_resilience(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine pour améliorer la résilience.
    Actions: ajuster αₙ, βₙ
    """
    changes = {}
    
    # Augmenter alpha pour plus de réactivité
    for i, strate in enumerate(config.get('strates', [])):
        old_alpha = strate.get('alpha', 0.5)
        new_alpha = min(old_alpha * 1.3, 1.0)
        strate['alpha'] = new_alpha
        if i == 0:
            changes['alpha'] = {'old': old_alpha, 'new': new_alpha}
            print(f"  → αₙ: {old_alpha:.2f} → {new_alpha:.2f}")
    
    # Augmenter beta pour un feedback plus fort
    for i, strate in enumerate(config.get('strates', [])):
        old_beta = strate.get('beta', 1.0)
        new_beta = min(old_beta * 1.2, 2.0)
        strate['beta'] = new_beta
        if i == 0:
            changes['beta'] = {'old': old_beta, 'new': new_beta}
            print(f"  → βₙ: {old_beta:.2f} → {new_beta:.2f}")
    
    return changes


def refine_innovation(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine pour améliorer l'innovation.
    Actions: ajuster θ(t), η(t), μₙ(t)
    
    NOTE FPS: Selon la feuille de route, θ(t) et η(t) sont
    des paramètres exploratoires à définir en phase 2.
    Pour la phase 1, on agit sur epsilon et dynamic_G.
    """
    changes = {}
    
    # Facteur depuis la config
    epsilon_increase = config.get('refinement_factors', {}).get('epsilon_increase', 1.5)
    
    # Augmenter la variation spiralée
    old_epsilon = config.get('spiral', {}).get('epsilon', 0.05)
    new_epsilon = min(old_epsilon * epsilon_increase, 0.15)
    config['spiral']['epsilon'] = new_epsilon
    changes['epsilon'] = {'old': old_epsilon, 'new': new_epsilon}
    print(f"  → ε (variation spirale): {old_epsilon:.3f} → {new_epsilon:.3f}")
    
    # Activer la régulation dynamique
    if not config.get('regulation', {}).get('dynamic_G', False):
        config['regulation']['dynamic_G'] = True
        changes['dynamic_G'] = {'old': False, 'new': True}
        print("  → Régulation G: static → dynamic")
    
    # NOTE: μₙ(t), θ(t), η(t) seront raffinés en phase 2
    # selon les critères observés (voir feuille de route)
    
    return changes


def refine_regulation(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine la régulation.
    Actions: ajuster βₙ, G(x)
    """
    changes = {}
    
    # Augmenter beta pour toutes les strates
    for i, strate in enumerate(config.get('strates', [])):
        old_beta = strate.get('beta', 1.0)
        new_beta = min(old_beta * 1.3, 2.5)
        strate['beta'] = new_beta
        if i == 0:
            changes['beta'] = {'old': old_beta, 'new': new_beta}
            print(f"  → βₙ: {old_beta:.2f} → {new_beta:.2f}")
    
    # Changer l'archétype de régulation si nécessaire
    old_arch = config.get('regulation', {}).get('G_arch', 'tanh')
    if old_arch == 'tanh' and stats.get('mean_values', []):
        # Si les erreurs sont grandes, passer à resonance
        mean_error = np.mean(stats['mean_values'])
        if mean_error > 1.0:
            config['regulation']['G_arch'] = 'resonance'
            changes['G_arch'] = {'old': old_arch, 'new': 'resonance'}
            print("  → G archétype: tanh → resonance")
    
    return changes


def refine_cpu(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Optimise la complexité computationnelle.
    """
    changes = {}
    
    # Réduire le nombre de métriques loguées si nécessaire
    log_metrics = config.get('system', {}).get('logging', {}).get('log_metrics', [])
    if len(log_metrics) > 10:
        # Garder seulement les essentielles
        essential = ['t', 'S(t)', 'C(t)', 'effort(t)', 'cpu_step(t)', 'entropy_S']
        old_metrics = log_metrics.copy()
        config['system']['logging']['log_metrics'] = essential
        changes['log_metrics'] = {'old': len(old_metrics), 'new': len(essential)}
        print(f"  → Métriques loguées: {len(old_metrics)} → {len(essential)}")
    
    return changes


def refine_chronic_effort(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine pour réduire l'effort chronique.
    Actions: ajuster αₙ, μₙ(t), σₙ(t)
    """
    changes = {}
    
    # Réduire alpha pour moins d'adaptation constante
    for i, strate in enumerate(config.get('strates', [])):
        old_alpha = strate.get('alpha', 0.5)
        new_alpha = max(old_alpha * 0.6, 0.1)
        strate['alpha'] = new_alpha
        if i == 0:
            changes['alpha'] = {'old': old_alpha, 'new': new_alpha}
            print(f"  → αₙ: {old_alpha:.2f} → {new_alpha:.2f}")
    
    # Augmenter sigma pour des transitions plus douces
    old_sigma = config['enveloppe'].get('sigma_n_static', 0.1)
    new_sigma = min(old_sigma * 1.3, 0.3)
    config['enveloppe']['sigma_n_static'] = new_sigma
    changes['sigma_n'] = {'old': old_sigma, 'new': new_sigma}
    print(f"  → σₙ: {old_sigma:.3f} → {new_sigma:.3f}")
    
    return changes


def refine_transient_effort(config: Dict, stats: Dict) -> Dict[str, Any]:
    """
    Raffine pour réduire l'effort transitoire.
    Actions: ajuster w_{ni}
    
    NOTE FPS: La réduction des poids respecte la contrainte
    de conservation (somme nulle) et la diagonale nulle.
    """
    changes = {}
    
    # Facteur depuis la config
    weight_reduction = config.get('refinement_factors', {}).get('weight_reduction', 0.8)
    
    # Réduire les poids de connexion
    for i, strate in enumerate(config.get('strates', [])):
        old_w = strate.get('w', [])
        if old_w and len(old_w) > 0:
            # Réduire tous les poids
            new_w = [w * weight_reduction for w in old_w]
            
            # Forcer la diagonale à zéro (pas d'auto-connexion)
            if i < len(new_w):
                new_w[i] = 0.0
            
            # Assurer que la somme reste nulle (conservation du signal)
            w_sum = sum(new_w)
            if abs(w_sum) > 1e-6:
                # Redistribuer l'écart sur les poids non-diagonaux
                non_diag_indices = [j for j in range(len(new_w)) if j != i]
                if non_diag_indices:
                    correction = -w_sum / len(non_diag_indices)
                    for j in non_diag_indices:
                        new_w[j] += correction
            
            strate['w'] = new_w
            if i == 0:
                changes['w_scale'] = {'old': 1.0, 'new': weight_reduction}
                print(f"  → Poids w_{{ni}}: réduits à {weight_reduction*100:.0f}%")
    
    return changes


# Dictionnaire des fonctions de raffinement
REFINEMENT_FUNCTIONS = {
    'fluidity': refine_fluidity,
    'stability': refine_stability,
    'resilience': refine_resilience,
    'innovation': refine_innovation,
    'regulation': refine_regulation,
    'cpu_cost': refine_cpu,
    'effort_internal': refine_chronic_effort,
    'effort_transient': refine_transient_effort
}


# ============== LOGGING ET EXPORT ==============

def log_refinement(changelog_path: str, date: datetime, run_id: str, 
                   criterion: str, old_value: Any, new_value: Any, 
                   reason: str) -> None:
    """
    Log une modification dans le changelog.
    
    Format: [Date] | RunID | Métrique=valeur | Action | seed=
    """
    os.makedirs(os.path.dirname(changelog_path) if os.path.dirname(changelog_path) else ".", 
                exist_ok=True)
    
    timestamp = date.strftime("%Y-%m-%d %H:%M:%S")
    
    # Formater les valeurs
    if isinstance(old_value, dict):
        old_str = json.dumps(deep_convert(old_value), indent=None)
        new_str = json.dumps(deep_convert(new_value), indent=None)
    else:
        old_str = str(old_value)
        new_str = str(new_value)
    
    log_entry = f"[{timestamp}] | {run_id} | {criterion}: {old_str} → {new_str} | {reason}\n"
    
    with open(changelog_path, 'a') as f:
        f.write(log_entry)


def export_threshold_journal(threshold_history: Dict[str, Any], 
                             output_path: str) -> None:
    """
    Exporte l'historique des seuils dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                exist_ok=True)
    
    # Préparer les données pour l'export
    journal = {
        'timestamp': datetime.now().isoformat(),
        'thresholds': threshold_history,
        'version': '1.0'
    }
    
    with open(output_path, 'w') as f:
        json.dump(deep_convert(journal), f, indent=2)
    
    print(f"  📝 Journal des seuils exporté : {output_path}")


# ============== ANALYSES SPÉCIALISÉES ==============

def compute_correlation_effort_cpu(effort_history: List[float], 
                                   cpu_history: List[float]) -> float:
    """
    Calcule la corrélation entre effort et CPU.
    
    Utilise la fonction de metrics.py pour cohérence.
    """
    try:
        return metrics.compute_correlation_effort_cpu(effort_history, cpu_history)
    except:
        # Fallback si metrics.py non disponible
        if len(effort_history) < 10 or len(cpu_history) < 10:
            return 0.0
        
        min_len = min(len(effort_history), len(cpu_history))
        effort_aligned = effort_history[-min_len:]
        cpu_aligned = cpu_history[-min_len:]
        
        correlation_matrix = np.corrcoef(effort_aligned, cpu_aligned)
        return correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0


def analyze_cross_metrics(run_data: Dict) -> Dict[str, float]:
    """
    Analyse les corrélations croisées entre métriques.
    """
    correlations = {}
    
    # Paires de métriques intéressantes
    pairs = [
        ('effort(t)', 'cpu_step(t)'),
        ('entropy_S', 'variance_d2S'),
        ('mean_abs_error', 'effort(t)'),
        ('C(t)', 'S(t)')
    ]
    
    for metric1, metric2 in pairs:
        if metric1 in run_data and metric2 in run_data:
            data1 = run_data[metric1]
            data2 = run_data[metric2]
            
            if len(data1) > 10 and len(data2) > 10:
                min_len = min(len(data1), len(data2))
                corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                correlations[f"{metric1}_vs_{metric2}"] = corr
    
    return correlations


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module analyze.py
    """
    print("=== Tests du module analyze.py ===\n")
    
    # Test 1: Chargement config
    print("Test 1 - Chargement configuration:")
    test_config = {
        'strates': [
            {'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'w': [0, 0.1, -0.1]},
            {'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'w': [0.1, 0, -0.1]},
            {'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'w': [0.1, -0.1, 0]}
        ],
        'latence': {'gamma_n_mode': 'static'},
        'enveloppe': {'env_mode': 'static', 'sigma_n_static': 0.1},
        'regulation': {'G_arch': 'tanh', 'dynamic_G': False},
        'spiral': {'epsilon': 0.05},
        'to_calibrate': {
            'variance_d2S': 0.01,
            'stability_ratio': 10,
            'entropy_S': 0.5,
            'mean_high_effort': 2.0,
            'd_effort_dt': 5.0
        }
    }
    print("  ✓ Config test créée")
    
    # Test 2: Raffinements
    print("\nTest 2 - Fonctions de raffinement:")
    
    # Test raffinement fluidité
    test_stats = {'trigger_rate': 0.7, 'mean_values': [0.02, 0.03, 0.025]}
    changes = refine_fluidity(test_config.copy(), test_stats)
    print(f"  Fluidité: {len(changes)} changements")
    
    # Test raffinement stabilité
    changes = refine_stability(test_config.copy(), test_stats)
    print(f"  Stabilité: {len(changes)} changements")
    
    # Test raffinement innovation
    changes = refine_innovation(test_config.copy(), test_stats)
    print(f"  Innovation: {len(changes)} changements")
    
    # Test 3: Corrélation
    print("\nTest 3 - Corrélation effort/CPU:")
    effort_test = [0.5, 0.6, 0.8, 1.2, 1.5, 1.3, 1.1, 0.9, 0.7, 0.6]
    cpu_test = [0.01, 0.012, 0.015, 0.022, 0.025, 0.021, 0.018, 0.016, 0.013, 0.011]
    corr = compute_correlation_effort_cpu(effort_test, cpu_test)
    print(f"  Corrélation: {corr:.3f}")
    
    # Test 4: Log refinement
    print("\nTest 4 - Logging:")
    os.makedirs("logs", exist_ok=True)
    log_refinement(
        "logs/test_changelog.txt",
        datetime.now(),
        "test_run",
        "alpha",
        0.5,
        0.7,
        "Test de logging"
    )
    print("  ✓ Changelog test créé")
    
    print("\n✅ Module analyze.py prêt pour l'évolution adaptative")
