"""
kuramoto.py - Oscillateurs de Kuramoto (groupe contrôle)
Version complète conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module implémente le modèle canonique d'oscillateurs de Kuramoto
pour servir de comparaison avec la dynamique FPS.

Kuramoto = synchronisation par phase uniquement
FPS = phase + amplitude + feedback + spiralisation

L'équation de Kuramoto :
dφᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(φⱼ - φᵢ)

Ce contrôle permet de montrer que la FPS apporte :
- Stabilité adaptative (amplitude)
- Harmonie spiralée (accord de phase)
- Résilience (feedback)
- Innovation (émergence)

(c) 2025 Gepetto & Andréa Gadal & Claude (Anthropic) 🌀
"""

import numpy as np
import time
import csv
import os
from typing import Dict, List, Tuple, Optional, Any
import json
from utils import deep_convert


def kuramoto_step(phases: np.ndarray, frequencies: np.ndarray, 
                  K: float, N: int, dt: float) -> np.ndarray:
    """
    Calcule un pas de temps de la dynamique de Kuramoto.
    
    dφᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(φⱼ - φᵢ)
    
    Args:
        phases: phases actuelles des oscillateurs
        frequencies: fréquences naturelles ωᵢ
        K: force de couplage
        N: nombre d'oscillateurs
        dt: pas de temps
    
    Returns:
        np.ndarray: dérivées des phases dφ/dt
    """
    dphases_dt = frequencies.copy()
    
    # Calcul du couplage all-to-all
    for i in range(N):
        coupling_sum = 0.0
        for j in range(N):
            if i != j:  # Pas d'auto-couplage
                coupling_sum += np.sin(phases[j] - phases[i])
        
        # Ajout du terme de couplage
        dphases_dt[i] += (K / N) * coupling_sum
    
    return dphases_dt


def compute_kuramoto_order(phases: np.ndarray) -> Tuple[float, float]:
    """
    Calcule le paramètre d'ordre de Kuramoto.
    
    r·e^(iψ) = (1/N) · Σₙ e^(iφₙ)
    
    Args:
        phases: phases des oscillateurs
    
    Returns:
        Tuple[float, float]: (r, ψ) - amplitude et phase moyenne
    """
    # Représentation complexe
    complex_sum = np.mean(np.exp(1j * phases))
    
    # Amplitude (niveau de synchronisation)
    r = np.abs(complex_sum)
    
    # Phase moyenne
    psi = np.angle(complex_sum)
    
    return r, psi


def run_kuramoto_simulation(config: Dict, loggers: Dict) -> Dict[str, Any]:
    """
    Execute une simulation complète Kuramoto avec les mêmes paramètres que FPS.
    
    Args:
        config: configuration (même format que FPS)
        loggers: dictionnaire avec csv_writer, run_id, etc.
    
    Returns:
        Dict avec les résultats (même format que FPS pour comparaison)
    """
    print("\n=== Simulation Kuramoto (Groupe Contrôle) ===")
    
    # Extraction des paramètres
    N = config['system']['N']
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    seed = config['system']['seed']
    
    # Seed pour reproductibilité
    np.random.seed(seed)
    
    # Paramètres Kuramoto
    K = 0.5  # Force de couplage modérée
    
    # Initialisation
    phases = np.random.uniform(0, 2*np.pi, N)  # Phases aléatoires
    frequencies = np.random.uniform(0, 1, N)    # Fréquences naturelles ωᵢ ~ U[0,1]
    
    print(f"Configuration Kuramoto:")
    print(f"  - N = {N} oscillateurs")
    print(f"  - K = {K} (couplage)")
    print(f"  - T = {T}, dt = {dt}")
    print(f"  - Fréquences ωᵢ ~ U[0,1]")
    
    # Arrays temporels
    t_array = np.arange(0, T, dt)
    n_steps = len(t_array)
    
    # Historiques
    history = []
    S_history = []
    C_history = []
    cpu_steps = []
    order_params = []
    effort_history = []  # Toujours 0 pour Kuramoto (pas d'adaptation)
    
    # Perturbation (même que FPS si définie)
    perturbation = config['system'].get('perturbation', {})
    pert_type = perturbation.get('type', 'none')
    pert_t0 = perturbation.get('t0', T/2)
    pert_amplitude = perturbation.get('amplitude', 1.0)
    
    # Boucle de simulation
    for step, t in enumerate(t_array):
        step_start = time.perf_counter()
        
        # Application de la perturbation (sur les fréquences)
        freq_perturbed = frequencies.copy()
        if pert_type == 'choc' and abs(t - pert_t0) < dt:
            # Perturbation impulsionnelle
            freq_perturbed += pert_amplitude * np.random.randn(N)
        elif pert_type == 'rampe' and t >= pert_t0:
            # Perturbation croissante
            ramp_value = pert_amplitude * (t - pert_t0) / (T - pert_t0)
            freq_perturbed += ramp_value
        
        # Pas de Kuramoto
        dphases_dt = kuramoto_step(phases, freq_perturbed, K, N, dt)
        phases += dphases_dt * dt
        
        # Normaliser les phases dans [0, 2π]
        phases = phases % (2 * np.pi)
        
        # Calcul du paramètre d'ordre
        r, psi = compute_kuramoto_order(phases)
        order_params.append(r)
        
        # Signal global (somme des oscillateurs avec amplitude unitaire)
        S_t = np.sum(np.sin(phases))
        S_history.append(S_t)
        
        # Coefficient d'accord (cohérence des phases adjacentes)
        if N > 1:
            phase_diffs = [phases[(i+1)%N] - phases[i] for i in range(N)]
            C_t = np.mean(np.cos(phase_diffs))
        else:
            C_t = 1.0
        C_history.append(C_t)
        
        # Temps CPU
        cpu_step = (time.perf_counter() - step_start) / N
        cpu_steps.append(cpu_step)
        
        # Métriques compatibles FPS
        metrics_dict = {
            't': t,
            'S(t)': S_t,
            'C(t)': C_t,
            'E(t)': r,  # Paramètre d'ordre comme "amplitude"
            'L(t)': 0,  # Pas de latence dans Kuramoto
            'cpu_step(t)': cpu_step,
            'effort(t)': 0.0,  # Pas d'effort (pas d'adaptation)
            'A_mean(t)': 1.0,  # Amplitude fixe
            'f_mean(t)': np.mean(frequencies),
            'effort_status': 'stable',
            'variance_d2S': 0.0,  # À calculer si nécessaire
            'entropy_S': 0.0,     # À calculer si nécessaire
            'mean_abs_error': 0.0,  # Pas de régulation
            'mean_high_effort': 0.0,
            'd_effort_dt': 0.0,
            't_retour': 0.0,
            'max_median_ratio': 1.0,
            'continuous_resilience': 1.0  # Valeur par défaut
        }
        
        # Écrire dans le CSV
        if hasattr(loggers.get('csv_writer'), 'writerow'):
            # Extraire les valeurs dans l'ordre des colonnes de config
            log_metrics = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    if isinstance(value, str):
                        row_data.append(value)
                    else:
                        row_data.append(f"{value:.6f}")
                else:
                    row_data.append("0.0")
            
            loggers['csv_writer'].writerow(row_data)
        
        # Historique
        history.append({
            't': t,
            'S': S_t,
            'C': C_t,
            'order': r,
            'phases': phases.copy(),
            'frequencies': frequencies.copy()
        })
        
        # Affichage progression
        if step % (n_steps // 10) == 0:
            progress = (step / n_steps) * 100
            print(f"  Progression: {progress:.0f}% - r={r:.3f}, C(t)={C_t:.3f}")
    
    # Statistiques finales
    print(f"\n📊 Résultats Kuramoto:")
    print(f"  - Synchronisation finale: r = {order_params[-1]:.3f}")
    print(f"  - Cohérence moyenne: C̄ = {np.mean(C_history):.3f}")
    print(f"  - CPU moyen: {np.mean(cpu_steps)*1000:.3f} ms/step")
    print(f"  - Signal S(t): μ={np.mean(S_history):.3f}, σ={np.std(S_history):.3f}")
    
    # Calculer des métriques supplémentaires pour comparaison
    from scipy import signal as scipy_signal
    
    # Variance de d²S/dt²
    if len(S_history) >= 3:
        dS_dt = np.gradient(S_history, dt)
        d2S_dt2 = np.gradient(dS_dt, dt)
        variance_d2S = np.var(d2S_dt2)
    else:
        variance_d2S = 0.0
    
    # Entropie spectrale
    if len(S_history) >= 10:
        freqs, psd = scipy_signal.periodogram(S_history, 1/dt)
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm + 1e-15  # Éviter log(0)
        entropy_S = -np.sum(psd_norm * np.log(psd_norm))
        entropy_S = entropy_S / np.log(len(psd_norm))  # Normaliser
    else:
        entropy_S = 0.5
    
    # Temps de retour après perturbation
    if pert_type != 'none' and pert_t0 < T/2:
        t_choc_idx = int(pert_t0 / dt)
        pre_shock_mean = np.mean(order_params[max(0, t_choc_idx-50):t_choc_idx])
        
        # Chercher le retour à 95% de la valeur pré-choc
        t_retour = 0.0
        for i in range(t_choc_idx, len(order_params)):
            if abs(order_params[i] - pre_shock_mean) < 0.05 * pre_shock_mean:
                t_retour = (i - t_choc_idx) * dt
                break
    else:
        t_retour = 0.0
    
    # Résilience continue pour perturbations non-ponctuelles
    if pert_type in ['sinus', 'bruit', 'rampe'] and len(C_history) >= 20:
        # Import de la fonction si disponible
        try:
            from metrics import compute_continuous_resilience
            continuous_resilience = compute_continuous_resilience(
                C_history, S_history, perturbation_active=True
            )
        except:
            # Calcul simplifié si la fonction n'est pas disponible
            C_mean = np.mean(C_history[-100:]) if len(C_history) >= 100 else np.mean(C_history)
            C_std = np.std(C_history[-100:]) if len(C_history) >= 100 else np.std(C_history)
            stability_score = C_mean / (1 + C_std) if C_mean > 0 else 0.0
            continuous_resilience = min(1.0, stability_score)
    else:
        continuous_resilience = 0.5  # Valeur par défaut pour Kuramoto
    
    # Résultats finaux
    results = {
        'logs': loggers.get('log_file', 'kuramoto_log.csv'),
        'metrics': {
            'mean_S': np.mean(S_history),
            'std_S': np.std(S_history),
            'mean_C': np.mean(C_history),
            'std_C': np.std(C_history),
            'mean_cpu_step': np.mean(cpu_steps),
            'final_order': order_params[-1],
            'variance_d2S': variance_d2S,
            'entropy_S': entropy_S,
            't_retour': t_retour,
            'mean_effort': 0.0,  # Toujours 0 pour Kuramoto
            'mode': 'Kuramoto',
            'continuous_resilience': continuous_resilience
        },
        'history': history,
        'run_id': loggers['run_id'],
        'S_history': np.array(S_history),
        'C_history': np.array(C_history),
        'cpu_history': np.array(cpu_steps),
        'order_params': np.array(order_params)
    }
    
    return deep_convert(results)


# ============== ANALYSE COMPARATIVE ==============

def analyze_kuramoto_performance(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyse les performances spécifiques de Kuramoto.
    
    Args:
        results: résultats de la simulation
    
    Returns:
        Dict avec métriques d'analyse
    """
    analysis = {}
    
    order_params = results.get('order_params', [])
    if len(order_params) > 0:
        # Temps de synchronisation (quand r > 0.9 pour la première fois)
        sync_time = None
        for i, r in enumerate(order_params):
            if r > 0.9:
                sync_time = i * results['metrics'].get('dt', 0.05)
                break
        
        analysis['sync_time'] = sync_time if sync_time else float('inf')
        
        # Stabilité de la synchronisation (std de r dans les 20% derniers)
        last_20_percent = int(len(order_params) * 0.8)
        analysis['sync_stability'] = np.std(order_params[last_20_percent:])
        
        # Efficacité : synchronisation finale / coût CPU
        mean_cpu = results['metrics'].get('mean_cpu_step', 1e-6)
        analysis['efficiency'] = order_params[-1] / mean_cpu if mean_cpu > 0 else 0
    
    return analysis


# ============== COMPARAISON AVEC FPS ==============

def compare_with_fps(kuramoto_results: Dict, fps_results: Dict) -> Dict[str, Any]:
    """
    Compare les résultats Kuramoto avec FPS.
    
    Args:
        kuramoto_results: résultats Kuramoto
        fps_results: résultats FPS
    
    Returns:
        Dict avec comparaisons
    """
    comparison = {
        'synchronization': {
            'kuramoto': kuramoto_results['metrics'].get('final_order', 0),
            'fps': fps_results['metrics'].get('mean_C', 0),
            'winner': 'fps' if fps_results['metrics'].get('mean_C', 0) > 
                              kuramoto_results['metrics'].get('final_order', 0) else 'kuramoto'
        },
        'stability': {
            'kuramoto': kuramoto_results['metrics'].get('std_S', float('inf')),
            'fps': fps_results['metrics'].get('std_S', float('inf')),
            'winner': 'fps' if fps_results['metrics'].get('std_S', float('inf')) < 
                              kuramoto_results['metrics'].get('std_S', float('inf')) else 'kuramoto'
        },
        'resilience': {
            'kuramoto': kuramoto_results['metrics'].get('t_retour', float('inf')),
            'fps': fps_results['metrics'].get('t_retour', float('inf')),
            'winner': 'fps' if fps_results['metrics'].get('t_retour', float('inf')) < 
                              kuramoto_results['metrics'].get('t_retour', float('inf')) else 'kuramoto'
        },
        'cpu_efficiency': {
            'kuramoto': kuramoto_results['metrics'].get('mean_cpu_step', float('inf')),
            'fps': fps_results['metrics'].get('mean_cpu_step', float('inf')),
            'winner': 'fps' if fps_results['metrics'].get('mean_cpu_step', float('inf')) < 
                              kuramoto_results['metrics'].get('mean_cpu_step', float('inf')) else 'kuramoto'
        },
        'innovation': {
            'kuramoto': kuramoto_results['metrics'].get('entropy_S', 0),
            'fps': fps_results['metrics'].get('entropy_S', 0),
            'winner': 'fps' if fps_results['metrics'].get('entropy_S', 0) > 
                              kuramoto_results['metrics'].get('entropy_S', 0) else 'kuramoto'
        }
    }
    
    # Score global
    fps_wins = sum(1 for metric in comparison.values() if metric['winner'] == 'fps')
    kuramoto_wins = len(comparison) - fps_wins
    
    comparison['overall_winner'] = 'fps' if fps_wins > kuramoto_wins else 'kuramoto'
    comparison['score'] = f"FPS {fps_wins} - {kuramoto_wins} Kuramoto"
    
    return comparison


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module kuramoto.py
    """
    print("=== Tests du module kuramoto.py ===\n")
    
    # Configuration de test
    test_config = {
        'system': {
            'N': 10,
            'T': 50,
            'dt': 0.1,
            'seed': 42,
            'perturbation': {
                'type': 'choc',
                't0': 25,
                'amplitude': 2.0
            },
            'logging': {
                'log_metrics': ['t', 'S(t)', 'C(t)', 'E(t)', 'cpu_step(t)']
            }
        }
    }
    
    # Test 1: Pas de Kuramoto
    print("Test 1 - Dynamique de base:")
    phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    frequencies = np.ones(5) * 0.5
    
    dphases = kuramoto_step(phases, frequencies, K=1.0, N=5, dt=0.1)
    print(f"  Phases initiales: {phases}")
    print(f"  Dérivées: {dphases}")
    
    # Test 2: Paramètre d'ordre
    print("\nTest 2 - Paramètre d'ordre:")
    # Cas synchronisé
    phases_sync = np.zeros(10)
    r_sync, psi_sync = compute_kuramoto_order(phases_sync)
    print(f"  Synchronisé: r={r_sync:.3f}, ψ={psi_sync:.3f}")
    
    # Cas désynchronisé
    phases_desync = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r_desync, psi_desync = compute_kuramoto_order(phases_desync)
    print(f"  Désynchronisé: r={r_desync:.3f}, ψ={psi_desync:.3f}")
    
    # Test 3: Mini simulation
    print("\nTest 3 - Mini simulation:")
    
    # Créer un mock logger
    class MockWriter:
        def writerow(self, data):
            pass
    
    test_loggers = {
        'csv_writer': MockWriter(),
        'run_id': 'test_kuramoto',
        'log_file': 'test_kuramoto.csv'
    }
    
    # Lancer la simulation
    results = run_kuramoto_simulation(test_config, test_loggers)
    
    print(f"\nRésultats:")
    print(f"  Synchronisation finale: {results['metrics']['final_order']:.3f}")
    print(f"  Signal moyen: {results['metrics']['mean_S']:.3f}")
    print(f"  CPU moyen: {results['metrics']['mean_cpu_step']*1000:.3f} ms")
    
    # Test 4: Analyse
    print("\nTest 4 - Analyse des performances:")
    analysis = analyze_kuramoto_performance(results)
    for key, value in analysis.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n✅ Module kuramoto.py prêt pour la comparaison!")
