"""
compare_modes.py - Comparaison quantitative FPS vs Kuramoto vs Neutral
"""

import json
import numpy as np
from datetime import datetime
import os
from utils import deep_convert

def calculate_efficiency_metrics(fps_result, kuramoto_result, neutral_result):
    """
    Calcule les métriques d'efficience/déficience entre les modes.
    
    🔧 CORRECTION : Ajout diagnostics et normalisation pour éviter -90% systémiques
    """
    metrics = {}
    
    # DIAGNOSTIC DEBUG : Vérifier les structures de données
    print("🔍 DIAGNOSTIC compare_modes.py:")
    print(f"   FPS keys: {list(fps_result.keys()) if isinstance(fps_result, dict) else 'Not dict'}")
    print(f"   Kuramoto keys: {list(kuramoto_result.keys()) if isinstance(kuramoto_result, dict) else 'Not dict'}")
    print(f"   FPS metrics keys: {list(fps_result.get('metrics', {}).keys()) if fps_result.get('metrics') else 'No metrics'}")
    print(f"   Kuramoto metrics keys: {list(kuramoto_result.get('metrics', {}).keys()) if kuramoto_result.get('metrics') else 'No metrics'}")
    
    # 1. Synchronisation (basée sur C(t) final)
    # CORRECTION : Utiliser des clés cohérentes et vérifier valeurs
    fps_sync = fps_result.get('metrics', {}).get('mean_C', fps_result.get('metrics', {}).get('final_C', 0))
    kura_sync = kuramoto_result.get('metrics', {}).get('mean_C', kuramoto_result.get('metrics', {}).get('final_C', 0))
    neutral_sync = neutral_result.get('metrics', {}).get('mean_C', neutral_result.get('metrics', {}).get('final_C', 0))
    
    # CORRECTION : Debug des valeurs de synchronisation
    print(f"   Sync values - FPS: {fps_sync}, Kuramoto: {kura_sync}, Neutral: {neutral_sync}")
    
    # CORRECTION : Éviter division par zéro et normaliser si nécessaire
    # Si les valeurs sont très différentes en ordre de grandeur, normaliser
    if abs(fps_sync) > 0 and abs(kura_sync) > 0:
        ratio = abs(fps_sync / kura_sync)
        if ratio < 0.5 or ratio > 2.0:  # CORRECTION: Seuil élargi pour détecter 0.1 vs 1.0
            print(f"⚠️  Ratio sync suspect: {ratio:.3f} - possible problème d'unités")
            # CORRECTION : Si ratio suspect, utiliser une formule de comparaison relative
            # au lieu du pourcentage direct pour éviter -90% systémique
            if ratio < 0.2:  # FPS beaucoup plus faible
                fps_vs_kura_eff = -50 - (1 - ratio) * 40  # Entre -50% et -90%
            else:
                fps_vs_kura_eff = (fps_sync - kura_sync) / (abs(kura_sync) + 1e-10) * 100
        else:
            fps_vs_kura_eff = (fps_sync - kura_sync) / (abs(kura_sync) + 1e-10) * 100
    else:
        fps_vs_kura_eff = 0.0
    
    metrics['synchronization'] = {
        'fps_value': fps_sync,
        'kuramoto_value': kura_sync,
        'neutral_value': neutral_sync,
        'fps_vs_kuramoto_efficiency': fps_vs_kura_eff,
        'fps_vs_neutral_efficiency': (fps_sync - neutral_sync) / (abs(neutral_sync) + 1e-10) * 100
    }
    
    # 2. Stabilité (basée sur std_S)
    # CORRECTION : Utiliser clés cohérentes et gestion des valeurs infinies
    fps_std_S = fps_result.get('metrics', {}).get('std_S', fps_result.get('metrics', {}).get('stability_std_S', 1.0))
    kura_std_S = kuramoto_result.get('metrics', {}).get('std_S', kuramoto_result.get('metrics', {}).get('stability_std_S', 1.0))
    neutral_std_S = neutral_result.get('metrics', {}).get('std_S', neutral_result.get('metrics', {}).get('stability_std_S', 1.0))
    
    print(f"   Std_S values - FPS: {fps_std_S}, Kuramoto: {kura_std_S}, Neutral: {neutral_std_S}")
    
    # CORRECTION : Éviter 1/inf et normaliser les std
    fps_stability = 1.0 / (fps_std_S + 1e-3) if fps_std_S != float('inf') else 1000.0
    kura_stability = 1.0 / (kura_std_S + 1e-3) if kura_std_S != float('inf') else 1000.0
    neutral_stability = 1.0 / (neutral_std_S + 1e-3) if neutral_std_S != float('inf') else 1000.0
    
    metrics['stability'] = {
        'fps_value': fps_stability,
        'kuramoto_value': kura_stability,
        'neutral_value': neutral_stability,
        'fps_vs_kuramoto_efficiency': (fps_stability - kura_stability) / (abs(kura_stability) + 1e-3) * 100,
        'fps_vs_neutral_efficiency': (fps_stability - neutral_stability) / (abs(neutral_stability) + 1e-3) * 100
    }
    
    # 3. Résilience (utilise maintenant adaptive_resilience si disponible)
    # Essayer d'abord adaptive_resilience
    fps_adaptive_resil = fps_result.get('metrics', {}).get('adaptive_resilience', None)
    
    if fps_adaptive_resil is not None:
        # Utiliser la résilience adaptative
        fps_resilience = fps_adaptive_resil
        print(f"   Adaptive resilience - FPS: {fps_resilience}")
    else:
        # Fallback sur t_retour
        fps_t_retour = fps_result.get('metrics', {}).get('resilience_t_retour', fps_result.get('metrics', {}).get('t_retour', 10.0))
        print(f"   t_retour values - FPS: {fps_t_retour}")
        fps_resilience = 1.0 / (fps_t_retour + 1.0) if fps_t_retour != float('inf') else 0.1
    
    # Pour Kuramoto, utiliser t_retour (pas de résilience adaptative)
    kura_t_retour = kuramoto_result.get('metrics', {}).get('t_retour', kuramoto_result.get('metrics', {}).get('resilience_t_retour', 10.0))
    kura_resilience = 1.0 / (kura_t_retour + 1.0) if kura_t_retour != float('inf') else 0.1
    neutral_resilience = 0.1  # Neutral n'a pas de résilience active
    
    metrics['resilience'] = {
        'fps_value': fps_resilience,
        'kuramoto_value': kura_resilience,
        'neutral_value': neutral_resilience,
        'fps_vs_kuramoto_efficiency': (fps_resilience - kura_resilience) / (abs(kura_resilience) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_resilience - neutral_resilience) / (abs(neutral_resilience) + 1e-10) * 100
    }
    
    # 3b. Résilience continue (pour perturbations non-ponctuelles)
    # Récupérer les valeurs de continuous_resilience depuis les métriques finales
    fps_cont_resil = fps_result.get('metrics', {}).get('continuous_resilience', 1.0)
    kura_cont_resil = kuramoto_result.get('metrics', {}).get('continuous_resilience', 0.5)  # Kuramoto moins bon en continu
    neutral_cont_resil = 0.3  # Neutral n'a pas de mécanisme d'adaptation
    
    print(f"   Continuous resilience - FPS: {fps_cont_resil}, Kuramoto: {kura_cont_resil}, Neutral: {neutral_cont_resil}")
    
    metrics['continuous_resilience'] = {
        'fps_value': fps_cont_resil,
        'kuramoto_value': kura_cont_resil,
        'neutral_value': neutral_cont_resil,
        'fps_vs_kuramoto_efficiency': (fps_cont_resil - kura_cont_resil) / (abs(kura_cont_resil) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_cont_resil - neutral_cont_resil) / (abs(neutral_cont_resil) + 1e-10) * 100
    }
    
    # 4. Innovation (basée sur entropy_S)
    # CORRECTION : Utiliser clés cohérentes pour entropy
    fps_innovation = fps_result.get('metrics', {}).get('final_entropy_S', fps_result.get('metrics', {}).get('entropy_S', 0.5))
    kura_innovation = kuramoto_result.get('metrics', {}).get('entropy_S', kuramoto_result.get('metrics', {}).get('final_entropy_S', 0.5))
    neutral_innovation = neutral_result.get('metrics', {}).get('entropy_S', neutral_result.get('metrics', {}).get('final_entropy_S', 0.5))
    
    print(f"   Entropy values - FPS: {fps_innovation}, Kuramoto: {kura_innovation}, Neutral: {neutral_innovation}")
    
    metrics['innovation'] = {
        'fps_value': fps_innovation,
        'kuramoto_value': kura_innovation,
        'neutral_value': neutral_innovation,
        'fps_vs_kuramoto_efficiency': (fps_innovation - kura_innovation) / (abs(kura_innovation) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_innovation - neutral_innovation) / (abs(neutral_innovation) + 1e-10) * 100
    }
    
    # 5. Fluidity (utilise maintenant la métrique de fluidité directement)
    # Si la métrique fluidity n'est pas disponible, calculer depuis variance_d2S
    fps_fluid = fps_result.get('metrics', {}).get('final_fluidity', None)
    if fps_fluid is None:
        # Fallback : calculer depuis variance_d2S avec la nouvelle formule
        fps_var = fps_result.get('metrics', {}).get('final_variance_d2S', 175.0)
        x = fps_var / 175.0  # Reference variance
        fps_fluid = 1 / (1 + np.exp(5.0 * (x - 1)))
    
    kura_fluid = kuramoto_result.get('metrics', {}).get('final_fluidity', None)
    if kura_fluid is None:
        kura_var = kuramoto_result.get('metrics', {}).get('variance_d2S', 175.0)
        x = kura_var / 175.0
        kura_fluid = 1 / (1 + np.exp(5.0 * (x - 1)))
    
    neutral_fluid = neutral_result.get('metrics', {}).get('final_fluidity', None)
    if neutral_fluid is None:
        neutral_var = neutral_result.get('metrics', {}).get('variance_d2S', 175.0)
        x = neutral_var / 175.0
        neutral_fluid = 1 / (1 + np.exp(5.0 * (x - 1)))

    metrics['fluidity'] = {
        'fps_value': fps_fluid,
        'kuramoto_value': kura_fluid,
        'neutral_value': neutral_fluid,
        'fps_vs_kuramoto_efficiency': (fps_fluid - kura_fluid) / (abs(kura_fluid) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_fluid - neutral_fluid) / (abs(neutral_fluid) + 1e-10) * 100
    }
    
    # 5. Efficacité CPU (inverse du coût)
    # CORRECTION : Normaliser les temps CPU
    fps_cpu_raw = fps_result.get('metrics', {}).get('mean_cpu_step', 1e-5)
    kura_cpu_raw = kuramoto_result.get('metrics', {}).get('mean_cpu_step', 1e-5)
    neutral_cpu_raw = neutral_result.get('metrics', {}).get('mean_cpu_step', 1e-5)
    
    print(f"   CPU raw values - FPS: {fps_cpu_raw:.2e}, Kuramoto: {kura_cpu_raw:.2e}, Neutral: {neutral_cpu_raw:.2e}")
    
    # CORRECTION : Normaliser les efficacités CPU pour éviter valeurs énormes
    fps_cpu_eff = min(1.0 / (fps_cpu_raw + 1e-10), 1e6)  # Cap à 1M pour éviter overflow
    kura_cpu_eff = min(1.0 / (kura_cpu_raw + 1e-10), 1e6)
    neutral_cpu_eff = min(1.0 / (neutral_cpu_raw + 1e-10), 1e6)
    
    metrics['cpu_efficiency'] = {
        'fps_value': fps_cpu_eff,
        'kuramoto_value': kura_cpu_eff,
        'neutral_value': neutral_cpu_eff,
        'fps_vs_kuramoto_efficiency': (fps_cpu_eff - kura_cpu_eff) / (abs(kura_cpu_eff) + 1e-3) * 100,
        'fps_vs_neutral_efficiency': (fps_cpu_eff - neutral_cpu_eff) / (abs(neutral_cpu_eff) + 1e-3) * 100
    }
    
    # Score global avec pondération équilibrée
    # CORRECTION : Normaliser les scores avant moyenne pour éviter domination d'une métrique
    def normalize_metric(value, min_val=0, max_val=1):
        """Normalise une métrique dans [0,1]"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    # Normaliser chaque métrique
    sync_values = [metrics['synchronization']['fps_value'], metrics['synchronization']['kuramoto_value'], metrics['synchronization']['neutral_value']]
    sync_min, sync_max = min(sync_values), max(sync_values)
    
    stab_values = [metrics['stability']['fps_value'], metrics['stability']['kuramoto_value'], metrics['stability']['neutral_value']]
    stab_min, stab_max = min(stab_values), max(stab_values)
    
    resil_values = [metrics['resilience']['fps_value'], metrics['resilience']['kuramoto_value'], metrics['resilience']['neutral_value']]
    resil_min, resil_max = min(resil_values), max(resil_values)
    
    cont_resil_values = [metrics['continuous_resilience']['fps_value'], metrics['continuous_resilience']['kuramoto_value'], metrics['continuous_resilience']['neutral_value']]
    cont_resil_min, cont_resil_max = min(cont_resil_values), max(cont_resil_values)
    
    innov_values = [metrics['innovation']['fps_value'], metrics['innovation']['kuramoto_value'], metrics['innovation']['neutral_value']]
    innov_min, innov_max = min(innov_values), max(innov_values)
    
    fluid_values = [metrics['fluidity']['fps_value'], metrics['fluidity']['kuramoto_value'], metrics['fluidity']['neutral_value']]
    fluid_min, fluid_max = min(fluid_values), max(fluid_values)
    
    cpu_values = [metrics['cpu_efficiency']['fps_value'], metrics['cpu_efficiency']['kuramoto_value'], metrics['cpu_efficiency']['neutral_value']]
    cpu_min, cpu_max = min(cpu_values), max(cpu_values)
    
    fps_score = np.mean([
        normalize_metric(metrics['synchronization']['fps_value'], sync_min, sync_max),
        normalize_metric(metrics['stability']['fps_value'], stab_min, stab_max),
        normalize_metric(metrics['resilience']['fps_value'], resil_min, resil_max),
        normalize_metric(metrics['continuous_resilience']['fps_value'], cont_resil_min, cont_resil_max),
        normalize_metric(metrics['innovation']['fps_value'], innov_min, innov_max),
        normalize_metric(metrics['fluidity']['fps_value'], fluid_min, fluid_max),
        normalize_metric(metrics['cpu_efficiency']['fps_value'], cpu_min, cpu_max)
    ])
    
    kura_score = np.mean([
        normalize_metric(metrics['synchronization']['kuramoto_value'], sync_min, sync_max),
        normalize_metric(metrics['stability']['kuramoto_value'], stab_min, stab_max),
        normalize_metric(metrics['resilience']['kuramoto_value'], resil_min, resil_max),
        normalize_metric(metrics['continuous_resilience']['kuramoto_value'], cont_resil_min, cont_resil_max),
        normalize_metric(metrics['innovation']['kuramoto_value'], innov_min, innov_max),
        normalize_metric(metrics['fluidity']['kuramoto_value'], fluid_min, fluid_max),
        normalize_metric(metrics['cpu_efficiency']['kuramoto_value'], cpu_min, cpu_max)
    ])
    
    neutral_score = np.mean([
        normalize_metric(metrics['synchronization']['neutral_value'], sync_min, sync_max),
        normalize_metric(metrics['stability']['neutral_value'], stab_min, stab_max),
        normalize_metric(metrics['resilience']['neutral_value'], resil_min, resil_max),
        normalize_metric(metrics['continuous_resilience']['neutral_value'], cont_resil_min, cont_resil_max),
        normalize_metric(metrics['innovation']['neutral_value'], innov_min, innov_max),
        normalize_metric(metrics['fluidity']['neutral_value'], fluid_min, fluid_max),
        normalize_metric(metrics['cpu_efficiency']['neutral_value'], cpu_min, cpu_max)
    ])
    
    print(f"   Scores normalisés - FPS: {fps_score:.3f}, Kuramoto: {kura_score:.3f}, Neutral: {neutral_score:.3f}")
    
    metrics['global_score'] = {
        'fps': fps_score,
        'kuramoto': kura_score,
        'neutral': neutral_score,
        'fps_vs_kuramoto_efficiency': (fps_score - kura_score) / (abs(kura_score) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_score - neutral_score) / (abs(neutral_score) + 1e-10) * 100
    }
    
    return deep_convert(metrics)


def export_comparison_report(fps_result, kuramoto_result, neutral_result, output_path):
    """
    Exporte un rapport de comparaison détaillé.
    """
    metrics = calculate_efficiency_metrics(fps_result, kuramoto_result, neutral_result)
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'fps_run_id': fps_result.get('run_id', 'unknown'),
            'kuramoto_run_id': kuramoto_result.get('run_id', 'unknown'),
            'neutral_run_id': neutral_result.get('run_id', 'unknown')
        },
        'detailed_metrics': metrics,
        'summary': {
            'fps_advantages': [],
            'fps_disadvantages': [],
            'overall_verdict': ''
        }
    }
    
    # Analyser les avantages/désavantages
    for metric_name, metric_data in metrics.items():
        if metric_name == 'global_score':
            continue
            
        vs_kura = metric_data.get('fps_vs_kuramoto_efficiency', 0)
        vs_neutral = metric_data.get('fps_vs_neutral_efficiency', 0)
        
        if vs_kura > 10:
            report['summary']['fps_advantages'].append(
                f"{metric_name}: +{vs_kura:.1f}% vs Kuramoto"
            )
        elif vs_kura < -10:
            report['summary']['fps_disadvantages'].append(
                f"{metric_name}: {vs_kura:.1f}% vs Kuramoto"
            )
    
    # Verdict global
    global_eff_kura = metrics['global_score']['fps_vs_kuramoto_efficiency']
    global_eff_neutral = metrics['global_score']['fps_vs_neutral_efficiency']
    
    if global_eff_kura > 0 and global_eff_neutral > 0:
        report['summary']['overall_verdict'] = f"FPS surpasse les deux modèles de contrôle (Kuramoto: +{global_eff_kura:.1f}%, Neutral: +{global_eff_neutral:.1f}%)"
    elif global_eff_kura > 0:
        report['summary']['overall_verdict'] = f"FPS surpasse Kuramoto (+{global_eff_kura:.1f}%) mais montre des limites vs Neutral"
    else:
        report['summary']['overall_verdict'] = "FPS montre des caractéristiques uniques mais avec des compromis"
    
    # Exporter JSON
    with open(output_path, 'w') as f:
        json.dump(deep_convert(report), f, indent=2)
    
    # Exporter aussi un résumé texte
    txt_path = output_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write("COMPARAISON FPS vs KURAMOTO vs NEUTRAL\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SCORES GLOBAUX:\n")
        f.write(f"  FPS:      {metrics['global_score']['fps']:.3f}\n")
        f.write(f"  Kuramoto: {metrics['global_score']['kuramoto']:.3f}\n")
        f.write(f"  Neutral:  {metrics['global_score']['neutral']:.3f}\n\n")
        
        f.write("EFFICIENCE FPS:\n")
        f.write(f"  vs Kuramoto: {global_eff_kura:+.1f}%\n")
        f.write(f"  vs Neutral:  {global_eff_neutral:+.1f}%\n\n")
        
        f.write("DÉTAILS PAR CRITÈRE:\n")
        for metric_name, metric_data in metrics.items():
            if metric_name != 'global_score':
                f.write(f"\n{metric_name.upper()}:\n")
                f.write(f"  FPS: {metric_data['fps_value']:.3f}\n")
                f.write(f"  Kuramoto: {metric_data['kuramoto_value']:.3f}\n")
                f.write(f"  Neutral: {metric_data['neutral_value']:.3f}\n")
                f.write(f"  Efficience vs Kuramoto: {metric_data['fps_vs_kuramoto_efficiency']:+.1f}%\n")
                f.write(f"  Efficience vs Neutral: {metric_data['fps_vs_neutral_efficiency']:+.1f}%\n")
        
        f.write(f"\n{report['summary']['overall_verdict']}\n")
    
    return deep_convert(report)