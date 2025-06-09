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
    """
    metrics = {}
    
    # 1. Synchronisation (basée sur C(t) final)
    fps_sync = fps_result.get('metrics', {}).get('mean_C', 0)
    kura_sync = kuramoto_result.get('metrics', {}).get('mean_C', 0)
    neutral_sync = neutral_result.get('metrics', {}).get('mean_C', 0)
    
    metrics['synchronization'] = {
        'fps_value': fps_sync,
        'kuramoto_value': kura_sync,
        'neutral_value': neutral_sync,
        'fps_vs_kuramoto_efficiency': (fps_sync - kura_sync) / (kura_sync + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_sync - neutral_sync) / (neutral_sync + 1e-10) * 100
    }
    
    # 2. Stabilité (basée sur std_S)
    fps_stability = 1.0 / (fps_result.get('metrics', {}).get('std_S', float('inf')) + 1e-10)
    kura_stability = 1.0 / (kuramoto_result.get('metrics', {}).get('std_S', float('inf')) + 1e-10)
    neutral_stability = 1.0 / (neutral_result.get('metrics', {}).get('std_S', float('inf')) + 1e-10)
    
    metrics['stability'] = {
        'fps_value': fps_stability,
        'kuramoto_value': kura_stability,
        'neutral_value': neutral_stability,
        'fps_vs_kuramoto_efficiency': (fps_stability - kura_stability) / (kura_stability + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_stability - neutral_stability) / (neutral_stability + 1e-10) * 100
    }
    
    # 3. Résilience (basée sur t_retour)
    fps_resilience = 1.0 / (fps_result.get('metrics', {}).get('resilience_t_retour', float('inf')) + 1)
    kura_resilience = 1.0 / (kuramoto_result.get('metrics', {}).get('t_retour', float('inf')) + 1)
    neutral_resilience = 0.1  # Neutral n'a pas de résilience active
    
    metrics['resilience'] = {
        'fps_value': fps_resilience,
        'kuramoto_value': kura_resilience,
        'neutral_value': neutral_resilience,
        'fps_vs_kuramoto_efficiency': (fps_resilience - kura_resilience) / (kura_resilience + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_resilience - neutral_resilience) / (neutral_resilience + 1e-10) * 100
    }
    
    # 4. Innovation (basée sur entropy_S)
    fps_innovation = fps_result.get('metrics', {}).get('final_entropy_S', 0)
    kura_innovation = kuramoto_result.get('metrics', {}).get('entropy_S', 0)
    neutral_innovation = neutral_result.get('metrics', {}).get('entropy_S', 0)
    
    metrics['innovation'] = {
        'fps_value': fps_innovation,
        'kuramoto_value': kura_innovation,
        'neutral_value': neutral_innovation,
        'fps_vs_kuramoto_efficiency': (fps_innovation - kura_innovation) / (kura_innovation + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_innovation - neutral_innovation) / (neutral_innovation + 1e-10) * 100
    }
    
    # 5. Efficacité CPU (inverse du coût)
    fps_cpu_eff = 1.0 / (fps_result.get('metrics', {}).get('mean_cpu_step', 1) + 1e-10)
    kura_cpu_eff = 1.0 / (kuramoto_result.get('metrics', {}).get('mean_cpu_step', 1) + 1e-10)
    neutral_cpu_eff = 1.0 / (neutral_result.get('metrics', {}).get('mean_cpu_step', 1) + 1e-10)
    
    metrics['cpu_efficiency'] = {
        'fps_value': fps_cpu_eff,
        'kuramoto_value': kura_cpu_eff,
        'neutral_value': neutral_cpu_eff,
        'fps_vs_kuramoto_efficiency': (fps_cpu_eff - kura_cpu_eff) / (kura_cpu_eff + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_cpu_eff - neutral_cpu_eff) / (neutral_cpu_eff + 1e-10) * 100
    }
    
    # Score global
    fps_score = np.mean([
        metrics['synchronization']['fps_value'],
        metrics['stability']['fps_value'],
        metrics['resilience']['fps_value'],
        metrics['innovation']['fps_value'],
        metrics['cpu_efficiency']['fps_value']
    ])
    
    kura_score = np.mean([
        metrics['synchronization']['kuramoto_value'],
        metrics['stability']['kuramoto_value'],
        metrics['resilience']['kuramoto_value'],
        metrics['innovation']['kuramoto_value'],
        metrics['cpu_efficiency']['kuramoto_value']
    ])
    
    neutral_score = np.mean([
        metrics['synchronization']['neutral_value'],
        metrics['stability']['neutral_value'],
        metrics['resilience']['neutral_value'],
        metrics['innovation']['neutral_value'],
        metrics['cpu_efficiency']['neutral_value']
    ])
    
    metrics['global_score'] = {
        'fps': fps_score,
        'kuramoto': kura_score,
        'neutral': neutral_score,
        'fps_vs_kuramoto_efficiency': (fps_score - kura_score) / (kura_score + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_score - neutral_score) / (neutral_score + 1e-10) * 100
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