"""
Test des nouvelles formules de métriques et vérification du fix continuous_resilience
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('.')
from fluidity_proposals import *

# Définir les modes de gamma et leurs dossiers
gamma_modes = {
    'static': 'gamma_static_run_20250630_174926',
    'dynamic': 'gamma_dynamic_run_20250630_175036',
    'sigmoid_up': 'gamma_sigmoid_up_run_20250630_175206',
    'sigmoid_down': 'gamma_sigmoid_down_run_20250630_175308',
    'sigmoid_oscillating': 'gamma_sigmoid_oscillating_run_20250630_175419',
    'sinusoidal': 'gamma_sinusoidal_run_20250630_175513',
    'sigmoid_adaptive': 'gamma_sigmoid_adaptive_run_20250630_175907'
}

def test_fluidity_formulas():
    """Tester les différentes formules de fluidity sur les données réelles"""
    print("=== Test des nouvelles formules de fluidity ===\n")
    
    results = {}
    
    for mode_name, folder_name in gamma_modes.items():
        base_path = Path(f'fps_pipeline_output/{folder_name}/logs')
        csv_files = list(base_path.glob('batch_run_*.csv'))
        
        if csv_files:
            # Lire le premier fichier CSV
            df = pd.read_csv(csv_files[0])
            
            # Récupérer variance_d2S
            variance_d2S_values = df['variance_d2S'].values
            variance_d2S_mean = np.mean(variance_d2S_values)
            
            # Calculer fluidity avec différentes formules
            fluidity_original = 1 / (1 + variance_d2S_mean)
            fluidity_adaptive = compute_fluidity_adaptive(variance_d2S_mean, ref_value=100.0)
            fluidity_log = compute_fluidity_log(variance_d2S_mean, scale=50.0)
            
            # Pour gamma_aware, on a besoin de l'historique de gamma
            if 'gamma_mean(t)' in df.columns:
                gamma_history = df['gamma_mean(t)'].values
                fluidity_gamma = compute_fluidity_gamma_aware(variance_d2S_mean, gamma_history)
            else:
                fluidity_gamma = fluidity_original
            
            # Pour percentile, on a besoin de l'historique de variance
            fluidity_percentile = compute_fluidity_percentile(variance_d2S_mean, variance_d2S_values)
            
            results[mode_name] = {
                'variance_d2S': variance_d2S_mean,
                'fluidity_original': fluidity_original,
                'fluidity_adaptive': fluidity_adaptive,
                'fluidity_log': fluidity_log,
                'fluidity_gamma': fluidity_gamma,
                'fluidity_percentile': fluidity_percentile
            }
    
    # Afficher les résultats
    print("Comparaison des formules de fluidity:\n")
    print(f"{'Mode':<20} {'Var_d2S':>10} {'Original':>10} {'Adaptive':>10} {'Log':>10} {'Gamma':>10} {'Percentile':>10}")
    print("-" * 90)
    
    for mode, values in results.items():
        print(f"{mode:<20} {values['variance_d2S']:>10.2f} {values['fluidity_original']:>10.6f} "
              f"{values['fluidity_adaptive']:>10.4f} {values['fluidity_log']:>10.4f} "
              f"{values['fluidity_gamma']:>10.6f} {values['fluidity_percentile']:>10.4f}")
    
    # Calculer la variance entre modes pour chaque formule
    print("\n\nVariance des scores entre modes:")
    for metric in ['fluidity_original', 'fluidity_adaptive', 'fluidity_log', 'fluidity_gamma', 'fluidity_percentile']:
        values = [r[metric] for r in results.values()]
        variance = np.var(values)
        print(f"{metric}: variance = {variance:.8f} (std = {np.std(values):.6f})")
    
    return results

def check_continuous_resilience():
    """Vérifier si continuous_resilience varie maintenant entre les modes"""
    print("\n\n=== Vérification de continuous_resilience après fix ===\n")
    
    # Lire les rapports de comparaison pour voir les valeurs
    for mode_name, folder_name in gamma_modes.items():
        report_path = Path(f'fps_pipeline_output/{folder_name}/reports/comparison_fps_vs_controls.json')
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # Extraire continuous_resilience
            cont_resil = report.get('detailed_metrics', {}).get('continuous_resilience', {}).get('fps_value', 'N/A')
            print(f"{mode_name:<20}: continuous_resilience = {cont_resil}")
    
    print("\n⚠️  Note: Si toutes les valeurs sont encore 1.0, il faut relancer les simulations")
    print("    car le fix n'affecte que les nouvelles exécutions.")

def check_perturbation_detection():
    """Vérifier la détection des perturbations dans la config"""
    print("\n\n=== Vérification de la détection des perturbations ===\n")
    
    # Lire une config pour voir la structure
    config_path = Path('config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ancienne structure
        old_pert = config.get('perturbation', {})
        print(f"Ancienne structure config['perturbation']: {old_pert}")
        
        # Nouvelle structure
        new_pert = config.get('system', {}).get('input', {}).get('perturbations', [])
        print(f"Nouvelle structure config['system']['input']['perturbations']: {new_pert}")
        
        # Test de détection avec notre fix
        perturbations = config.get('system', {}).get('input', {}).get('perturbations', [])
        perturbation_active = len(perturbations) > 0 and any(p.get('type', 'none') != 'none' for p in perturbations)
        print(f"\nPerturbation détectée avec le fix: {perturbation_active}")

def main():
    """Main function"""
    print("🧪 Test des nouvelles métriques FPS\n")
    
    # Test 1: Nouvelles formules de fluidity
    fluidity_results = test_fluidity_formulas()
    
    # Test 2: Vérification continuous_resilience
    check_continuous_resilience()
    
    # Test 3: Vérification détection perturbations
    check_perturbation_detection()
    
    print("\n\n=== Recommandations ===")
    print("1. La formule 'adaptive' ou 'log' semble mieux capturer les variations")
    print("2. La formule 'gamma_aware' prend en compte gamma directement")
    print("3. Il faut relancer les simulations pour voir l'effet du fix continuous_resilience")
    print("4. Considérer d'ajouter des métriques spécifiques à gamma dans metrics.py")

if __name__ == "__main__":
    main() 