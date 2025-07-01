#!/usr/bin/env python3
"""
Test de la métrique de fluidité avec différents modes gamma
"""

import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

def run_simulation_with_gamma_mode(gamma_mode):
    """Lance une simulation avec un mode gamma spécifique"""
    print(f"\n🔬 Test avec gamma_mode = {gamma_mode}")
    
    # Charger la config de base
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Modifier le mode gamma
    config['latence']['gamma_mode'] = gamma_mode
    
    # Réduire la durée pour un test rapide
    config['system']['T'] = 20
    config['system']['dt'] = 0.1
    
    # Sauvegarder la config temporaire
    temp_config = f'test_config_{gamma_mode}.json'
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Lancer la simulation
    cmd = f'python3 simulate.py {temp_config} FPS'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ❌ Erreur : {result.stderr}")
        return None
    
    # Extraire le run_id depuis stdout
    for line in result.stdout.split('\n'):
        if 'Run ID:' in line:
            run_id = line.split('Run ID:')[1].strip()
            break
    else:
        print("  ❌ Run ID non trouvé")
        return None
    
    # Lire le CSV généré
    csv_path = f'logs/{run_id}.csv'
    try:
        df = pd.read_csv(csv_path)
        
        # Calculer les statistiques
        stats = {
            'gamma_mode': gamma_mode,
            'variance_d2S_mean': df['variance_d2S'].mean(),
            'variance_d2S_std': df['variance_d2S'].std(),
            'fluidity_mean': df['fluidity'].mean(),
            'fluidity_std': df['fluidity'].std(),
            'fluidity_min': df['fluidity'].min(),
            'fluidity_max': df['fluidity'].max()
        }
        
        print(f"  ✓ variance_d2S moyenne : {stats['variance_d2S_mean']:.2f}")
        print(f"  ✓ fluidity moyenne : {stats['fluidity_mean']:.4f}")
        print(f"  ✓ fluidity range : [{stats['fluidity_min']:.4f}, {stats['fluidity_max']:.4f}]")
        
        # Nettoyer
        import os
        os.remove(temp_config)
        
        return stats
        
    except Exception as e:
        print(f"  ❌ Erreur lecture CSV : {e}")
        return None

def main():
    """Test principal"""
    print("=== TEST FLUIDITÉ SELON MODES GAMMA ===")
    print(f"Date : {datetime.now()}")
    
    # Modes à tester
    gamma_modes = [
        'static',
        'dynamic', 
        'sigmoid_up',
        'sigmoid_down',
        'sigmoid_oscillating',
        'sinusoidal'
    ]
    
    results = []
    
    for mode in gamma_modes:
        stats = run_simulation_with_gamma_mode(mode)
        if stats:
            results.append(stats)
    
    # Créer un DataFrame avec les résultats
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n📊 RÉSUMÉ DES RÉSULTATS")
        print("=" * 60)
        print(df_results.to_string(index=False))
        
        # Calculer la variation de fluidité
        fluidity_range = df_results['fluidity_mean'].max() - df_results['fluidity_mean'].min()
        print(f"\n🎯 Variation de fluidité entre modes : {fluidity_range:.4f}")
        print(f"   (ancienne formule : ~0.00000007)")
        print(f"   Amélioration : {fluidity_range/0.00000007:.0f}x")
        
        # Sauvegarder les résultats
        output_file = f'fluidity_gamma_test_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Résultats sauvegardés : {output_file}")
        
        # Identifier le mode le plus fluide
        best_mode = df_results.loc[df_results['fluidity_mean'].idxmax(), 'gamma_mode']
        worst_mode = df_results.loc[df_results['fluidity_mean'].idxmin(), 'gamma_mode']
        
        print(f"\n🏆 Mode le plus fluide : {best_mode}")
        print(f"❌ Mode le moins fluide : {worst_mode}")

if __name__ == '__main__':
    main() 