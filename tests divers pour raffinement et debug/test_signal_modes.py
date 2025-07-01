#!/usr/bin/env python3
"""
test_signal_modes.py - Test rapide des modes de signal
======================================================
Version simplifiée pour tester directement les deux modes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Import direct des modules FPS
import simulate
import init

def test_both_modes():
    """Test les deux modes de signal et compare les résultats."""
    
    # Charger la config de base
    with open('config.json', 'r') as f:
        config_base = json.load(f)
    
    # Créer un dossier pour les résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'test_comparison_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Test 1: Mode Extended
    print("\n=== TEST MODE EXTENDED ===")
    config_extended = config_base.copy()
    config_extended['system']['signal_mode'] = 'extended'
    config_extended['debug'] = {'log_detailed': True}
    
    # Sauvegarder la config
    config_path_ext = os.path.join(output_dir, 'config_extended.json')
    with open(config_path_ext, 'w') as f:
        json.dump(config_extended, f, indent=2)
    
    # Lancer la simulation
    try:
        result_ext = simulate.run_simulation(config_path_ext, mode="FPS")
        results['extended'] = result_ext
        print(f"✓ Simulation Extended terminée")
    except Exception as e:
        print(f"❌ Erreur mode Extended : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Mode Simple
    print("\n=== TEST MODE SIMPLE ===")
    config_simple = config_base.copy()
    config_simple['system']['signal_mode'] = 'simple'
    config_simple['debug'] = {'log_detailed': True}
    
    # Sauvegarder la config
    config_path_simple = os.path.join(output_dir, 'config_simple.json')
    with open(config_path_simple, 'w') as f:
        json.dump(config_simple, f, indent=2)
    
    # Lancer la simulation
    try:
        result_simple = simulate.run_simulation(config_path_simple, mode="FPS")
        results['simple'] = result_simple
        print(f"✓ Simulation Simple terminée")
    except Exception as e:
        print(f"❌ Erreur mode Simple : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyser et comparer
    analyze_and_plot(results, output_dir)

def analyze_and_plot(results, output_dir):
    """Analyse et trace les résultats."""
    
    print("\n=== ANALYSE DES RÉSULTATS ===")
    
    # Extraire les données
    S_extended = np.array(results['extended']['S_history'])
    S_simple = np.array(results['simple']['S_history'])
    
    # Créer le temps
    dt = 0.1  # Par défaut dans config
    t_extended = np.arange(len(S_extended)) * dt
    t_simple = np.arange(len(S_simple)) * dt
    
    # Statistiques
    print(f"\nMode EXTENDED:")
    print(f"  - S(t) moyen : {np.mean(S_extended):.4f}")
    print(f"  - S(t) std : {np.std(S_extended):.4f}")
    print(f"  - S(t) min/max : [{np.min(S_extended):.4f}, {np.max(S_extended):.4f}]")
    
    # Détecter quand S devient plat
    threshold = 0.01
    flat_idx_ext = np.where(np.abs(S_extended) < threshold)[0]
    if len(flat_idx_ext) > 0:
        print(f"  - S devient plat à t = {t_extended[flat_idx_ext[0]]:.1f}")
        print(f"  - Proportion plate : {len(flat_idx_ext)/len(S_extended)*100:.1f}%")
    
    print(f"\nMode SIMPLE:")
    print(f"  - S(t) moyen : {np.mean(S_simple):.4f}")
    print(f"  - S(t) std : {np.std(S_simple):.4f}")
    print(f"  - S(t) min/max : [{np.min(S_simple):.4f}, {np.max(S_simple):.4f}]")
    
    flat_idx_simple = np.where(np.abs(S_simple) < threshold)[0]
    if len(flat_idx_simple) > 0:
        print(f"  - S devient plat à t = {t_simple[flat_idx_simple[0]]:.1f}")
        print(f"  - Proportion plate : {len(flat_idx_simple)/len(S_simple)*100:.1f}%")
    
    # Graphiques
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Signal S(t) complet
    ax = axes[0, 0]
    ax.plot(t_extended, S_extended, label='Extended', color='blue', alpha=0.7)
    ax.plot(t_simple, S_simple, label='Simple', color='red', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('S(t)')
    ax.set_title('Signal Global S(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Zoom sur la transition
    ax = axes[0, 1]
    # Zoomer autour de T/2
    T_half = len(S_extended) // 2
    zoom_start = max(0, T_half - 100)
    zoom_end = min(len(S_extended), T_half + 100)
    
    ax.plot(t_extended[zoom_start:zoom_end], S_extended[zoom_start:zoom_end], 
            label='Extended', color='blue', alpha=0.7, linewidth=2)
    ax.plot(t_simple[zoom_start:zoom_end], S_simple[zoom_start:zoom_end], 
            label='Simple', color='red', alpha=0.7, linewidth=2)
    ax.set_xlabel('Temps')
    ax.set_ylabel('S(t)')
    ax.set_title('Zoom sur la transition (autour de T/2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Histogramme des valeurs
    ax = axes[1, 0]
    ax.hist(S_extended, bins=50, alpha=0.5, label='Extended', color='blue', density=True)
    ax.hist(S_simple, bins=50, alpha=0.5, label='Simple', color='red', density=True)
    ax.set_xlabel('Valeur de S(t)')
    ax.set_ylabel('Densité')
    ax.set_title('Distribution des valeurs de S(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Évolution de |S(t)|
    ax = axes[1, 1]
    ax.semilogy(t_extended, np.abs(S_extended) + 1e-10, label='|S(t)| Extended', color='blue', alpha=0.7)
    ax.semilogy(t_simple, np.abs(S_simple) + 1e-10, label='|S(t)| Simple', color='red', alpha=0.7)
    ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label=f'Seuil plat ({threshold})')
    ax.set_xlabel('Temps')
    ax.set_ylabel('|S(t)| (échelle log)')
    ax.set_title('Amplitude absolue de S(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_simple_vs_extended.png'), dpi=150)
    plt.close()
    
    # Charger et analyser les logs détaillés si disponibles
    analyze_detailed_logs(results, output_dir)
    
    print(f"\n✓ Graphiques sauvegardés dans : {output_dir}/")

def analyze_detailed_logs(results, output_dir):
    """Analyse les logs détaillés si disponibles."""
    
    # Chercher les fichiers de debug
    for mode in ['extended', 'simple']:
        if mode not in results:
            continue
            
        run_id = results[mode].get('run_id', '')
        debug_file = f'fps_pipeline_output/{run_id}/logs/debug_detailed_{run_id}.csv'
        
        if os.path.exists(debug_file):
            print(f"\n=== ANALYSE DÉTAILLÉE {mode.upper()} ===")
            
            df = pd.read_csv(debug_file)
            
            # Analyser les valeurs de G
            G_cols = [col for col in df.columns if col.startswith('G_')]
            if G_cols:
                print("\nValeurs de régulation G:")
                for col in G_cols:
                    G_values = df[col].values
                    print(f"  {col}:")
                    print(f"    - Moyenne : {np.mean(G_values):.6f}")
                    print(f"    - % proche de 0 (<0.01) : {np.sum(np.abs(G_values) < 0.01)/len(G_values)*100:.1f}%")
                    print(f"    - Min/Max : [{np.min(G_values):.6f}, {np.max(G_values):.6f}]")
            
            # Créer un graphique détaillé
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Analyse détaillée - Mode {mode.upper()}', fontsize=14)
            
            # 1. Évolution de G
            ax = axes[0, 0]
            for col in G_cols[:5]:  # Max 5 strates
                ax.plot(df['t'], df[col], label=col, alpha=0.7)
            ax.set_xlabel('Temps')
            ax.set_ylabel('G(erreur)')
            ax.set_title('Valeurs de régulation G')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Input In
            ax = axes[0, 1]
            In_cols = [col for col in df.columns if col.startswith('In_')]
            if In_cols:
                # Tracer seulement la première strate pour clarté
                ax.plot(df['t'], df[In_cols[0]], label='In(t)', color='green')
                ax.set_xlabel('Temps')
                ax.set_ylabel('In(t)')
                ax.set_title('Input contextuel')
                ax.grid(True, alpha=0.3)
            
            # 3. Erreurs
            ax = axes[1, 0]
            err_cols = [col for col in df.columns if col.startswith('erreur_')]
            for col in err_cols[:5]:
                ax.plot(df['t'], df[col], label=col.replace('erreur_', 'e'), alpha=0.7)
            ax.set_xlabel('Temps')
            ax.set_ylabel('En - On')
            ax.set_title('Erreurs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Relation S(t) vs moyenne(G)
            ax = axes[1, 1]
            if G_cols and 'S_t' in df.columns:
                G_mean = df[G_cols].mean(axis=1)
                ax.scatter(G_mean, df['S_t'], alpha=0.5, s=10)
                ax.set_xlabel('Moyenne des G')
                ax.set_ylabel('S(t)')
                ax.set_title('Impact de G sur S(t)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'detailed_analysis_{mode}.png'), dpi=150)
            plt.close()

if __name__ == "__main__":
    print("=== TEST COMPARATIF DES MODES DE SIGNAL ===")
    test_both_modes() 