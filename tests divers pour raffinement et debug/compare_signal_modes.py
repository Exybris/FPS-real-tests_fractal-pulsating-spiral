#!/usr/bin/env python3
"""
compare_signal_modes.py - Compare les modes de signal FPS
========================================================
Ce script lance deux simulations identiques sauf pour le mode de signal :
1. Mode "extended" (avec régulation G)
2. Mode "simple" (sans régulation G)

Il analyse ensuite les différences pour diagnostiquer pourquoi S(t) devient plat.
"""

import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import sys

def create_config_with_debug(base_config_path, signal_mode, output_dir):
    """Crée une configuration avec le mode de signal spécifié et le debug activé."""
    # Charger la config de base
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Modifier les paramètres
    config['system']['signal_mode'] = signal_mode
    config['debug'] = {'log_detailed': True}
    
    # Sauvegarder la nouvelle config
    config_path = os.path.join(output_dir, f'config_{signal_mode}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def run_simulation(config_path):
    """Lance une simulation avec la configuration donnée."""
    # Capturer l'état avant la simulation
    before_dirs = set(os.listdir('fps_pipeline_output'))
    
    cmd = [sys.executable, 'main.py', 'run', '--config', config_path]
    print(f"Lancement : {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur lors de la simulation :")
        print(result.stderr)
        return None
    
    # Afficher la sortie pour debug
    print("Sortie de la simulation :")
    print(result.stdout[-500:])  # Derniers 500 caractères
    
    # Trouver le nouveau dossier créé
    after_dirs = set(os.listdir('fps_pipeline_output'))
    new_dirs = after_dirs - before_dirs
    
    if new_dirs:
        run_id = list(new_dirs)[0]
        print(f"Nouveau run détecté : {run_id}")
        return run_id
    
    # Sinon, chercher dans la sortie
    for line in result.stdout.split('\n'):
        if 'run_' in line and ('Created' in line or 'Output' in line or 'Run ID' in line):
            import re
            match = re.search(r'run_\d{8}_\d{6}', line)
            if match:
                return match.group(0)
    
    return None

def load_and_compare_results(run_id_extended, run_id_simple):
    """Charge et compare les résultats des deux runs."""
    # Chemins des fichiers
    path_extended = f'fps_pipeline_output/{run_id_extended}/logs/metrics_{run_id_extended}.csv'
    path_simple = f'fps_pipeline_output/{run_id_simple}/logs/metrics_{run_id_simple}.csv'
    
    debug_extended = f'fps_pipeline_output/{run_id_extended}/logs/debug_detailed_{run_id_extended}.csv'
    debug_simple = f'fps_pipeline_output/{run_id_simple}/logs/debug_detailed_{run_id_simple}.csv'
    
    # Charger les métriques principales
    df_extended = pd.read_csv(path_extended)
    df_simple = pd.read_csv(path_simple)
    
    # Charger les logs détaillés si disponibles
    df_debug_extended = None
    df_debug_simple = None
    
    if os.path.exists(debug_extended):
        df_debug_extended = pd.read_csv(debug_extended)
    if os.path.exists(debug_simple):
        df_debug_simple = pd.read_csv(debug_simple)
    
    return {
        'extended': {'metrics': df_extended, 'debug': df_debug_extended},
        'simple': {'metrics': df_simple, 'debug': df_debug_simple}
    }

def plot_comparison(data, output_dir):
    """Crée des graphiques de comparaison."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Comparaison modes Extended vs Simple', fontsize=16)
    
    # 1. Signal S(t)
    ax = axes[0, 0]
    ax.plot(data['extended']['metrics']['t'], data['extended']['metrics']['S(t)'], 
            label='Extended', color='blue', alpha=0.7)
    ax.plot(data['simple']['metrics']['t'], data['simple']['metrics']['S(t)'], 
            label='Simple', color='red', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('S(t)')
    ax.set_title('Signal Global S(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Coefficient C(t)
    ax = axes[0, 1]
    ax.plot(data['extended']['metrics']['t'], data['extended']['metrics']['C(t)'], 
            label='Extended', color='blue', alpha=0.7)
    ax.plot(data['simple']['metrics']['t'], data['simple']['metrics']['C(t)'], 
            label='Simple', color='red', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('C(t)')
    ax.set_title('Coefficient d\'accord C(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Effort
    ax = axes[1, 0]
    ax.plot(data['extended']['metrics']['t'], data['extended']['metrics']['effort(t)'], 
            label='Extended', color='blue', alpha=0.7)
    ax.plot(data['simple']['metrics']['t'], data['simple']['metrics']['effort(t)'], 
            label='Simple', color='red', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('effort(t)')
    ax.set_title('Effort interne')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Erreur moyenne
    ax = axes[1, 1]
    ax.plot(data['extended']['metrics']['t'], data['extended']['metrics']['mean_abs_error'], 
            label='Extended', color='blue', alpha=0.7)
    ax.plot(data['simple']['metrics']['t'], data['simple']['metrics']['mean_abs_error'], 
            label='Simple', color='red', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('mean_abs_error')
    ax.set_title('Erreur absolue moyenne')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Si debug disponible : G values pour mode extended
    if data['extended']['debug'] is not None:
        ax = axes[2, 0]
        df_debug = data['extended']['debug']
        
        # Tracer G pour chaque strate
        for n in range(5):  # Supposons 5 strates
            if f'G_{n}' in df_debug.columns:
                ax.plot(df_debug['t'], df_debug[f'G_{n}'], 
                       label=f'G_{n}', alpha=0.7)
        
        ax.set_xlabel('Temps')
        ax.set_ylabel('G(erreur)')
        ax.set_title('Valeurs de régulation G (Extended)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Si debug disponible : Amplitudes An
    if data['extended']['debug'] is not None:
        ax = axes[2, 1]
        df_debug = data['extended']['debug']
        
        # Tracer An pour chaque strate
        for n in range(5):  # Supposons 5 strates
            if f'An_{n}' in df_debug.columns:
                ax.plot(df_debug['t'], df_debug[f'An_{n}'], 
                       label=f'A_{n}', alpha=0.7)
        
        ax.set_xlabel('Temps')
        ax.set_ylabel('An(t)')
        ax.set_title('Amplitudes An(t) (Extended)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_signal_modes.png'), dpi=150)
    plt.close()
    
    # Graphique supplémentaire pour le diagnostic détaillé
    if data['extended']['debug'] is not None:
        plot_detailed_diagnosis(data['extended']['debug'], output_dir, 'extended')

def plot_detailed_diagnosis(df_debug, output_dir, mode_name):
    """Crée des graphiques détaillés pour le diagnostic."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Diagnostic détaillé - Mode {mode_name}', fontsize=16)
    
    # 1. Input In(t)
    ax = axes[0, 0]
    for n in range(5):
        if f'In_{n}' in df_debug.columns:
            ax.plot(df_debug['t'], df_debug[f'In_{n}'], 
                   label=f'In_{n}', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('In(t)')
    ax.set_title('Input contextuel In(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Erreurs En - On
    ax = axes[0, 1]
    for n in range(5):
        if f'erreur_{n}' in df_debug.columns:
            ax.plot(df_debug['t'], df_debug[f'erreur_{n}'], 
                   label=f'erreur_{n}', alpha=0.7)
    ax.set_xlabel('Temps')
    ax.set_ylabel('En - On')
    ax.set_title('Erreurs En(t) - On(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Relation erreur vs G
    ax = axes[1, 0]
    for n in range(5):
        if f'erreur_{n}' in df_debug.columns and f'G_{n}' in df_debug.columns:
            ax.scatter(df_debug[f'erreur_{n}'], df_debug[f'G_{n}'], 
                      label=f'Strate {n}', alpha=0.5, s=10)
    ax.set_xlabel('Erreur (En - On)')
    ax.set_ylabel('G(erreur)')
    ax.set_title('Fonction de régulation G')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. S(t) vs somme des G
    ax = axes[1, 1]
    # Calculer la somme des G
    G_cols = [f'G_{n}' for n in range(5) if f'G_{n}' in df_debug.columns]
    if G_cols:
        G_sum = df_debug[G_cols].sum(axis=1)
        ax.scatter(G_sum, df_debug['S_t'], alpha=0.5, s=10)
        ax.set_xlabel('Somme des G')
        ax.set_ylabel('S(t)')
        ax.set_title('Impact de G sur S(t)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'diagnosis_detailed_{mode_name}.png'), dpi=150)
    plt.close()

def analyze_results(data, output_dir):
    """Analyse les résultats et génère un rapport."""
    report = []
    report.append("=== RAPPORT D'ANALYSE COMPARATIVE ===\n")
    report.append(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Statistiques S(t)
    report.append("\n1. ANALYSE DU SIGNAL S(t)")
    for mode in ['extended', 'simple']:
        S_values = data[mode]['metrics']['S(t)'].values
        report.append(f"\n{mode.upper()}:")
        report.append(f"  - Moyenne : {np.mean(S_values):.4f}")
        report.append(f"  - Écart-type : {np.std(S_values):.4f}")
        report.append(f"  - Min/Max : [{np.min(S_values):.4f}, {np.max(S_values):.4f}]")
        
        # Détecter quand S devient plat
        threshold = 0.01
        flat_indices = np.where(np.abs(S_values) < threshold)[0]
        if len(flat_indices) > 0:
            first_flat = data[mode]['metrics']['t'].iloc[flat_indices[0]]
            report.append(f"  - S devient plat (|S| < {threshold}) à t = {first_flat:.2f}")
            report.append(f"  - Proportion plate : {len(flat_indices)/len(S_values)*100:.1f}%")
    
    # Analyse du debug si disponible
    if data['extended']['debug'] is not None:
        report.append("\n\n2. ANALYSE DÉTAILLÉE (MODE EXTENDED)")
        df_debug = data['extended']['debug']
        
        # Analyser les G values
        report.append("\nValeurs de régulation G:")
        for n in range(5):
            if f'G_{n}' in df_debug.columns:
                G_values = df_debug[f'G_{n}'].values
                report.append(f"  Strate {n}:")
                report.append(f"    - Moyenne : {np.mean(G_values):.4f}")
                report.append(f"    - % proche de 0 : {np.sum(np.abs(G_values) < 0.01)/len(G_values)*100:.1f}%")
        
        # Analyser les amplitudes
        report.append("\nAmplitudes An(t):")
        for n in range(5):
            if f'An_{n}' in df_debug.columns:
                A_values = df_debug[f'An_{n}'].values
                report.append(f"  Strate {n}:")
                report.append(f"    - Moyenne : {np.mean(A_values):.4f}")
                report.append(f"    - Min : {np.min(A_values):.4f}")
    
    # Diagnostic
    report.append("\n\n3. DIAGNOSTIC")
    
    # Vérifier si le problème vient de G
    if data['extended']['debug'] is not None:
        G_cols = [f'G_{n}' for n in range(5) if f'G_{n}' in df_debug.columns]
        if G_cols:
            G_mean = df_debug[G_cols].mean().mean()
            if G_mean < 0.1:
                report.append("⚠️  Les valeurs de G sont très proches de 0")
                report.append("   → La régulation annule le signal dans le mode extended")
                report.append("   → C'est pourquoi S(t) devient plat")
    
    # Comparer les deux modes
    S_extended_mean = np.mean(data['extended']['metrics']['S(t)'].values)
    S_simple_mean = np.mean(data['simple']['metrics']['S(t)'].values)
    
    if S_simple_mean > 10 * S_extended_mean:
        report.append("\n✓ Le mode 'simple' maintient un signal actif")
        report.append("✓ Le mode 'extended' supprime le signal via G")
        report.append("→ Le problème vient bien de la multiplication par G dans compute_S")
    
    # Sauvegarder le rapport
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    print(f"\nRapport sauvegardé : {report_path}")

def main():
    """Fonction principale."""
    print("=== COMPARAISON DES MODES DE SIGNAL FPS ===\n")
    
    # Créer un dossier pour les résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'comparison_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer les configurations
    print("1. Création des configurations...")
    config_extended = create_config_with_debug('config.json', 'extended', output_dir)
    config_simple = create_config_with_debug('config.json', 'simple', output_dir)
    
    # Lancer les simulations
    print("\n2. Lancement des simulations...")
    print("\n--- Simulation 1: Mode EXTENDED ---")
    run_id_extended = run_simulation(config_extended)
    if not run_id_extended:
        print("Erreur lors de la simulation extended")
        return
    print(f"✓ Run ID: {run_id_extended}")
    
    print("\n--- Simulation 2: Mode SIMPLE ---")
    run_id_simple = run_simulation(config_simple)
    if not run_id_simple:
        print("Erreur lors de la simulation simple")
        return
    print(f"✓ Run ID: {run_id_simple}")
    
    # Charger et comparer les résultats
    print("\n3. Chargement des résultats...")
    data = load_and_compare_results(run_id_extended, run_id_simple)
    
    # Créer les graphiques
    print("\n4. Création des graphiques...")
    plot_comparison(data, output_dir)
    
    # Analyser et générer le rapport
    print("\n5. Analyse des résultats...")
    analyze_results(data, output_dir)
    
    print(f"\n✓ Analyse terminée. Résultats dans : {output_dir}/")
    print("  - comparison_signal_modes.png : Comparaison des métriques")
    print("  - diagnosis_detailed_*.png : Diagnostic détaillé")
    print("  - analysis_report.txt : Rapport d'analyse")

if __name__ == "__main__":
    main() 