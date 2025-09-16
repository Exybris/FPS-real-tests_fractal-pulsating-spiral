"""
visualize_individual.py - Visualisation des dynamiques individuelles FPS
========================================================================

Module dédié à l'analyse et visualisation des comportements individuels
de chaque strate dans le système FPS.

Fonctionnalités :
- Graphiques individuels par strate (An_i, On_i, En_i, erreur_i, G_i)
- Comparaisons inter-strates
- Matrices de corrélation
- Évolution temporelle détaillée
- Export des données individuelles

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

# Configuration matplotlib pour de beaux graphiques
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True

def load_individual_data(debug_file_path: str) -> pd.DataFrame:
    """
    Charge les données debug détaillées avec toutes les métriques individuelles.
    """
    try:
        df = pd.read_csv(debug_file_path)
        print(f"✅ Données chargées: {len(df)} points temporels, {len(df.columns)} colonnes")
        return df
    except Exception as e:
        print(f"❌ Erreur chargement {debug_file_path}: {e}")
        return pd.DataFrame()

def plot_individual_amplitudes(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Visualise l'évolution des amplitudes An_i pour chaque strate.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Couleurs distinctes pour chaque strate
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Graphique global
    ax_global = axes[0]
    for i in range(5):
        col = f'An_{i}'
        if col in df.columns:
            ax_global.plot(df['t'], df[col], label=f'Strate {i}', 
                          color=colors[i], linewidth=2, alpha=0.8)
    
    ax_global.set_title('🎵 Amplitudes An(t) - Vue globale', fontsize=14, fontweight='bold')
    ax_global.set_xlabel('Temps t')
    ax_global.set_ylabel('Amplitude An(t)')
    ax_global.legend()
    ax_global.grid(True, alpha=0.3)
    
    # Graphiques individuels
    for i in range(5):
        ax = axes[i + 1]
        col = f'An_{i}'
        
        if col in df.columns:
            values = df[col].values
            
            # Ligne principale
            ax.plot(df['t'], values, color=colors[i], linewidth=2, label=f'An_{i}(t)')
            
            # Statistiques
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Lignes de référence
            ax.axhline(mean_val, color=colors[i], linestyle='--', alpha=0.6, label=f'Moyenne: {mean_val:.3f}')
            ax.fill_between(df['t'], mean_val - std_val, mean_val + std_val, 
                           color=colors[i], alpha=0.2, label=f'±σ: {std_val:.3f}')
            
            ax.set_title(f'Strate {i} - An_{i}(t)', fontweight='bold')
            ax.set_xlabel('Temps t')
            ax.set_ylabel(f'An_{i}(t)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Données An_{i}\nnon disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Strate {i} - Données manquantes')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'individual_amplitudes_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Graphique amplitudes sauvé: {filename}")
    return filename

def plot_signature_analysis(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Analyse les SIGNATURES PHASIQUES - La voix propre de chaque strate selon Andréa !
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Signatures phasiques - empreintes
    ax_sig = axes[0, 0]
    signatures = []
    for i in range(5):
        phi_col = f'phi_{i}'
        if phi_col in df.columns:
            phi_vals = df[phi_col].values
            signature = phi_vals[0]  # Empreinte initiale
            signatures.append(signature)
            
    if signatures:
        # Représentation des signatures en étoile
        strates = list(range(5))
        
        ax_sig.scatter(strates, signatures, s=100, c=colors, alpha=0.8)
        for i, sig in enumerate(signatures):
            ax_sig.plot([i, i], [0, sig], '--', color=colors[i], alpha=0.5)
            ax_sig.text(i, sig + 0.02, f'{sig:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax_sig.set_title('🎭 Signatures Phasiques\\nEmpreintes individuelles', fontweight='bold')
        ax_sig.set_xlabel('Strates')
        ax_sig.set_ylabel('Signature φₙ(0) [rad]')
        ax_sig.set_xticks(strates)
        ax_sig.set_xticklabels([f'S{i}' for i in strates])
        ax_sig.grid(True, alpha=0.3)
        
    # 2. Danses autour des signatures
    ax_dance = axes[0, 1]
    for i in range(5):
        phi_col = f'phi_{i}'
        if phi_col in df.columns:
            phi_vals = df[phi_col].values
            signature = phi_vals[0]
            
            # Écart par rapport à la signature
            dance = phi_vals - signature
            ax_dance.plot(df['t'], dance, label=f'Danse S{i}', 
                         color=colors[i], alpha=0.8, linewidth=1.5)
    
    ax_dance.set_title('💃 Danses autour des Signatures\\nÉcarts φₙ(t) - φₙ(0)', fontweight='bold')
    ax_dance.set_xlabel('Temps t')
    ax_dance.set_ylabel('Écart signature [rad]')
    ax_dance.legend()
    ax_dance.grid(True, alpha=0.3)
    
    # 3. Corrélations r(t) ↔ C(t)
    ax_corr = axes[0, 2]
    if 'r_t' in df.columns:
        # Calculer C(t) approximatif
        phase_cols = [f'phi_{i}' for i in range(5)]
        if all(col in df.columns for col in phase_cols):
            C_approx = []
            for idx, row in df.iterrows():
                phi_vals = [row[col] for col in phase_cols]
                # Cohérence basée sur écart-type des phases
                C_val = 1.0 / (1.0 + np.std(phi_vals))  # Plus les phases sont alignées, plus C est élevé
                C_approx.append(C_val)
            
            ax_corr.scatter(df['r_t'], C_approx, alpha=0.6, c=df['t'], cmap='viridis')
            ax_corr.set_xlabel('r(t) - Ratio spiralé')
            ax_corr.set_ylabel('C(t) - Cohérence approx.')
            ax_corr.set_title('🌀 r(t) ↔ C(t)\\nLien Spirale-Cohérence', fontweight='bold')
            
            # Corrélation
            corr = np.corrcoef(df['r_t'], C_approx)[0,1]
            ax_corr.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax_corr.transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat'))
            ax_corr.grid(True, alpha=0.3)
    
    # 4. Affinités phasiques (distances signatures)
    ax_affinity = axes[1, 0]
    if signatures:
        # Matrice des distances entre signatures
        N = len(signatures)
        affinity_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                # Distance angulaire circulaire
                diff = abs(signatures[i] - signatures[j])
                affinity_matrix[i, j] = min(diff, 2*np.pi - diff)
        
        im = ax_affinity.imshow(affinity_matrix, cmap='RdYlBu_r')
        ax_affinity.set_title('🔗 Affinités Phasiques\\nDistances entre signatures', fontweight='bold')
        
        # Labels
        ax_affinity.set_xticks(range(N))
        ax_affinity.set_yticks(range(N))
        ax_affinity.set_xticklabels([f'S{i}' for i in range(N)])
        ax_affinity.set_yticklabels([f'S{i}' for i in range(N)])
        
        # Valeurs dans les cellules
        for i in range(N):
            for j in range(N):
                ax_affinity.text(j, i, f'{affinity_matrix[i,j]:.2f}', 
                               ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax_affinity, label='Distance [rad]')
    
    # 5. Évolution dynamisme par strate
    ax_dynamism = axes[1, 1]
    dynamism_data = []
    labels = []
    for i in range(5):
        phi_col = f'phi_{i}'
        if phi_col in df.columns:
            phi_vals = df[phi_col].values
            signature = phi_vals[0]
            dance = phi_vals - signature
            dynamism = np.std(dance)  # Dynamisme = variabilité de la danse
            dynamism_data.append(dynamism)
            labels.append(f'S{i}')
    
    if dynamism_data:
        bars = ax_dynamism.bar(labels, dynamism_data, color=colors[:len(dynamism_data)], alpha=0.7)
        ax_dynamism.set_title('🌊 Dynamisme par Strate\\nVariabilité de la danse', fontweight='bold')
        ax_dynamism.set_ylabel('Dynamisme σ(danse)')
        
        # Valeurs sur les barres
        for bar, val in zip(bars, dynamism_data):
            ax_dynamism.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{val:.3f}', ha='center', va='bottom')
    
    # 6. Timeline des signatures évolutives
    ax_timeline = axes[1, 2]
    for i in range(5):
        phi_col = f'phi_{i}'
        if phi_col in df.columns:
            phi_vals = df[phi_col].values
            # Signature évolutive (moyenne mobile)
            window = min(10, len(phi_vals)//4)
            if window > 1:
                signature_evolution = pd.Series(phi_vals).rolling(window=window, center=True).mean()
                ax_timeline.plot(df['t'], signature_evolution, label=f'Sig{i}', 
                               color=colors[i], alpha=0.8, linewidth=2)
    
    ax_timeline.set_title('📈 Évolution des Signatures\\nTendances temporelles', fontweight='bold')
    ax_timeline.set_xlabel('Temps t')
    ax_timeline.set_ylabel('Signature évolutive')
    ax_timeline.legend()
    ax_timeline.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'signature_analysis_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analyse signatures sauvée: {filename}")
    return filename

def plot_spiral_dynamics(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Visualise la dynamique spiralée : r(t) et fréquences fn(t) - CŒUR du système FPS !
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Ratio spiralé r(t)
    ax_rt = axes[0, 0]
    if 'r_t' in df.columns:
        r_vals = df['r_t'].values
        ax_rt.plot(df['t'], r_vals, color='gold', linewidth=3, alpha=0.8, label='r(t)')
        
        # Analyser variation
        r_min, r_max = np.min(r_vals), np.max(r_vals)
        r_range = r_max - r_min
        
        ax_rt.axhline(1.618, color='red', linestyle='--', alpha=0.7, label='φ (nombre d\'or)')
        ax_rt.fill_between(df['t'], r_min, r_max, alpha=0.2, color='gold')
        
        ax_rt.set_title(f'🌀 Ratio Spiralé r(t)\nVariation: {r_range:.6f}', fontweight='bold')
        ax_rt.set_xlabel('Temps t')
        ax_rt.set_ylabel('r(t)')
        ax_rt.legend()
        ax_rt.grid(True, alpha=0.3)
    else:
        ax_rt.text(0.5, 0.5, 'r(t) non disponible', ha='center', va='center', transform=ax_rt.transAxes)
        ax_rt.set_title('⚠️ r(t) manquant')
    
    # 2. Fréquences fn(t) - vue globale
    ax_freq = axes[0, 1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(5):
        fn_col = f'fn_{i}'
        if fn_col in df.columns:
            ax_freq.plot(df['t'], df[fn_col], label=f'f_{i}', 
                        color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax_freq.set_title('📊 Fréquences fn(t) - Vue globale', fontweight='bold')
    ax_freq.set_xlabel('Temps t')
    ax_freq.set_ylabel('Fréquence fn(t)')
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.3)
    
    # 3. Contrainte spiralée : ratios fn+1/fn vs r(t)
    ax_ratios = axes[1, 0]
    if 'r_t' in df.columns:
        ratios_data = []
        for i in range(4):  # f1/f0, f2/f1, f3/f2, f4/f3
            fn_col = f'fn_{i}'
            fn_next_col = f'fn_{i+1}'
            if fn_col in df.columns and fn_next_col in df.columns:
                ratios = df[fn_next_col] / df[fn_col]
                ratios_data.append(ratios.values)
                ax_ratios.plot(df['t'], ratios, label=f'f{i+1}/f{i}', 
                              color=colors[i], alpha=0.7, linewidth=1)
        
        # Ligne r(t) cible
        ax_ratios.plot(df['t'], df['r_t'], color='gold', linewidth=3, 
                      label='r(t) cible', linestyle='--')
        
        ax_ratios.set_title('⚖️ Contrainte Spiralée : fn+1/fn vs r(t)', fontweight='bold')
        ax_ratios.set_xlabel('Temps t')
        ax_ratios.set_ylabel('Ratio fréquences')
        ax_ratios.legend()
        ax_ratios.grid(True, alpha=0.3)
    else:
        ax_ratios.text(0.5, 0.5, 'Données insuffisantes', ha='center', va='center', transform=ax_ratios.transAxes)
    
    # 4. Distribution des fréquences
    ax_dist = axes[1, 1]
    freq_data = []
    labels = []
    for i in range(5):
        fn_col = f'fn_{i}'
        if fn_col in df.columns:
            freq_data.append(df[fn_col].values)
            labels.append(f'f_{i}')
    
    if freq_data:
        ax_dist.boxplot(freq_data, tick_labels=labels)
        ax_dist.set_title('📦 Distribution Fréquences', fontweight='bold')
        ax_dist.set_xlabel('Strates')
        ax_dist.set_ylabel('Fréquence')
        ax_dist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'spiral_dynamics_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Graphique dynamique spiralée sauvé: {filename}")
    return filename

def plot_individual_phases(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Visualise l'évolution des phases φₙ(t) pour chaque strate - CLÉ de la dynamique FPS !
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Vue globale - toutes les phases
    ax_global = axes[0]
    for i in range(5):
        phi_col = f'phi_{i}'
        if phi_col in df.columns:
            ax_global.plot(df['t'], df[phi_col], label=f'φ_{i}', 
                          color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax_global.set_title('🌀 Phases φₙ(t) - Vue globale', fontsize=14, fontweight='bold')
    ax_global.set_xlabel('Temps t')
    ax_global.set_ylabel('Phase φₙ(t) [rad]')
    ax_global.legend()
    ax_global.grid(True, alpha=0.3)
    
    # Phases individuelles avec analyse
    for i in range(5):
        ax = axes[i + 1]
        phi_col = f'phi_{i}'
        
        if phi_col in df.columns:
            values = df[phi_col].values
            
            # Ligne principale
            ax.plot(df['t'], values, color=colors[i], linewidth=2, 
                   label=f'φ_{i}(t)', alpha=0.8)
            
            # Analyse des excursions (phases > 2π)
            excursions = values[np.abs(values) > 2*np.pi]
            if len(excursions) > 0:
                max_excursion = np.max(np.abs(values))
                ax.axhline(2*np.pi, color='red', linestyle='--', alpha=0.5, label='2π')
                ax.axhline(-2*np.pi, color='red', linestyle='--', alpha=0.5, label='-2π')
                ax.text(0.02, 0.98, f'Max excursion: {max_excursion:.2f}rad', 
                       transform=ax.transAxes, va='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Statistiques
            std_val = np.std(values)
            range_val = np.max(values) - np.min(values)
            
            ax.set_title(f'Strate {i} - φ_{i}(t)\nσ={std_val:.3f}, range={range_val:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Temps t')
            ax.set_ylabel(f'φ_{i}(t) [rad]')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Données φ_{i}\nnon disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Strate {i} - Données manquantes')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'individual_phases_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Graphique phases sauvé: {filename}")
    return filename

def plot_individual_outputs(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Visualise les sorties observées On_i vs attendues En_i pour chaque strate.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Vue globale - toutes les sorties observées
    ax_global = axes[0]
    for i in range(5):
        On_col = f'On_{i}'
        if On_col in df.columns:
            ax_global.plot(df['t'], df[On_col], label=f'On_{i}', 
                          color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax_global.set_title('🎯 Sorties observées On(t) - Vue globale', fontsize=14, fontweight='bold')
    ax_global.set_xlabel('Temps t')
    ax_global.set_ylabel('Sortie On(t)')
    ax_global.legend()
    ax_global.grid(True, alpha=0.3)
    
    # Comparaisons individuelles On vs En
    for i in range(5):
        ax = axes[i + 1]
        On_col = f'On_{i}'
        En_col = f'En_{i}'
        
        if On_col in df.columns and En_col in df.columns:
            # Sorties observées et attendues
            ax.plot(df['t'], df[On_col], color=colors[i], linewidth=2, 
                   label=f'On_{i} (observé)', alpha=0.8)
            ax.plot(df['t'], df[En_col], color=colors[i], linestyle='--', linewidth=2,
                   label=f'En_{i} (attendu)', alpha=0.6)
            
            # Zone d'erreur
            erreur_col = f'erreur_{i}'
            if erreur_col in df.columns:
                erreur = df[erreur_col].values
                ax.fill_between(df['t'], df[On_col], df[En_col], 
                               color='red', alpha=0.2, label=f'Erreur (σ={np.std(erreur):.3f})')
            
            ax.set_title(f'Strate {i} - On vs En', fontweight='bold')
            ax.set_xlabel('Temps t')
            ax.set_ylabel('Valeur')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Données On_{i}/En_{i}\nnon disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'individual_outputs_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Graphique sorties sauvé: {filename}")
    return filename

def plot_regulation_analysis(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Analyse détaillée de la régulation G_i et des erreurs par strate.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Erreurs par strate
    ax = axes[0, 0]
    for i in range(5):
        erreur_col = f'erreur_{i}'
        if erreur_col in df.columns:
            ax.plot(df['t'], df[erreur_col], label=f'erreur_{i}', 
                   color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax.set_title('🎯 Erreurs En - On par strate', fontweight='bold')
    ax.set_xlabel('Temps t')
    ax.set_ylabel('Erreur')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Régulation G_i par strate
    ax = axes[0, 1]
    for i in range(5):
        G_col = f'G_{i}'
        if G_col in df.columns:
            ax.plot(df['t'], df[G_col], label=f'G_{i}', 
                   color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax.set_title('🔧 Régulation G_i par strate', fontweight='bold')
    ax.set_xlabel('Temps t')
    ax.set_ylabel('G_i')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Distribution des erreurs
    ax = axes[1, 0]
    erreurs_data = []
    labels = []
    for i in range(5):
        erreur_col = f'erreur_{i}'
        if erreur_col in df.columns:
            erreurs_data.append(df[erreur_col].values)
            labels.append(f'Strate {i}')
    
    if erreurs_data:
        ax.boxplot(erreurs_data, labels=labels)
        ax.set_title('📊 Distribution des erreurs', fontweight='bold')
        ax.set_ylabel('Erreur')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Distribution de G_i
    ax = axes[1, 1]
    G_data = []
    labels = []
    for i in range(5):
        G_col = f'G_{i}'
        if G_col in df.columns:
            G_data.append(df[G_col].values)
            labels.append(f'Strate {i}')
    
    if G_data:
        ax.boxplot(G_data, labels=labels)
        ax.set_title('📊 Distribution de G_i', fontweight='bold')
        ax.set_ylabel('G_i')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 5. Corrélation erreur vs G
    ax = axes[2, 0]
    for i in range(5):
        erreur_col = f'erreur_{i}'
        G_col = f'G_{i}'
        if erreur_col in df.columns and G_col in df.columns:
            ax.scatter(df[erreur_col], df[G_col], 
                      label=f'Strate {i}', color=colors[i], alpha=0.6, s=20)
    
    ax.set_title('🔗 Relation Erreur vs G_i', fontweight='bold')
    ax.set_xlabel('Erreur (En - On)')
    ax.set_ylabel('G_i')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Feedback estimé Fn_i
    ax = axes[2, 1]
    betas = [0.22, 0.32, 0.35, 0.24, 0.38]  # depuis config.json
    
    for i in range(5):
        G_col = f'G_{i}'
        if G_col in df.columns:
            # Fn_i ≈ beta_i * G_i * gamma (approximation)
            Fn_approx = betas[i] * df[G_col].values
            ax.plot(df['t'], Fn_approx, label=f'Fn_{i} (approx)', 
                   color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax.set_title('⚡ Feedback estimé Fn_i', fontweight='bold')
    ax.set_xlabel('Temps t')
    ax.set_ylabel('Feedback Fn_i')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'regulation_analysis_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analyse régulation sauvée: {filename}")
    return filename

def create_correlation_matrix(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Crée une matrice de corrélation entre les dynamiques des différentes strates.
    """
    # Sélectionner les colonnes d'intérêt
    individual_cols = []
    strata_names = []
    
    for i in range(5):
        for metric in ['An', 'On', 'En', 'erreur', 'G']:
            col = f'{metric}_{i}'
            if col in df.columns:
                individual_cols.append(col)
                strata_names.append(f'S{i}_{metric}')
    
    if len(individual_cols) < 2:
        print("⚠️ Pas assez de données pour la matrice de corrélation")
        return ""
    
    # Calculer la matrice de corrélation
    corr_data = df[individual_cols].corr()
    
    # Créer la visualisation
    plt.figure(figsize=(14, 12))
    
    # Masquer la diagonale et le triangle supérieur pour plus de clarté
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    
    # Créer la heatmap
    sns.heatmap(corr_data, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={"shrink": .8},
                xticklabels=strata_names,
                yticklabels=strata_names)
    
    plt.title('🔗 Matrice de corrélation inter-strates\n(Dynamiques individuelles)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'correlation_matrix_{run_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Matrice de corrélation sauvée: {filename}")
    return filename

def generate_individual_stats(df: pd.DataFrame, output_dir: str, run_id: str) -> str:
    """
    Génère des statistiques détaillées sur chaque strate.
    """
    stats = {}
    
    for i in range(5):
        strata_stats = {
            'id': i,
            'amplitude': {},
            'phase': {},  # NOUVEAU
            'output': {},
            'error': {},
            'regulation': {}
        }
        
        # Statistiques amplitude
        An_col = f'An_{i}'
        if An_col in df.columns:
            values = df[An_col].values
            strata_stats['amplitude'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'variability_score': float(np.std(values) / (np.mean(values) + 1e-10))
            }
        
        # Statistiques phase
        phi_col = f'phi_{i}'
        if phi_col in df.columns:
            values = df[phi_col].values
            strata_stats['phase'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'max_excursion': float(np.max(np.abs(values))),
                'spiral_activity': float(np.std(values) > np.pi)  # Bool -> float: phases spiralantes ?
            }
        
        # Statistiques sortie observée
        On_col = f'On_{i}'
        if On_col in df.columns:
            values = df[On_col].values
            strata_stats['output'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Statistiques erreur
        erreur_col = f'erreur_{i}'
        if erreur_col in df.columns:
            values = df[erreur_col].values
            strata_stats['error'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'rmse': float(np.sqrt(np.mean(values**2))),
                'max_abs': float(np.max(np.abs(values)))
            }
        
        # Statistiques régulation
        G_col = f'G_{i}'
        if G_col in df.columns:
            values = df[G_col].values
            strata_stats['regulation'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'effectiveness': float(np.std(values) / (np.std(df[erreur_col].values) + 1e-10)) if erreur_col in df.columns else 0.0
            }
        
        stats[f'strata_{i}'] = strata_stats
    
    # Sauver les statistiques
    import json
    stats_file = os.path.join(output_dir, f'individual_stats_{run_id}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Statistiques individuelles sauvées: {stats_file}")
    return stats_file

def visualize_all_individual_dynamics(debug_file_path: str, output_dir: str, run_id: str) -> Dict[str, str]:
    """
    Fonction principale pour générer toutes les visualisations des dynamiques individuelles.
    """
    print(f"🎨 GÉNÉRATION VISUALISATIONS INDIVIDUELLES")
    print(f"=" * 55)
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    df = load_individual_data(debug_file_path)
    if df.empty:
        print("❌ Aucune donnée chargée, abandon.")
        return {}
    
    results = {}
    
    try:
        # 1. Graphiques des amplitudes
        results['amplitudes'] = plot_individual_amplitudes(df, output_dir, run_id)
        
        # 2. SIGNATURES PHASIQUES - Voix propre selon Andréa ! 🎭
        results['signatures'] = plot_signature_analysis(df, output_dir, run_id)
        
        # 3. Dynamique spiralée r(t) et fn(t) - CŒUR du FPS
        results['spiral_dynamics'] = plot_spiral_dynamics(df, output_dir, run_id)
        
        # 4. Graphiques des phases - CLÉ de la dynamique FPS
        results['phases'] = plot_individual_phases(df, output_dir, run_id)
        
        # 4. Graphiques des sorties
        results['outputs'] = plot_individual_outputs(df, output_dir, run_id)
        
        # 3. Analyse de la régulation
        results['regulation'] = plot_regulation_analysis(df, output_dir, run_id)
        
        # 4. Matrice de corrélation
        results['correlation'] = create_correlation_matrix(df, output_dir, run_id)
        
        # 5. Statistiques détaillées
        results['stats'] = generate_individual_stats(df, output_dir, run_id)
        
        # 6. Résumé HTML
        results['html_report'] = create_html_report(results, output_dir, run_id)
        
        print(f"\n✅ Toutes les visualisations générées avec succès !")
        print(f"📁 Dossier: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_html_report(results: Dict[str, str], output_dir: str, run_id: str) -> str:
    """
    Crée un rapport HTML interactif avec toutes les visualisations.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌀 Rapport Dynamiques Individuelles FPS - {run_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .section {{ margin-bottom: 40px; }}
            .image {{ text-align: center; margin: 20px 0; }}
            .image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .stats {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 15px 0; }}
            .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; font-style: italic; }}
            .highlight {{ background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌀 Dynamiques Individuelles FPS</h1>
            <div class="highlight">
                <strong>Run ID:</strong> {run_id}<br>
                <strong>Généré le:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
    """
    
    # Ajouter chaque section avec son image
    sections = [
        ('amplitudes', '🎵 Amplitudes An(t) par Strate', 
         'Évolution temporelle des amplitudes de chaque strate. Chaque couleur représente une strate différente.'),
        ('spiral_dynamics', '🌀 Dynamique Spiralée r(t) & fn(t)', 
         'CŒUR du système FPS ! Visualise le ratio spiralé r(t), les fréquences fn(t) et la contrainte spiralée fn+1/fn ≈ r(t).'),
        ('phases', '🌀 Phases φₙ(t) par Strate', 
         'Évolution des phases individuelles - CLÉ de la dynamique spiralée FPS ! Détecte les excursions au-delà de 2π.'),
        ('outputs', '🎯 Sorties Observées vs Attendues', 
         'Comparaison entre les sorties observées On(t) et attendues En(t) pour chaque strate.'),
        ('regulation', '🔧 Analyse de la Régulation', 
         'Mécanismes de régulation G(t) et erreurs de tracking pour chaque strate.'),
        ('correlation', '🔗 Matrice de Corrélation Inter-Strates', 
         'Corrélations entre les dynamiques des différentes strates.')
    ]
    
    for key, title, description in sections:
        if key in results and results[key]:
            image_name = os.path.basename(results[key])
            html_content += f"""
            <div class="section">
                <h2>{title}</h2>
                <p>{description}</p>
                <div class="image">
                    <img src="{image_name}" alt="{title}">
                </div>
            </div>
            """
    
    html_content += """
            <div class="footer">
                <p>Généré automatiquement par le système FPS 🌀<br>
                (c) 2025 Gepetto & Andréa Gadal & Claude</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Sauver le fichier HTML
    html_file = os.path.join(output_dir, f'individual_dynamics_report_{run_id}.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Rapport HTML généré: {html_file}")
    return html_file


if __name__ == "__main__":
    # Test avec le dernier run
    debug_file = "logs/debug_detailed_run_20250725-172835_FPS_seed12345.csv"
    output_dir = "visualizations_individual"
    run_id = "run_20250725-172835_FPS_seed12345"
    
    if os.path.exists(debug_file):
        results = visualize_all_individual_dynamics(debug_file, output_dir, run_id)
        print("\n🎉 Visualisations terminées !")
        for key, path in results.items():
            print(f"  {key}: {path}")
    else:
        print(f"❌ Fichier debug non trouvé: {debug_file}") 