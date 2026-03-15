"""
visualize.py (version Notebook)- Visualisation complète du système FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module donne des yeux à la danse spiralée FPS :
- Évolution temporelle des signaux
- Comparaisons entre strates
- Diagrammes de phase
- Tableaux de bord interactifs
- Grille empirique avec notation visuelle
- Animations de l'évolution spiralée
- Matrices de corrélation
- Rapports HTML complets

La visualisation est le miroir qui permet de voir l'invisible,
de comprendre l'émergence et de partager la beauté du système.

(c) 2025 Exybris
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.stats import pearsonr
import warnings
from collections import defaultdict
import utils

# Configuration matplotlib pour de beaux graphiques
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2

# Couleurs FPS thématiques
FPS_COLORS = {
    'primary': '#2E86AB',    # Bleu profond
    'secondary': '#A23B72',  # Magenta
    'accent': '#F18F01',     # Orange
    'success': '#87BE3F',    # Vert
    'warning': '#FFC43D',    # Jaune
    'danger': '#C73E1D',     # Rouge
    'spiral': '#6A4C93'      # Violet spirale
}

# Palette pour multiples strates
STRATA_COLORS = plt.cm.viridis(np.linspace(0, 1, 20))


# ============== ÉVOLUTION TEMPORELLE ==============

def plot_signal_evolution(t_array: np.ndarray, S_array: np.ndarray, 
                          title: str = "Évolution du signal global S(t)") -> plt.Figure:
    """
    Trace l'évolution temporelle du signal global S(t).
    
    Args:
        t_array: array temporel
        S_array: valeurs du signal
        title: titre du graphique
    
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Signal principal
    ax.plot(t_array, S_array, color=FPS_COLORS['primary'], 
            linewidth=2.5, label='S(t)', alpha=0.8)
    
    # Zone d'enveloppe (±1 écart-type glissant)
    window = min(50, len(S_array) // 10)
    if window > 3:
        rolling_mean = np.convolve(S_array, np.ones(window)/window, mode='same')
        rolling_std = np.array([np.std(S_array[max(0, i-window//2):min(len(S_array), i+window//2)]) 
                                for i in range(len(S_array))])
        
        ax.fill_between(t_array, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std,
                        alpha=0.2, color=FPS_COLORS['primary'],
                        label='±1σ glissant')
    
    # Ligne de zéro
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Annotations
    ax.set_xlabel('Temps', fontsize=12)
    ax.set_ylabel('S(t)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Grille améliorée
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Ajustement des marges
    plt.tight_layout()
    
    return fig


# ============== COMPARAISON DES STRATES ==============

def plot_strata_comparison(t_array: np.ndarray, An_arrays: np.ndarray, 
                           fn_arrays: np.ndarray) -> plt.Figure:
    """
    Compare l'évolution des amplitudes et fréquences par strate.
    
    Args:
        t_array: array temporel
        An_arrays: amplitudes par strate (shape: [N_strates, T])
        fn_arrays: fréquences par strate
    
    Returns:
        Figure matplotlib
    """
    N_strates = An_arrays.shape[0] if An_arrays.ndim > 1 else 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Amplitudes
    for n in range(N_strates):
        An = An_arrays[n] if An_arrays.ndim > 1 else An_arrays
        color = STRATA_COLORS[n % len(STRATA_COLORS)]
        ax1.plot(t_array, An, color=color, alpha=0.7, 
                 linewidth=2, label=f'Strate {n}')
    
    ax1.set_ylabel('Amplitude Aₙ(t)', fontsize=12)
    ax1.set_title('Évolution des amplitudes par strate', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Fréquences
    for n in range(N_strates):
        fn = fn_arrays[n] if fn_arrays.ndim > 1 else fn_arrays
        color = STRATA_COLORS[n % len(STRATA_COLORS)]
        ax2.plot(t_array, fn, color=color, alpha=0.7, 
                 linewidth=2, label=f'Strate {n}')
    
    ax2.set_xlabel('Temps', fontsize=12)
    ax2.set_ylabel('Fréquence fₙ(t)', fontsize=12)
    ax2.set_title('Évolution des fréquences par strate', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# ============== DIAGRAMME DE PHASE ==============

def plot_phase_diagram(phi_n_arrays: np.ndarray) -> plt.Figure:
    """
    Trace le diagramme de phase des strates.
    
    Args:
        phi_n_arrays: phases par strate (shape: [N_strates, T])
    
    Returns:
        Figure matplotlib
    """
    N_strates = phi_n_arrays.shape[0] if phi_n_arrays.ndim > 1 else 1
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Tracer chaque strate
    for n in range(N_strates):
        phi = phi_n_arrays[n] if phi_n_arrays.ndim > 1 else phi_n_arrays
        color = STRATA_COLORS[n % len(STRATA_COLORS)]
        
        # Représentation polaire
        r = np.ones_like(phi) * (0.5 + n * 0.5 / N_strates)
        ax.plot(phi, r, 'o', color=color, markersize=4, 
                alpha=0.6, label=f'Strate {n}')
    
    # Cercle unitaire
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
    
    ax.set_title('Diagramme de phase des strates', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.2)
    
    # Légende circulaire
    if N_strates <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig


# ============== TABLEAU DE BORD DES MÉTRIQUES ==============

def plot_metrics_dashboard(metrics_history: Union[Dict[str, List], List[Dict]]) -> plt.Figure:
    """
    Crée un tableau de bord complet avec toutes les métriques clés.
    
    Args:
        metrics_history: historique des métriques (dict ou list de dicts)
    
    Returns:
        Figure matplotlib
    """
    # Convertir en format uniforme si nécessaire
    if isinstance(metrics_history, list) and len(metrics_history) > 0:
        # Liste de dicts -> dict de listes
        keys = metrics_history[0].keys()
        history_dict = {k: [m.get(k, 0) for m in metrics_history] for k in keys}
    else:
        history_dict = metrics_history
    
    # Créer la grille de subplots - Augmenter la taille pour le nouveau bloc
    fig = plt.figure(figsize=(16, 15))
    gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Signal global S(t)
    if 'S(t)' in history_dict:
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(history_dict['S(t)'], color=FPS_COLORS['primary'], linewidth=2)
        ax1.set_title('Signal global S(t)', fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
    
    # 2. Coefficient d'accord C(t)
    if 'C(t)' in history_dict:
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(history_dict['C(t)'], color=FPS_COLORS['spiral'], linewidth=2)
        ax2.set_title('Accord spiralé C(t)', fontweight='bold')
        ax2.set_ylabel('Coefficient')
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)
    
    # 3. Effort et CPU
    ax3 = fig.add_subplot(gs[1, 0])
    if 'effort(t)' in history_dict:
        ax3.plot(history_dict['effort(t)'], color=FPS_COLORS['warning'], 
                 linewidth=2, label='Effort')
    if 'cpu_step(t)' in history_dict:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(history_dict['cpu_step(t)'], color=FPS_COLORS['danger'], 
                      linewidth=2, alpha=0.7, label='CPU')
        ax3_twin.set_ylabel('CPU (s)', color=FPS_COLORS['danger'])
    ax3.set_title('Effort & CPU', fontweight='bold')
    ax3.set_ylabel('Effort', color=FPS_COLORS['warning'])
    ax3.grid(True, alpha=0.3)
    
    # 4. Métriques de qualité
    ax4 = fig.add_subplot(gs[1, 1])
    if 'entropy_S' in history_dict:
        ax4.plot(history_dict['entropy_S'], color=FPS_COLORS['accent'], 
                 linewidth=2, label='Entropie')
    # Afficher fluidity au lieu de variance_d2S
    if 'fluidity' in history_dict:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(history_dict['fluidity'], color=FPS_COLORS['secondary'], 
                      linewidth=2, alpha=0.7, label='Fluidité')
        ax4_twin.set_ylabel('Fluidité', color=FPS_COLORS['secondary'])
        ax4_twin.set_ylim(0, 1.1)  # Fluidité entre 0 et 1
    elif 'variance_d2S' in history_dict:
        # Fallback : calculer fluidity depuis variance_d2S
        variance_data = np.array(history_dict['variance_d2S'])
        x = variance_data / 175.0  # Reference variance
        fluidity_data = 1 / (1 + np.exp(5.0 * (x - 1)))
        ax4_twin = ax4.twinx()
        ax4_twin.plot(fluidity_data, color=FPS_COLORS['secondary'], 
                      linewidth=2, alpha=0.7, label='Fluidité (calculée)')
        ax4_twin.set_ylabel('Fluidité', color=FPS_COLORS['secondary'])
        ax4_twin.set_ylim(0, 1.1)
    ax4.set_title('Innovation & Fluidité', fontweight='bold')
    ax4.set_ylabel('Entropie', color=FPS_COLORS['accent'])
    ax4.grid(True, alpha=0.3)
    
    # 5. Régulation
    if 'mean_abs_error' in history_dict:
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(history_dict['mean_abs_error'], color=FPS_COLORS['success'], linewidth=2)
        ax5.set_title('Erreur de régulation', fontweight='bold')
        ax5.set_ylabel('|Eₙ - Oₙ|')
        ax5.grid(True, alpha=0.3)
    
    # 6. Distribution des efforts
    if 'effort(t)' in history_dict:
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(history_dict['effort(t)'], bins=30, color=FPS_COLORS['warning'], 
                 alpha=0.7, edgecolor='black')
        ax6.set_title('Distribution de l\'effort', fontweight='bold')
        ax6.set_xlabel('Effort')
        ax6.set_ylabel('Fréquence')
    
    # 7. Statut de l'effort
    if 'effort_status' in history_dict:
        ax7 = fig.add_subplot(gs[2, 1])
        status_counts = defaultdict(int)
        for status in history_dict['effort_status']:
            status_counts[status] += 1
        
        colors = {'stable': FPS_COLORS['success'], 
                  'transitoire': FPS_COLORS['warning'],
                  'chronique': FPS_COLORS['danger']}
        
        ax7.pie(status_counts.values(), labels=status_counts.keys(), 
                colors=[colors.get(s, 'gray') for s in status_counts.keys()],
                autopct='%1.1f%%', startangle=90)
        ax7.set_title('Répartition des états d\'effort', fontweight='bold')
    
    # 8. NOUVEAU BLOC : Alignement En/On et gamma
    ax8 = fig.add_subplot(gs[3, :])  # Prend toute la largeur de la quatrième ligne
    
    # Tracer En_mean et On_mean
    if 'En_mean(t)' in history_dict and 'On_mean(t)' in history_dict:
        t_array = np.arange(len(history_dict['En_mean(t)']))
        ax8.plot(t_array, history_dict['En_mean(t)'], 'g--', linewidth=2, 
                 label='En (attendu)', alpha=0.8)
        ax8.plot(t_array, history_dict['On_mean(t)'], 'b-', linewidth=2, 
                 label='On (observé)', alpha=0.8)
        
        # Ajouter In_mean si disponible
        if 'In_mean(t)' in history_dict:
            ax8.plot(t_array, history_dict['In_mean(t)'], 'r:', linewidth=2,
                     label='In (input)', alpha=0.8)
        
        # Ajouter An_mean et fn_mean si disponibles
        if 'An_mean(t)' in history_dict:
            ax8.plot(t_array, history_dict['An_mean(t)'], 'm-', linewidth=1.5,
                     label='An (amplitude)', alpha=0.7)
        
        if 'fn_mean(t)' in history_dict:
            # Créer un axe secondaire pour fn car échelle différente
            ax8_twin = ax8.twinx()
            ax8_twin.plot(t_array, history_dict['fn_mean(t)'], 'c-', linewidth=1.5,
                         label='fn (fréquence)', alpha=0.7)
            ax8_twin.set_ylabel('Fréquence moyenne', color='c')
            ax8_twin.tick_params(axis='y', labelcolor='c')
            ax8_twin.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
        
        # Gamma comme remplissage translucide en bas
        if 'gamma_mean(t)' in history_dict:
            gamma_data = np.array(history_dict['gamma_mean(t)'])
            # Normaliser gamma pour l'affichage (entre 0 et 0.2 de l'échelle y)
            y_range = ax8.get_ylim()[1] - ax8.get_ylim()[0]
            gamma_scaled = gamma_data * 0.2 * y_range + ax8.get_ylim()[0]
            ax8.fill_between(t_array, ax8.get_ylim()[0], gamma_scaled, 
                            color='orange', alpha=0.3, label='gamma (latence)')
        
        ax8.set_title('Dynamique complète : In → An → On/En avec fn et gamma', fontweight='bold')
        ax8.set_xlabel('Temps')
        ax8.set_ylabel('Amplitude')
        ax8.legend(loc='upper left')
        ax8.grid(True, alpha=0.3)
    
    # 9. Résumé statistique (déplacé)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculer les statistiques
    stats_text = "Statistiques globales\n\n"
    
    if 'S(t)' in history_dict:
        S_data = np.array(history_dict['S(t)'])
        # Filtrer les valeurs aberrantes
        S_data_clean = S_data[np.isfinite(S_data)]
        if len(S_data_clean) > 0:
            stats_text += f"Signal S(t):\n"
            stats_text += f"  Moyenne: {np.mean(S_data_clean):.3f}\n"
            stats_text += f"  Écart-type: {np.std(S_data_clean):.3f}\n"
            stats_text += f"  Min/Max: [{np.min(S_data_clean):.3f}, {np.max(S_data_clean):.3f}]\n\n"
        else:
            stats_text += "Signal S(t): Données invalides\n\n"
    
    if 'effort(t)' in history_dict:
        effort_data = np.array(history_dict['effort(t)'])
        # Filtrer les valeurs aberrantes et limiter les valeurs extrêmes
        effort_data_clean = effort_data[np.isfinite(effort_data)]
        if len(effort_data_clean) > 0:
            # Limiter les valeurs extrêmes au 99e percentile
            percentile_99 = np.percentile(effort_data_clean, 99)
            effort_data_clean = effort_data_clean[effort_data_clean <= percentile_99]
            
            stats_text += f"Effort:\n"
            stats_text += f"  Moyenne: {np.mean(effort_data_clean):.3f}\n"
            stats_text += f"  Percentile 90: {np.percentile(effort_data_clean, 90):.3f}\n"
        else:
            stats_text += "Effort: Données invalides\n"
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 10. NOUVEAU : Best pair score par pas
    if 'best_pair_score' in history_dict:
        ax10 = fig.add_subplot(gs[4, :])
        score = np.array(history_dict['best_pair_score'], dtype=float)
        ax10.plot(score, color=FPS_COLORS['accent'], linewidth=2, label='Best pair score (courant)')
        # Marquer les améliorations
        if len(score) > 1:
            best_so_far = np.maximum.accumulate(np.nan_to_num(score, nan=0.0))
            improvements = np.where(np.diff(best_so_far, prepend=best_so_far[0]) > 1e-9)[0]
            ax10.scatter(improvements, score[improvements], color=FPS_COLORS['success'], s=30, zorder=3, label='Amélioration')
        ax10.set_title('Évolution du best_pair_score (viser 5)', fontweight='bold')
        ax10.set_xlabel('Pas de temps')
        ax10.set_ylabel('Score (0-5)')
        ax10.set_ylim(0, 5.1)
        ax10.grid(True, alpha=0.3)
        ax10.legend(loc='lower right')
    
    # Titre global
    fig.suptitle('Tableau de bord FPS - Vue d\'ensemble', fontsize=16, fontweight='bold')
    
    return fig


# ============== VISUALISATION D'EXPLORATION ==============


def plot_exploration_analysis(df: pd.DataFrame) -> plt.Figure:

                fig, axes = plt.subplots(3, 2, figsize=(16, 12))

                # 1. Distribution de l'effort
                axes[0, 0].hist(df['effort(t)'], bins=50, color='orange', alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Distribution de l\'effort', fontweight='bold')
                axes[0, 0].set_xlabel('Effort')
                axes[0, 0].set_ylabel('Fréquence')

                # 2. Évolution S(t) avec bandes de cohérence
                ax = axes[0, 1]
                ax.plot(df['t'], df['S(t)'], 'b-', linewidth=1.5, label='S(t)', alpha=0.8)
                # Colorer selon C(t)
                scatter = ax.scatter(df['t'], df['S(t)'], c=df['C(t)'], 
                     cmap='RdYlGn', s=1, alpha=0.5, vmin=-1, vmax=1)
                ax.set_title('Signal S(t) coloré par cohérence C(t)', fontweight='bold')
                ax.set_xlabel('Temps')
                ax.set_ylabel('S(t)')
                plt.colorbar(scatter, ax=ax, label='C(t)')

                # 3. Effort vs Fluidité
                if 'fluidity' in df.columns:
                    axes[1, 0].scatter(df['effort(t)'], df['fluidity'], 
                       c=df['t'], cmap='viridis', s=5, alpha=0.5)
                    axes[1, 0].set_title('Effort vs Fluidité (temps en couleur)', fontweight='bold')
                    axes[1, 0].set_xlabel('Effort')
                    axes[1, 0].set_ylabel('Fluidité')

                # 4. Évolution de la résilience
                if 'continuous_resilience' in df.columns:
                    axes[1, 1].plot(df['t'], df['continuous_resilience'], 
                    'g-', linewidth=2, alpha=0.8)
                    axes[1, 1].set_title('Résilience continue', fontweight='bold')
                    axes[1, 1].set_xlabel('Temps')
                    axes[1, 1].set_ylabel('Résilience')
                    axes[1, 1].set_ylim(0, 1.1)

                # 5. Matrice de corrélation (sélection de métriques)
                metrics_to_correlate = ['S(t)', 'C(t)', 'effort(t)', 'mean_abs_error']
                if 'fluidity' in df.columns:
                    metrics_to_correlate.append('fluidity')
                if 'continuous_resilience' in df.columns:
                    metrics_to_correlate.append('continuous_resilience')

                corr_matrix = df[metrics_to_correlate].corr()
                im = axes[2, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                axes[2, 0].set_xticks(range(len(metrics_to_correlate)))
                axes[2, 0].set_yticks(range(len(metrics_to_correlate)))
                axes[2, 0].set_xticklabels(metrics_to_correlate, rotation=45, ha='right')
                axes[2, 0].set_yticklabels(metrics_to_correlate)
                axes[2, 0].set_title('Matrice de corrélation', fontweight='bold')
                plt.colorbar(im, ax=axes[2, 0])

                # Ajouter les valeurs dans la matrice
                for i in range(len(metrics_to_correlate)):
                    for j in range(len(metrics_to_correlate)):
                        text = axes[2, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)

                # 6. Évolution temporelle des moyennes des strates
                axes[2, 1].plot(df['t'], df['An_mean(t)'], label='Amplitude', linewidth=2)
                axes[2, 1].plot(df['t'], df['fn_mean(t)'], label='Fréquence', linewidth=2)
                if 'gamma_mean(t)' in df.columns:
                    axes[2, 1].plot(df['t'], df['gamma_mean(t)'], label='Gamma', linewidth=2)
                axes[2, 1].set_title('Moyennes des strates', fontweight='bold')
                axes[2, 1].set_xlabel('Temps')
                axes[2, 1].set_ylabel('Valeur')
                axes[2, 1].legend()

                return fig


# ============== SIGNAUX PRINCIPAUX ==============

def plot_principal_signals(history) -> plt.Figure :
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

                # Signal global S(t)
                t_vals = [h['t'] for h in history]
                S_vals = [h['S(t)'] for h in history]
                ax1.plot(t_vals, S_vals, linewidth=2, color='#2E86AB', alpha=0.8)
                ax1.set_title('Signal global S(t)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Amplitude', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

                # Cohérence C(t)
                C_vals = [h['C(t)'] for h in history]
                ax2.plot(t_vals, C_vals, linewidth=2, color='#A23B72', alpha=0.8)
                ax2.set_title('Cohérence spiralée C(t)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Temps', fontsize=12)
                ax2.set_ylabel('Cohérence', fontsize=12)
                ax2.set_ylim(-1.1, 1.1)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

                return fig


# ============== VISUALISATION AMPLITUDES ET FRÉQUENCES ==============

def plot_amp_freq(history, config) -> plt.Figure :
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

                t_vals = [h['t'] for h in history]
                # Amplitudes moyennes
                An_vals = [h.get('An', [0]*config['system']['N']) for h in history]
                An_mean = [np.mean(a) for a in An_vals]
                ax1.plot(t_vals, An_mean, linewidth=2, color='#F18F01', alpha=0.8, label='Amplitude moyenne')
                ax1.set_title('Amplitude moyenne An(t)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Amplitude', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Fréquences moyennes
                fn_vals = [h.get('fn', [0]*config['system']['N']) for h in history]
                fn_mean = [np.mean(f) for f in fn_vals]
                ax2.plot(t_vals, fn_mean, linewidth=2, color='#6A994E', alpha=0.8, label='Fréquence moyenne')
                ax2.set_title('Fréquence moyenne fn(t)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Temps', fontsize=12)
                ax2.set_ylabel('Fréquence', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                return fig


# ============== GRILLE EMPIRIQUE ==============

def calculate_empirical_scores_notebook(history, config) :
    """
    Version notebook de calculate_empirical_scores.
    
    Calcule les scores 1-5 pour chaque critère basé sur l'historique des derniers 20% de la run.
    """
    scores = {}
    
    if not history or len(history) < 20:
        print("⚠️ Pas assez de données pour calculer les scores empiriques")
        return {
            'Stabilité': 3, 'Régulation': 3, 'Fluidité': 3,
            'Résilience': 3, 'Innovation': 3, 'Coût CPU': 3, 'Effort interne': 3
        }
    
    # Extraire les métriques des derniers pas
    last_20_percent = int(len(history) * 0.2)
    recent_history = history[-last_20_percent:]
    
    # 1. STABILITÉ - basée sur la variation du signal
    S_values = [h.get('S(t)', 0) for h in recent_history]
    std_s = np.std(S_values)
    if std_s < 0.5:
        scores['Stabilité'] = 5
    elif std_s < 0.7:
        scores['Stabilité'] = 4
    elif std_s < 1.0:
        scores['Stabilité'] = 3
    elif std_s < 1.3:
        scores['Stabilité'] = 2
    else:
        scores['Stabilité'] = 1
    
    # 2. RÉGULATION - basée sur l'erreur moyenne
    errors = [h.get('mean_abs_error', 1.0) for h in recent_history]
    mean_error = np.mean(errors)
    if mean_error < 0.1:
        scores['Régulation'] = 5
    elif mean_error < 0.5:
        scores['Régulation'] = 4
    elif mean_error < 1.0:
        scores['Régulation'] = 3
    elif mean_error < 1.5:
        scores['Régulation'] = 2
    else:
        scores['Régulation'] = 1
    
    # 3. FLUIDITÉ - basée sur la métrique de fluidité
    fluidity_values = [h.get('fluidity', 0.5) for h in recent_history]
    mean_fluidity = np.mean(fluidity_values)
    if mean_fluidity >= 0.9:
        scores['Fluidité'] = 5
    elif mean_fluidity >= 0.7:
        scores['Fluidité'] = 4
    elif mean_fluidity >= 0.5:
        scores['Fluidité'] = 3
    elif mean_fluidity >= 0.3:
        scores['Fluidité'] = 2
    else:
        scores['Fluidité'] = 1
    
    # 4. RÉSILIENCE - basée sur adaptive_resilience
    resilience_values = [h.get('adaptive_resilience', 0.5) for h in recent_history]
    mean_resilience = np.mean(resilience_values)
    if mean_resilience >= 0.9:
        scores['Résilience'] = 5
    elif mean_resilience >= 0.7:
        scores['Résilience'] = 4
    elif mean_resilience >= 0.5:
        scores['Résilience'] = 3
    elif mean_resilience >= 0.3:
        scores['Résilience'] = 2
    else:
        scores['Résilience'] = 1
    
    # 5. INNOVATION - basée sur l'entropie
    entropy_values = [h.get('entropy_S', 0) for h in recent_history]
    mean_entropy = np.mean(entropy_values)
    if mean_entropy > 0.8:
        scores['Innovation'] = 5
    elif mean_entropy > 0.6:
        scores['Innovation'] = 4
    elif mean_entropy > 0.4:
        scores['Innovation'] = 3
    elif mean_entropy > 0.3:
        scores['Innovation'] = 2
    else:
        scores['Innovation'] = 1
    
    # 6. COÛT CPU - basé sur cpu_step
    cpu_values = [h.get('cpu_step(t)', 0.001) for h in recent_history]
    mean_cpu = np.mean(cpu_values)
    if mean_cpu < 0.001:
        scores['Coût CPU'] = 5
    elif mean_cpu < 0.005:
        scores['Coût CPU'] = 4
    elif mean_cpu < 0.01:
        scores['Coût CPU'] = 3
    elif mean_cpu < 0.05:
        scores['Coût CPU'] = 2
    else:
        scores['Coût CPU'] = 1
    
    # 7. EFFORT INTERNE - basé sur effort(t)
    effort_values = [h.get('effort(t)', 1.0) for h in recent_history]
    mean_effort = np.mean(effort_values)
    if mean_effort < 0.5:
        scores['Effort interne'] = 5
    elif mean_effort < 1.0:
        scores['Effort interne'] = 4
    elif mean_effort < 2.0:
        scores['Effort interne'] = 3
    elif mean_effort < 5.0:
        scores['Effort interne'] = 2
    else:
        scores['Effort interne'] = 1
    
    return scores


def create_empirical_grid_notebook(scores_dict) -> plt.Figure:
    """
    Version notebook de create_empirical_grid.
    
    Crée une grille visuelle avec les scores 1-5.
    """
    
    # Définition des icônes et couleurs
    score_config = {
        1: {'icon': '*', 'color': '#C73E1D', 'label': 'Rupture/Chaotique'},
        2: {'icon': '**', 'color': '#FF6B35', 'label': 'Instable'},
        3: {'icon': '***', 'color': '#FFC43D', 'label': 'Fonctionnel'},
        4: {'icon': '****', 'color': '#87BE3F', 'label': 'Harmonieux'},
        5: {'icon': '*****', 'color': '#2E86AB', 'label': 'FPS-idéal'}
    }
    
    # Critères dans l'ordre
    criteria = ['Stabilité', 'Régulation', 'Fluidité', 'Résilience', 
                'Innovation', 'Coût CPU', 'Effort interne']
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Configuration
    n_criteria = len(criteria)
    y_positions = np.arange(n_criteria)
    
    # Fond alternant
    for i in range(n_criteria):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, alpha=0.1, color='gray')
    
    # Placer les scores
    for i, criterion in enumerate(criteria):
        score = scores_dict.get(criterion, 3)
        config = score_config[score]
        
        # Nom du critère
        ax.text(-0.1, i, criterion, fontsize=12, va='center', ha='right', 
                fontweight='bold')
        
        # Score visuel (icône)
        ax.text(0.15, i, config['icon'], fontsize=24, va='center', ha='center',
                color=config['color'], fontweight='bold')
        
        # Barre de progression
        ax.barh(i, score/5, left=0.3, height=0.6,
                color=config['color'], alpha=0.7, edgecolor='black', linewidth=1)
        
        # Valeur numérique
        ax.text(0.3 + score/5 + 0.05, i, str(score) + '/5', 
                fontsize=10, va='center', ha='left')
    
    # Configuration des axes
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.5, n_criteria - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Titre et légende
    ax.set_title('Grille Empirique FPS - Évaluation Qualitative', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Légende des scores
    legend_elements = []
    for score in range(1, 6):
        config = score_config[score]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=config['color'], markersize=10,
                      label=f"{score}: {config['label']}")
        )
    
    ax.legend(handles=legend_elements, loc='center right', 
             bbox_to_anchor=(1.25, 0.5), frameon=True,
             title='Échelle de notation', title_fontsize=12)
    
    # Score global
    global_score = sum(scores_dict.values()) / len(scores_dict)
    ax.text(0.5, -1.5, f'Score Global: {global_score:.1f}/5', 
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    return fig


# ============== Évolution temporelle des scores empiriques ==============

def plot_scores_evolution(history: List[Dict], config: Dict = None, 
                         save_path: Optional[str] = None, calculate_all_scores = None) -> tuple:
    """
    Affiche l'évolution temporelle de tous les scores empiriques.
    
    Crée 7 sous-plots empilés montrant comment chaque critère évolue,
    avec fond coloré selon le régime et marqueurs des moments importants.
    
    Args:
        history: historique complet de la simulation
        config: configuration (pour calculate_all_scores)
        save_path: chemin pour sauvegarder la figure
    """
    if not history or len(history) < 20:
        print("⚠️ Pas assez d'historique pour visualiser l'évolution")
        return
    
    print("📊 Génération de l'évolution des scores empiriques...")
    
    # Calculer les scores pour chaque timestep
    t_values = []
    scores_dict = {
        'Stabilité': [],
        'Régulation': [],
        'Fluidité': [],
        'Résilience': [],
        'Innovation': [],
        'Coût CPU': [],
        'Effort interne': []
    }
    
    # Calculer par fenêtres glissantes
    window_size = 50
    for i in range(len(history)):
        if i < window_size:
            continue
        
        # Fenêtre locale
        local_history = history[max(0, i-window_size):i+1]
        
        try:
            # Utiliser calculate_all_scores si disponible
            scores_result = calculate_all_scores(local_history, config)
            current_scores = scores_result.get('current', {})
            
            t_values.append(history[i]['t'])
            
            scores_dict['Stabilité'].append(current_scores.get('stability', 3))
            scores_dict['Régulation'].append(current_scores.get('regulation', 3))
            scores_dict['Fluidité'].append(current_scores.get('fluidity', 3))
            scores_dict['Résilience'].append(current_scores.get('resilience', 3))
            scores_dict['Innovation'].append(current_scores.get('innovation', 3))
            scores_dict['Coût CPU'].append(current_scores.get('cpu_cost', 3))
            scores_dict['Effort interne'].append(current_scores.get('effort', 3))
        except:
            continue
    
    if not t_values:
        print("⚠️ Impossible de calculer les scores")
        return
    
    # Créer la figure
    fig, axes = plt.subplots(7, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Évolution Temporelle des Scores Empiriques FPS', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Couleurs pour chaque critère
    colors = {
        'Stabilité': '#2E86AB',
        'Régulation': '#2E86AB', 
        'Fluidité': '#2E86AB',
        'Résilience': '#87BE3F',
        'Innovation': '#87BE3F',
        'Coût CPU': '#FFC43D',
        'Effort interne': '#FFC43D'
    }
    
    # Tracer chaque score
    for idx, (criterion, scores) in enumerate(scores_dict.items()):
        ax = axes[idx]
        
        # Ligne du score
        ax.plot(t_values, scores, color=colors[criterion], linewidth=2, label=criterion)
        ax.fill_between(t_values, scores, 0, alpha=0.2, color=colors[criterion])
        
        # Ligne de référence (score parfait = 5)
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=4, color='orange', linestyle='--', alpha=0.2, linewidth=0.5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        
        # Configuration
        ax.set_ylabel(criterion, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 5.5)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_yticks([1, 2, 3, 4, 5])
        
        # Fond alternant pour lisibilité
        if idx % 2 == 0:
            ax.set_facecolor('#f9f9f9')
    
    # X-axis label sur le dernier subplot
    axes[-1].set_xlabel('Temps', fontsize=12, fontweight='bold')

    return fig, t_values, scores_dict


# ============== Carte de chaleur (γ, G)) ==============

def plot_gamma_G_heatmap(history: List[Dict], gamma_journal: Dict = None,
                        save_path: Optional[str] = None):
    """
    Crée une heatmap 2D de l'espace (γ, G) exploré.
    
    Montre quelles combinaisons ont été testées et leurs performances,
    avec la trajectoire d'exploration superposée.
    
    Args:
        history: historique complet
        gamma_journal: journal gamma_adaptive_aware (pour coupled_states)
        save_path: chemin pour sauvegarder
    """
    if not gamma_journal or 'coupled_states' not in gamma_journal:
        print("⚠️ Pas de journal gamma disponible pour la heatmap")
        return
    
    print("Génération de la carte de chaleur (γ, G)...")
    
    coupled_states = gamma_journal['coupled_states']
    
    if not coupled_states:
        print("⚠️ Aucun état couplé trouvé")
        return
    
    # Préparer les données
    G_archs = ['tanh', 'resonance', 'spiral_log', 'adaptive', 'adaptive_aware']
    gamma_values = np.linspace(0.1, 1.0, 10)
    
    # Créer la matrice de performance
    performance_matrix = np.zeros((len(G_archs), len(gamma_values)))
    visit_count_matrix = np.zeros((len(G_archs), len(gamma_values)))
    
    for (gamma, G_arch), state_info in coupled_states.items():
        # Trouver les indices
        if G_arch not in G_archs:
            continue
        
        G_idx = G_archs.index(G_arch)
        gamma_idx = np.argmin(np.abs(gamma_values - gamma))
        
        # Performance moyenne
        perfs = state_info.get('performances', [])
        if perfs:
            performance_matrix[G_idx, gamma_idx] = np.mean(perfs[-5:])
            visit_count_matrix[G_idx, gamma_idx] = len(perfs)
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Carte de l\'Espace (γ, G) Exploré', fontsize=16, fontweight='bold')
    
    # ===== Subplot 1: Performance moyenne =====
    im1 = ax1.imshow(performance_matrix, aspect='auto', cmap='RdYlGn', 
                     vmin=0, vmax=5, interpolation='nearest')
    
    ax1.set_xticks(range(len(gamma_values)))
    ax1.set_xticklabels([f'{g:.1f}' for g in gamma_values])
    ax1.set_yticks(range(len(G_archs)))
    ax1.set_yticklabels(G_archs)
    
    ax1.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('G Architecture', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Moyenne par État', fontsize=12)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Score Performance (0-5)', fontsize=10)
    
    # Annoter les cellules avec les valeurs
    for i in range(len(G_archs)):
        for j in range(len(gamma_values)):
            if performance_matrix[i, j] > 0:
                text = ax1.text(j, i, f'{performance_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", 
                              fontsize=8, fontweight='bold')

    # ===== Subplot 2: Nombre de visites =====
    im2 = ax2.imshow(visit_count_matrix, aspect='auto', cmap='Blues',
                     interpolation='nearest')
    
    ax2.set_xticks(range(len(gamma_values)))
    ax2.set_xticklabels([f'{g:.1f}' for g in gamma_values])
    ax2.set_yticks(range(len(G_archs)))
    ax2.set_yticklabels(G_archs)
    
    ax2.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('G Architecture', fontsize=12, fontweight='bold')
    ax2.set_title('Nombre de Visites par État', fontsize=12)
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Nombre de visites', fontsize=10)
    
    # Annoter avec nombre de visites
    for i in range(len(G_archs)):
        for j in range(len(gamma_values)):
            if visit_count_matrix[i, j] > 0:
                text = ax2.text(j, i, f'{int(visit_count_matrix[i, j])}',
                              ha="center", va="center", color="white" if visit_count_matrix[i, j] > 50 else "black",
                              fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Sauvegardé: {save_path}")
    
    plt.close()
    
    # Statistiques
    print(f"\nStatistiques de l'exploration:")
    total_states = len(coupled_states)
    total_visits = int(np.sum(visit_count_matrix))
    print(f"  - États uniques explorés: {total_states}")
    print(f"  - Visites totales: {total_visits}")
    
    # Meilleur état
    best_idx = np.unravel_index(np.argmax(performance_matrix), performance_matrix.shape)
    best_G = G_archs[best_idx[0]]
    best_gamma = gamma_values[best_idx[1]]
    best_perf = performance_matrix[best_idx]
    
    print(f"  - Meilleur état trouvé: γ={best_gamma:.1f}, G={best_G}, Score={best_perf:.2f}")

    plot_gamma_G_heatmap(
    history=results['history'],
    gamma_journal=gamma_journal,
    save_path=os.path.join(dirs['figures'], 'gamma_G_heatmap.png')
    )

    return fig


# ============== Chronologie des découvertes ==============

def plot_discovery_timeline(history: List[Dict], gamma_journal: Dict = None,
                           save_path: Optional[str] = None):
    """
    Crée une timeline narrative des événements importants.
    
    Montre l'évolution du meilleur couple (γ, G) découvert,
    les transitions de régime, et les moments de percée.
    
    Args:
        history: historique complet
        gamma_journal: journal gamma (transitions, découvertes)
        save_path: chemin pour sauvegarder
    """
    if not history:
        print("⚠️ Pas d'historique disponible")
        return
    
    print("Génération de la chronologie des découvertes...")
    
    # Extraire les données temporelles
    t_values = [h['t'] for h in history]
    gamma_values = [h.get('gamma', 1.0) for h in history]
    G_arch_values = [h.get('G_arch_used', 'tanh') for h in history]
    
    # Extraire best_pair depuis history (avec gestion robuste des None)
    best_pair_scores = []
    best_pair_gammas = []

    for h in history:
        score = h.get('best_pair_score')
        gamma = h.get('best_pair_gamma')
    
        # Convertir None en 0 pour le score
        best_pair_scores.append(score if score is not None else 0)
        best_pair_gammas.append(gamma)
    
    # Créer la figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.5, 1.5, 1], hspace=0.3)
    
    fig.suptitle('Chronologie des Découvertes FPS', fontsize=16, fontweight='bold')
    
    # ===== Subplot 1: Gamma et Best Score =====
    ax1 = fig.add_subplot(gs[0])
    
    # Gamma actuel
    ax1_gamma = ax1.twinx()
    ax1_gamma.plot(t_values, gamma_values, color='#2E86AB', linewidth=2, 
                   label='γ actuel', alpha=0.7)
    ax1_gamma.set_ylabel('Gamma (γ)', fontsize=11, fontweight='bold', color='#2E86AB')
    ax1_gamma.set_ylim(0, 1.1)
    ax1_gamma.tick_params(axis='y', labelcolor='#2E86AB')
    
    # Best pair score
    ax1.plot(t_values, best_pair_scores, color='#87BE3F', linewidth=3,
            label='Meilleur Score Découvert', marker='o', markersize=3, markevery=50)
    ax1.fill_between(t_values, best_pair_scores, alpha=0.2, color='#87BE3F')
    
    ax1.set_ylabel('Best Pair Score', fontsize=11, fontweight='bold', color='#87BE3F')
    ax1.set_ylim(0, 5.5)
    ax1.tick_params(axis='y', labelcolor='#87BE3F')
    ax1.axhline(y=4.5, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper left', fontsize=9)
    ax1_gamma.legend(loc='upper right', fontsize=9)
    ax1.set_title('Évolution de γ et Découverte du Meilleur État', fontsize=12, pad=10)
    
    # ===== Subplot 2: G Architecture Timeline =====
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Convertir G_arch en valeurs numériques pour visualiser
    G_arch_map = {'tanh': 0, 'resonance': 1, 'spiral_log': 2, 'adaptive': 3}
    G_arch_numeric = [G_arch_map.get(g, 0) for g in G_arch_values]
    
    ax2.plot(t_values, G_arch_numeric, color='#FF6B35', linewidth=2, 
            drawstyle='steps-post')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(['tanh', 'resonance', 'spiral_log', 'adaptive'])
    ax2.set_ylabel('G Architecture', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', axis='x')
    ax2.set_title('Architecture G(x) Utilisée', fontsize=12, pad=10)
    ax2.set_facecolor('#f9f9f9')
    
    # ===== Subplot 3: Moments de Percée =====
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Détecter les moments où best_pair_score augmente significativement
    breakthroughs = []
    for i in range(1, len(best_pair_scores)):
        if best_pair_scores[i] > best_pair_scores[i-1] + 0.3:  # Augmentation > 0.3
            breakthroughs.append((t_values[i], best_pair_scores[i]))
    
    if breakthroughs:
        bt_times, bt_scores = zip(*breakthroughs)
        ax3.scatter(bt_times, bt_scores, color='gold', s=200, marker='*', 
                   edgecolor='orange', linewidth=2, label='Percée!', zorder=5)
        
        # Annoter les percées
        for t, score in breakthroughs[:5]:  # Limiter à 5 annotations
            ax3.annotate(f'{score:.2f}', (t, score), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax3.plot(t_values, best_pair_scores, color='#87BE3F', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 5.5)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_title('Moments de Percée (Δscore > 0.3)', fontsize=12, pad=10)
    
    # ===== Subplot 4: Régimes (si disponible) =====
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    if gamma_journal and 'transitions' in gamma_journal:
        transitions = gamma_journal['transitions']
        
        if transitions:
            # Dessiner les régimes comme des blocs colorés
            regime_colors = {
                'exploration': '#FFC43D',
                'transcendent': '#87BE3F',
                'transcendent_synergy': '#2E86AB',
                'stable': '#87BE3F',
                'rest': '#C73E1D'
            }
            
            current_regime = 'exploration'
            regime_start = 0
            
            for transition in transitions:
                t_trans = transition.get('t', 0)
                new_regime = transition.get('regime', 'exploration')
                
                # Dessiner le bloc du régime précédent
                color = regime_colors.get(current_regime, 'gray')
                ax4.axvspan(regime_start, t_trans, alpha=0.3, color=color)
                
                current_regime = new_regime
                regime_start = t_trans
            
            # Dernier régime jusqu'à la fin
            color = regime_colors.get(current_regime, 'gray')
            ax4.axvspan(regime_start, t_values[-1], alpha=0.3, color=color)
            
            ax4.set_yticks([])
            ax4.set_title('Régimes Traversés', fontsize=12, pad=10)
        else:
            ax4.text(0.5, 0.5, 'Pas de transitions enregistrées', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=10, style='italic', color='gray')
            ax4.set_yticks([])
    else:
        ax4.text(0.5, 0.5, 'Journal gamma non disponible', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=10, style='italic', color='gray')
        ax4.set_yticks([])
    
    ax4.set_xlabel('Temps', fontsize=12, fontweight='bold')
    ax4.set_xlim(t_values[0], t_values[-1])

    return fig, t_values, breakthroughs, best_pair_scores


# ============== Évolution des métrques brutes ==============

def plot_metrics_evolution(history: List[Dict], 
                          metrics_to_plot: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = False):
    """
    Affiche l'évolution temporelle des métriques brutes du système FPS.
    
    Contrairement à plot_scores_evolution qui montre les scores agrégés (1-5),
    cette fonction montre les valeurs RAW des métriques au cours du temps.
    
    Args:
        history: historique complet de la simulation
        metrics_to_plot: liste des métriques à tracer (None = sélection par défaut)
        save_path: chemin pour sauvegarder
        show: si True, affiche dans le notebook
    """
    if not history or len(history) < 10:
        print("⚠️ Pas assez d'historique pour visualiser")
        return
    
    print("📈 Génération de l'évolution des métriques brutes...")
    
    # Extraire le temps
    t_values = [h['t'] for h in history]
    
    # Métriques par défaut (organisées par thème)
    if metrics_to_plot is None:
        metric_groups = {
            'Signaux Principaux': ['S(t)', 'C(t)', 'E(t)'],
            'Filtre perceprif S(t) (Prior perceptif)': ['S(t)'],
            'Signal Global O(t)': ['On_mean(t)'],
            'État cible E(t) (Prior prospectif)': ['En_mean(t)'],
            'Erreur' : ['mean_abs_error'],
            'Effort & Régulation': ['effort(t)', 'mean_abs_error', 'd_effort_dt'],
            'Adaptation & Innovation': ['entropy_S', 'fluidity', 'temporal_coherence'],
            'Paramètres Gamma': ['gamma', 'gamma_mean(t)'],
            'Fréquence': ['fn_mean(t)'],
            'Amplitude': ['An_mean(t)'],
            'Temps Caractéristiques': ['tau_A_mean', 'tau_f_mean', 'tau_S', 'tau_gamma'],
            'Résilience': ['adaptive_resilience', 'continuous_resilience'],
            'Best Pair': ['best_pair_score', 'best_pair_gamma'],
            'Stabilité' : ['std_S', 'variance_d2S'],
            'Input' : ['In_mean(t)'],
            'Erreur' : ['En_mean(t)', 'On_mean(t)']
        }
    else:
        # Si l'utilisateur spécifie, tout mettre dans un groupe
        metric_groups = {'Métriques Sélectionnées': metrics_to_plot}
    
    # Créer une figure avec subplots pour chaque groupe
    n_groups = len(metric_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(16, 4*n_groups), sharex=True)
    
    if n_groups == 1:
        axes = [axes]  # Pour cohérence
    
    fig.suptitle('Évolution Temporelle des Métriques FPS (Valeurs Brutes)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Couleurs variées
    colors = ['#2E86AB', '#87BE3F', '#FFC43D', '#FF6B35', '#C73E1D', 
              '#A23B72', '#6A994E', '#BC4B51', '#5F0F40', '#0FA3B1']
    
    for group_idx, (group_name, metrics) in enumerate(metric_groups.items()):
        ax = axes[group_idx]
        
        # Pour chaque métrique du groupe
        plotted_any = False
        for metric_idx, metric in enumerate(metrics):
            # Extraire les valeurs
            values = []
            for h in history:
                val = h.get(metric)
                # Gérer les None
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(val)
                else:
                    values.append(None)
            
            # Vérifier qu'on a des valeurs
            non_none_values = [v for v in values if v is not None]
            if not non_none_values:
                continue
            
            plotted_any = True
            
            # Choisir la couleur
            color = colors[metric_idx % len(colors)]
            
            # Tracer
            # Convertir None en NaN pour matplotlib
            values_plot = [v if v is not None else np.nan for v in values]
            
            ax.plot(t_values, values_plot, 
                   color=color, linewidth=2, label=metric, alpha=0.8)
        
        if plotted_any:
            ax.set_ylabel('Valeur', fontweight='bold', fontsize=11)
            ax.set_title(group_name, fontsize=12, fontweight='bold', pad=10)
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            # Fond alternant pour lisibilité
            if group_idx % 2 == 0:
                ax.set_facecolor('#f9f9f9')
        else:
            ax.text(0.5, 0.5, f'Aucune donnée disponible pour {group_name}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic', color='gray')
            ax.set_yticks([])
    
    # X-axis label sur le dernier subplot
    axes[-1].set_xlabel('Temps', fontsize=12, fontweight='bold')

    return fig

def plot_metrics_evolution_custom(history: List[Dict],
                                  metrics_config: Dict[str, List[str]],
                                  save_path: Optional[str] = None,
                                  show: bool = False):
    """
    Version personnalisable avec configuration explicite des groupes de métriques.
    
    Args:
        history: historique complet
        metrics_config: dict {nom_groupe: [liste_métriques]}
        save_path: chemin pour sauvegarder
        show: si True, affiche dans le notebook
        
    Exemple:
        metrics_config = {
            'Mon Groupe 1': ['S(t)', 'C(t)'],
            'Mon Groupe 2': ['gamma', 'effort(t)']
        }
    """
    return plot_metrics_evolution(history, None, save_path, show)

def plot_single_metric_detailed(history: List[Dict], 
                                metric_name: str,
                                save_path: Optional[str] = None,
                                show: bool = False):
    """
    Vue détaillée d'UNE SEULE métrique avec statistiques avancées.
    
    Args:
        history: historique complet
        metric_name: nom de la métrique à analyser
        save_path: chemin pour sauvegarder
        show: si True, affiche dans le notebook
    """
    if not history:
        print("⚠️ Pas d'historique disponible")
        return
    
    print(f"🔍 Analyse détaillée de {metric_name}...")
    
    # Extraire les données
    t_values = [h['t'] for h in history]
    values = []
    for h in history:
        val = h.get(metric_name)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            values.append(val)
        else:
            values.append(None)
    
    # Filtrer les None pour les stats
    non_none_values = [v for v in values if v is not None]
    non_none_times = [t for t, v in zip(t_values, values) if v is not None]
    
    if not non_none_values:
        print(f"⚠️ Aucune valeur disponible pour {metric_name}")
        return
    
    # Créer la figure avec 3 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Analyse Détaillée : {metric_name}', fontsize=16, fontweight='bold')
    
    # ===== Subplot 1: Série temporelle principale =====
    ax1 = fig.add_subplot(gs[0, :])
    
    values_plot = [v if v is not None else np.nan for v in values]
    ax1.plot(non_none_times, non_none_values, color='#2E86AB', linewidth=2)
    ax1.fill_between(non_none_times, non_none_values, alpha=0.2, color='#2E86AB')
    
    # Lignes de statistiques
    mean_val = np.mean(non_none_values)
    std_val = np.std(non_none_values)
    ax1.axhline(y=mean_val, color='green', linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_val:.4f}', alpha=0.7)
    ax1.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1, 
               label=f'+1σ: {mean_val+std_val:.4f}', alpha=0.5)
    ax1.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1, 
               label=f'-1σ: {mean_val-std_val:.4f}', alpha=0.5)
    
    ax1.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax1.set_title('Évolution Temporelle', fontsize=12, pad=10)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # ===== Subplot 2: Histogramme =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.hist(non_none_values, bins=50, color='#87BE3F', alpha=0.7, edgecolor='black')
    ax2.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label='Moyenne')
    ax2.set_xlabel('Valeur', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution des Valeurs', fontsize=11, pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== Subplot 3: Box plot =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    bp = ax3.boxplot([non_none_values], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#FFC43D')
    ax3.set_ylabel('Valeur', fontsize=11, fontweight='bold')
    ax3.set_title('Box Plot', fontsize=11, pad=10)
    ax3.set_xticklabels([metric_name])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== Subplot 4: Statistiques textuelles =====
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculer les statistiques
    stats_text = f"""
    📊 STATISTIQUES DÉTAILLÉES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📈 Valeurs:
       • Minimum:     {np.min(non_none_values):.6f}
       • Maximum:     {np.max(non_none_values):.6f}
       • Médiane:     {np.median(non_none_values):.6f}
       • Moyenne:     {mean_val:.6f}
       • Écart-type:  {std_val:.6f}
    
    📊 Distribution:
       • Q1 (25%):    {np.percentile(non_none_values, 25):.6f}
       • Q3 (75%):    {np.percentile(non_none_values, 75):.6f}
       • IQR:         {np.percentile(non_none_values, 75) - np.percentile(non_none_values, 25):.6f}
    
    ⏱️  Temporel:
       • Nb points:   {len(non_none_values)}
       • Durée:       {non_none_times[-1] - non_none_times[0]:.2f}s
       • Δt moyen:    {np.mean(np.diff(non_none_times)):.4f}s
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    return fig


# ============== ANALYSE DES CORRÉLATIONS NÉGATIVES ET POSITIVES ==============

def analyze_correlations(history: List[Dict], 
                         metrics_to_analyze: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         show: bool = False):
    """
    Analyse les corrélations entre métriques du système FPS.
    
    Crée une matrice de corrélation (heatmap) et identifie automatiquement
    les paires de métriques les plus corrélées.
    
    Args:
        history: historique complet de la simulation
        metrics_to_analyze: liste des métriques à analyser (None = sélection auto)
        save_path: chemin pour sauvegarder
        show: si True, affiche dans le notebook
        
    Returns:
        dict: résultats de l'analyse (corrélations fortes, etc.)
    """
    if not history or len(history) < 20:
        print("⚠️ Pas assez d'historique pour analyser les corrélations")
        return None
    
    print("Analyse des corrélations entre métriques...")
    
    # Métriques par défaut (numériques uniquement)
    if metrics_to_analyze is None:
        metrics_to_analyze = [
            'S(t)', 'C(t)', 'E(t)',
            'effort(t)', 'entropy_S', 'fluidity',
            'mean_abs_error', 'variance_d2S', 'std_S',
            'gamma', 'gamma_mean(t)',
            'An_mean(t)', 'fn_mean(t)',
            'En_mean(t)', 'On_mean(t)', 'In_mean(t)',
            'tau_A_mean', 'tau_f_mean', 'tau_S',
            'temporal_coherence', 'adaptive_resilience', 'continuous_resilience',
            'best_pair_score', 'best_pair_gamma'
            'decorrelation_time', 'autocorr_tau',
            'mean_high_effort', 'd_effort_dt', 'max_median_ratio'
        ]
    
    # Créer un DataFrame avec les métriques
    data = {}
    for metric in metrics_to_analyze:
        values = []
        for h in history:
            val = h.get(metric)
            # Convertir None en NaN
            if val is None:
                values.append(np.nan)
            else:
                values.append(float(val))
        data[metric] = values
    
    df = pd.DataFrame(data)
    
    # Supprimer les colonnes avec trop de NaN (>50%)
    df_clean = df.dropna(axis=1, thresh=len(df) * 0.5)
    
    if df_clean.shape[1] < 2:
        print("⚠️ Pas assez de métriques valides pour analyser les corrélations")
        return None
    
    print(f"✓ {df_clean.shape[1]} métriques analysées sur {len(df_clean)} timesteps")
    
    # Calculer la matrice de corrélation
    corr_matrix = df_clean.corr()
    
    # Créer la figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1], 
                         hspace=0.3, wspace=0.3)
    
    fig.suptitle('Analyse des Corrélations entre Métriques FPS', 
                 fontsize=16, fontweight='bold')
    
    # ===== Subplot 1: Matrice de corrélation (heatmap) =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Heatmap
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_yticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(corr_matrix.columns, fontsize=8)
    
    ax1.set_title('Matrice de Corrélation (Pearson)', fontsize=12, fontweight='bold', pad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Corrélation', fontsize=10)
    
    # Annoter les valeurs significatives (|corr| > 0.5)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i != j:  # Pas la diagonale
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5 and not np.isnan(corr_val):
                    color = 'white' if abs(corr_val) > 0.7 else 'black'
                    ax1.text(j, i, f'{corr_val:.2f}',
                           ha='center', va='center', color=color, 
                           fontsize=7, fontweight='bold')
    
    # ===== Subplot 2: Top 10 corrélations positives =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extraire les corrélations (sans la diagonale)
    correlations = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                correlations.append({
                    'metric1': corr_matrix.columns[i],
                    'metric2': corr_matrix.columns[j],
                    'corr': corr_val
                })
    
    # Trier par valeur absolue
    correlations_sorted = sorted(correlations, key=lambda x: abs(x['corr']), reverse=True)
    
    # Top 10 positives
    top_positive = [c for c in correlations_sorted if c['corr'] > 0][:10]
    
    if top_positive:
        labels = [f"{c['metric1'][:8]}\nvs\n{c['metric2'][:8]}" for c in top_positive]
        values = [c['corr'] for c in top_positive]
        
        y_pos = np.arange(len(labels))
        colors = ['#87BE3F' if v > 0.7 else '#2E86AB' for v in values]
        
        ax2.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.set_xlabel('Corrélation', fontsize=10, fontweight='bold')
        ax2.set_title('Top 10 Corrélations Positives', fontsize=11, fontweight='bold', pad=10)
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'Aucune corrélation positive forte',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=10, style='italic', color='gray')
    
    # ===== Subplot 3: Top 10 corrélations négatives =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Top 10 négatives
    top_negative = [c for c in correlations_sorted if c['corr'] < 0][:10]
    
    if top_negative:
        labels = [f"{c['metric1'][:8]}\nvs\n{c['metric2'][:8]}" for c in top_negative]
        values = [c['corr'] for c in top_negative]
        
        y_pos = np.arange(len(labels))
        colors = ['#C73E1D' if v < -0.7 else '#FF6B35' for v in values]
        
        ax3.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=7)
        ax3.set_xlabel('Corrélation', fontsize=10, fontweight='bold')
        ax3.set_title('Top 10 Corrélations Négatives', fontsize=11, fontweight='bold', pad=10)
        ax3.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax3.axvline(x=-0.7, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.set_xlim(-1, 0)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'Aucune corrélation négative forte',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=10, style='italic', color='gray')
    
    # ===== Subplot 4: Statistiques textuelles =====
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    # Compter les corrélations fortes
    strong_positive = sum(1 for c in correlations if c['corr'] > 0.7)
    moderate_positive = sum(1 for c in correlations if 0.5 < c['corr'] <= 0.7)
    strong_negative = sum(1 for c in correlations if c['corr'] < -0.7)
    moderate_negative = sum(1 for c in correlations if -0.7 <= c['corr'] < -0.5)
    
    stats_text = f"""
    STATISTIQUES DES CORRÉLATIONS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Métriques analysées:   {df_clean.shape[1]}
    Timesteps:             {len(df_clean)}
    Paires analysées:      {len(correlations)}
    
    Corrélations POSITIVES:
       • Fortes (> 0.7):      {strong_positive}
       • Modérées (0.5-0.7):  {moderate_positive}
    
    Corrélations NÉGATIVES:
       • Fortes (< -0.7):     {strong_negative}
       • Modérées (-0.7:-0.5): {moderate_negative}
    
    Top 3 corrélations les plus fortes:
    """
    
    for i, c in enumerate(correlations_sorted[:3], 1):
        stats_text += f"\n       {i}. {c['metric1']} ↔ {c['metric2']}: {c['corr']:.3f}"
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Retourner les résultats
    results = {
        'correlation_matrix': corr_matrix,
        'top_positive': top_positive,
        'top_negative': top_negative,
        'strong_positive_count': strong_positive,
        'strong_negative_count': strong_negative
    }
    
    return fig, corr_matrix, top_negative, top_positive, strong_negative, strong_positive

def plot_scatter_pairs(history: List[Dict],
                      pairs: List[Tuple[str, str]],
                      save_path: Optional[str] = None,
                      show: bool = False):
    """
    Crée des scatter plots pour des paires spécifiques de métriques.
    
    Args:
        history: historique complet
        pairs: liste de tuples (metric1, metric2) à tracer
        save_path: chemin pour sauvegarder
        show: si True, affiche dans le notebook
    """
    if not history or len(history) < 10:
        print("⚠️ Pas assez d'historique")
        return
    
    print(f"📊 Création de {len(pairs)} scatter plots...")
    
    # Calculer le layout
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Scatter Plots des Paires de Métriques', fontsize=16, fontweight='bold')
    
    for idx, (metric1, metric2) in enumerate(pairs):
        ax = axes[idx]
        
        # Extraire les données
        x_data = []
        y_data = []
        
        for h in history:
            x_val = h.get(metric1)
            y_val = h.get(metric2)
            
            if x_val is not None and y_val is not None:
                x_data.append(float(x_val))
                y_data.append(float(y_val))
        
        if len(x_data) < 5:
            ax.text(0.5, 0.5, f'Pas assez de données\npour {metric1}\nvs {metric2}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.5, s=20, c='#2E86AB', edgecolor='black', linewidth=0.5)
        
        # Régression linéaire
        try:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2, label='Tendance')
            
            # Calculer corrélation
            corr, p_val = pearsonr(x_data, y_data)
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        except:
            pass
        
        ax.set_xlabel(metric1, fontsize=10, fontweight='bold')
        ax.set_ylabel(metric2, fontsize=10, fontweight='bold')
        ax.set_title(f'{metric1} vs {metric2}', fontsize=11, pad=10)
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Masquer les axes vides
    for idx in range(len(pairs), len(axes)):
        axes[idx].axis('off')

    return fig


"""
Visualise les patterns d'annulation et l'évolution temporelle des contributions.
Le 0 n'est pas le silence, mais l'équilibre des possibles.
"""

def visualize_stratum_patterns(history, config, output_dir=None, show=True):
    """
    Crée une visualisation complète des patterns par strate.
    
    Args:
        csv_path: Chemin vers le CSV stratum_details
        output_dir: Dossier de sortie pour les figures (optionnel)
        show: Afficher les plots (True) ou juste sauvegarder (False)
    
    Returns:
        dict avec les données analysées
    """
    # 2. Extraire N et créer les arrays
    N = config['system']['N']
    t_vals = []
    An_data = []
    On_data = []
    fn_data = []
    phin_data = []
    S_contrib_data = []

    # 3. Remplir depuis history
    for h in history:
        t_vals.append(h['t'])
        An_data.append(h.get('An', []))
        On_data.append(h.get('O', []))  # 'O' pas 'On' !
        fn_data.append(h.get('fn', []))
        phin_data.append(h.get('phi_n_t', []))
        S_contrib_data.append(h.get('S_contrib', []))

    # 4. Convertir en numpy arrays
    t = np.array(t_vals)
    An_data = np.array(An_data)
    On_data = np.array(On_data)
    fn_data = np.array(fn_data)
    phin_data = np.array(phin_data)
    S_contrib_data = np.array(S_contrib_data)

    # DEBUG
    print(f"Shape de On_data: {On_data.shape}")
    print(f"N attendu: {N}")
    print(f"Nombre réel de strates: {On_data.shape[1] if len(On_data.shape) > 1 else 'problème de shape'}")
    
    # ========================================================================
    # FIGURE 1 : L'ANNULATION EN ACTION
    # ========================================================================
    
    fig1 = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.3)
    
    # 1A. Évolution temporelle de On pour quelques strates
    ax1a = fig1.add_subplot(gs[0, :])
    
    # Sélection intelligente des strates à visualiser
    strates_to_plot = utils.select_representative_strata(N, config, n_strata_to_show=5)
    key_strates = utils.select_representative_strata(N, config, n_strata_to_show=3)
    
    print(f"Visualisation de {len(strates_to_plot)} strates représentatives: {strates_to_plot}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(strates_to_plot)))
    
    for i, n in enumerate(strates_to_plot):
        ax1a.plot(t, On_data[:, n], label=f'Strate {n}', 
                 color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax1a.set_xlabel('Temps (s)', fontsize=11)
    ax1a.set_ylabel('On(t)', fontsize=11)
    ax1a.set_title('Oscillations Individuelles - Chaque Strate Danse', 
                   fontsize=13, fontweight='bold')
    ax1a.legend(loc='upper right', fontsize=9)
    ax1a.grid(True, alpha=0.3)
    ax1a.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 1B. Somme instantanée de tous les On
    ax1b = fig1.add_subplot(gs[1, 0])
    
    On_sum = On_data.sum(axis=1)
    On_abs_sum = np.abs(On_data).sum(axis=1)
    
    ax1b.plot(t, On_abs_sum, label='Σ|On| (énergie totale)', 
             color='red', alpha=0.6, linewidth=2)
    ax1b.plot(t, np.abs(On_sum), label='|ΣOn| (signal net)', 
             color='blue', alpha=0.8, linewidth=2)
    ax1b.fill_between(t, 0, On_abs_sum, alpha=0.2, color='red', 
                       label='Annulation')
    
    ax1b.set_xlabel('Temps (s)', fontsize=11)
    ax1b.set_ylabel('Amplitude', fontsize=11)
    ax1b.set_title('L\'Annulation - L\'Équilibre des Contraires', 
                   fontsize=12, fontweight='bold')
    ax1b.legend(loc='upper right', fontsize=9)
    ax1b.grid(True, alpha=0.3)
    
    # 1C. Ratio d'annulation dans le temps
    ax1c = fig1.add_subplot(gs[1, 1])
    
    # Éviter division par zéro
    cancellation_ratio = np.zeros_like(t)
    for i in range(len(t)):
        if On_abs_sum[i] > 1e-10:
            cancellation_ratio[i] = 1 - np.abs(On_sum[i]) / On_abs_sum[i]
        else:
            cancellation_ratio[i] = 0
    
    ax1c.plot(t, cancellation_ratio * 100, color='purple', linewidth=2)
    ax1c.axhline(y=92.6, color='red', linestyle='--', linewidth=1.5, 
                label='Moyenne: 92.6%', alpha=0.7)
    
    ax1c.set_xlabel('Temps (s)', fontsize=11)
    ax1c.set_ylabel('Annulation (%)', fontsize=11)
    ax1c.set_title('Taux d\'Annulation Temporel', 
                   fontsize=12, fontweight='bold')
    ax1c.legend(loc='lower right', fontsize=9)
    ax1c.grid(True, alpha=0.3)
    ax1c.set_ylim([0, 100])
    
    # 1D. Heatmap des On(t) par strate
    ax1d = fig1.add_subplot(gs[2, :])
    
    im = ax1d.imshow(On_data.T, aspect='auto', cmap='RdBu_r', 
                     extent=[t[0], t[-1], N-0.5, -0.5],
                     vmin=-0.05, vmax=0.05, interpolation='bilinear')
    
    ax1d.set_xlabel('Temps (s)', fontsize=11)
    ax1d.set_ylabel('Strate n', fontsize=11)
    ax1d.set_title('Carte Thermique - Le Ballet Collectif des 50 Strates', 
                   fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1d, orientation='vertical', pad=0.01)
    cbar.set_label('On(t)', fontsize=10)
    
    # Marquer les strates dominantes
    for n in [9, 10, 28]:
        ax1d.axhline(y=n, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
    
    fig1.suptitle('L\'ÉQUILIBRE DES POSSIBLES - Le 0 comme Somme des Contraires', 
                  fontsize=15, fontweight='bold', y=0.995)
    
    if output_dir:
        output_path = Path(output_dir) / 'stratum_annulation_patterns.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure 1 sauvegardée: {output_path}")
    
    # ========================================================================
    # FIGURE 2 : CONTRIBUTIONS À S(t) ET LEUR ÉVOLUTION
    # ========================================================================
    
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = GridSpec(3, 2, figure=fig2, hspace=0.3, wspace=0.3)
    
    # 2A. Évolution temporelle des contributions pour strates clés
    ax2a = fig2.add_subplot(gs2[0, :])

    colors_key = ['red', 'blue', 'green']
    
    for i, n in enumerate(key_strates):
        ax2a.plot(t, S_contrib_data[:, n], label=f'Strate {n}', 
                 color=colors_key[i], linewidth=2, alpha=0.7)
    
    ax2a.set_xlabel('Temps (s)', fontsize=11)
    ax2a.set_ylabel('S_contrib(t)', fontsize=11)
    ax2a.set_title('Contributions Dominantes - Les Leaders du Signal', 
                   fontsize=13, fontweight='bold')
    ax2a.legend(loc='upper right', fontsize=10)
    ax2a.grid(True, alpha=0.3)
    ax2a.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 2B. Somme cumulative des contributions
    ax2b = fig2.add_subplot(gs2[1, 0])
    
    S_total = S_contrib_data.sum(axis=1)
    S_cumulative = np.cumsum(S_total)
    
    ax2b.plot(t, S_cumulative, color='darkblue', linewidth=2.5)
    ax2b.fill_between(t, 0, S_cumulative, alpha=0.3, color='darkblue')
    
    ax2b.set_xlabel('Temps (s)', fontsize=11)
    ax2b.set_ylabel('Σ S_contrib cumulatif', fontsize=11)
    ax2b.set_title('Accumulation du Signal - L\'Histoire se Construit', 
                   fontsize=12, fontweight='bold')
    ax2b.grid(True, alpha=0.3)
    ax2b.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 2C. Distribution des contributions totales par strate
    ax2c = fig2.add_subplot(gs2[1, 1])
    
    total_contribs = S_contrib_data.sum(axis=0)
    strate_indices = np.arange(N)
    
    colors_bars = ['red' if n in [9, 10, 28] else 'steelblue' for n in range(N)]
    ax2c.bar(strate_indices, total_contribs, color=colors_bars, alpha=0.7, 
             edgecolor='black', linewidth=0.5)
    
    ax2c.set_xlabel('Strate n', fontsize=11)
    ax2c.set_ylabel('Contribution totale à S(t)', fontsize=11)
    ax2c.set_title('Répartition - Qui Contribue Quoi ?', 
                   fontsize=12, fontweight='bold')
    ax2c.grid(True, alpha=0.3, axis='y')
    ax2c.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 2D. Heatmap des contributions temporelles
    ax2d = fig2.add_subplot(gs2[2, :])
    
    im2 = ax2d.imshow(S_contrib_data.T, aspect='auto', cmap='RdBu_r',
                      extent=[t[0], t[-1], N-0.5, -0.5],
                      vmin=-0.01, vmax=0.01, interpolation='bilinear')
    
    ax2d.set_xlabel('Temps (s)', fontsize=11)
    ax2d.set_ylabel('Strate n', fontsize=11)
    ax2d.set_title('Contributions Temporelles - La Symphonie Complète', 
                   fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2d, orientation='vertical', pad=0.01)
    cbar2.set_label('S_contrib(t)', fontsize=10)
    
    # Marquer les strates dominantes
    for n in [9, 10, 28]:
        ax2d.axhline(y=n, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
    
    fig2.suptitle('CONTRIBUTIONS À S(t) - Chaque Voix Dans la Chorale', 
                  fontsize=15, fontweight='bold', y=0.995)
    
    if output_dir:
        output_path2 = Path(output_dir) / 'stratum_contributions_evolution.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"✅ Figure 2 sauvegardée: {output_path2}")
    
    # ========================================================================
    # FIGURE 3 : DIVERGENCE DES FRÉQUENCES
    # ========================================================================
    
    fig3 = plt.figure(figsize=(16, 8))
    gs3 = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)
    
    # 3A. Évolution des fréquences dans le temps
    ax3a = fig3.add_subplot(gs3[0, :])
    
    # Tracer toutes les strates avec un gradient de couleur
    for n in range(N):
        color = plt.cm.plasma(n / N)
        ax3a.plot(t, fn_data[:, n], color=color, alpha=0.3, linewidth=0.8)
    
    # Mettre en valeur des strates représentatives
    highlight_strata = utils.select_representative_strata(N, config, n_strata_to_show=6)
    for n in highlight_strata:
        color = plt.cm.plasma(n / N)
        ax3a.plot(t, fn_data[:, n], color=color, linewidth=2, 
                label=f'Strate {n}', alpha=0.9)
    
    ax3a.set_xlabel('Temps (s)', fontsize=11)
    ax3a.set_ylabel('fn(t) [Hz]', fontsize=11)
    ax3a.set_title('Divergence des Fréquences - Séparation des Échelles', 
                   fontsize=13, fontweight='bold')
    ax3a.legend(loc='upper left', fontsize=9, ncol=2)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_yscale('log')
    
    # 3B. Distribution finale des fréquences
    ax3b = fig3.add_subplot(gs3[1, 0])
    
    fn_final = fn_data[-1, :]
    fn_initial = fn_data[0, :]
    
    ax3b.scatter(strate_indices, fn_initial, label='fn initiale', 
                marker='o', s=49, alpha=0.6, color='blue')
    ax3b.scatter(strate_indices, fn_final, label='fn finale', 
                marker='s', s=49, alpha=0.6, color='red')
    
    ax3b.set_xlabel('Strate n', fontsize=11)
    ax3b.set_ylabel('fn [Hz]', fontsize=11)
    ax3b.set_title('Avant / Après - L\'Escalade Géométrique', 
                   fontsize=12, fontweight='bold')
    ax3b.legend(loc='upper left', fontsize=10)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_yscale('log')
    
    # 3C. Ratio fn_final / fn_initial
    ax3c = fig3.add_subplot(gs3[1, 1])
    
    fn_ratio = fn_final / (fn_initial + 1e-10)
    
    ax3c.plot(strate_indices, fn_ratio, marker='o', linewidth=2, 
             markersize=6, color='purple', alpha=0.7)
    
    ax3c.set_xlabel('Strate n', fontsize=11)
    ax3c.set_ylabel('fn_final / fn_initial', fontsize=11)
    ax3c.set_title('Amplification - Combien Chaque Strate S\'Accélère', 
                   fontsize=12, fontweight='bold')
    ax3c.grid(True, alpha=0.3)
    ax3c.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    fig3.suptitle('⚡ SÉPARATION MULTI-ÉCHELLE - La Cascade Temporelle', 
                  fontsize=15, fontweight='bold', y=0.995)
    
    if output_dir:
        output_path3 = Path(output_dir) / 'stratum_frequency_divergence.png'
        plt.savefig(output_path3, dpi=150, bbox_inches='tight')
        print(f"✅ Figure 3 sauvegardée: {output_path3}")

    # ========================================================================
    # FIGURE 4 : PHASE PAR STRATE ET PATTERNS D'ANNULATION
    # ========================================================================
    
    fig4 = plt.figure(figsize=(16, 10))
    gs4 = GridSpec(3, 2, figure=fig4, hspace=0.3, wspace=0.3)

    # 1A. Évolution temporelle de On pour quelques strates
    ax4a = fig4.add_subplot(gs4[0, :])
    
    # Sélection intelligente des strates à visualiser
    strates_to_plot = utils.select_representative_strata(N, config, n_strata_to_show=5)
    key_strates = utils.select_representative_strata(N, config, n_strata_to_show=3)
    
    print(f"Visualisation de {len(strates_to_plot)} strates représentatives: {strates_to_plot}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(strates_to_plot)))
    
    for i, n in enumerate(strates_to_plot):
        ax4a.plot(t, phin_data[:, n], label=f'Strate {n}', 
                 color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax4a.set_xlabel('Temps (s)', fontsize=11)
    ax4a.set_ylabel('phin(t)', fontsize=11)
    ax4a.set_title('Phases Individuelles - Chaque Strate exprime son identité', 
                   fontsize=13, fontweight='bold')
    ax4a.legend(loc='upper right', fontsize=9)
    ax4a.grid(True, alpha=0.3)
    ax4a.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    

    # 1D. Heatmap des phin(t) par strate
    ax4d = fig4.add_subplot(gs4[2, :])
    
    im = ax4d.imshow(phin_data.T, aspect='auto', cmap='RdBu_r', 
                     extent=[t[0], t[-1], N-0.5, -0.5],
                     vmin=-0.05, vmax=0.05, interpolation='bilinear')
    
    ax4d.set_xlabel('Temps (s)', fontsize=11)
    ax4d.set_ylabel('Strate n', fontsize=11)
    ax4d.set_title('Carte Thermique - Le Ballet Collectif des phases des 50 Strates', 
                   fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax4d, orientation='vertical', pad=0.01)
    cbar.set_label('phin(t)', fontsize=10)
    
    # Marquer les strates dominantes
    for n in [9, 10, 28]:
        ax4d.axhline(y=n, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
    
    fig4.suptitle('L\'ÉQUILIBRE DES IDENTITÉS - Chimères ou non ?', 
                  fontsize=15, fontweight='bold', y=0.995)
    
    if output_dir:
        output_path4 = Path(output_dir) / 'stratum_phase_annulation_patterns.png'
        plt.savefig(output_path4, dpi=150, bbox_inches='tight')
        print(f"✅ Figure 4 sauvegardée: {output_path4}")
    
    if show:
        plt.show()
    
    print("\nVisualisation terminée !")
    
    # Retourner les données pour analyse ultérieure
    return {
        't': t,
        'An_data': An_data,
        'On_data': On_data,
        'fn_data': fn_data,
        'phin_data': phin_data,
        'S_contrib_data': S_contrib_data,
        'cancellation_ratio': cancellation_ratio,
        'S_total': S_total,
        'S_cumulative': S_cumulative
    }


if __name__ == "__main__":
    print("FPS - Visualisation des dynamiques par strates")
    print("="*70)
    print("\nUtilisation:")
    print("  data = visualize_stratum_patterns(csv_path, output_dir='figures/')")
    print("\nOù:")
    print("  csv_path: chemin vers stratum_details_*.csv")
    print("  output_dir: dossier pour sauvegarder les figures")
    print("\nLe 0 n'est pas le silence, mais l'équilibre des possibles.")


# ============== ANIMATION SPIRALE ==============

def animate_spiral_evolution(data: Dict[str, np.ndarray], 
                             output_path: str) -> None:
    """
    Crée une animation de l'évolution spiralée.
    
    Args:
        data: dictionnaire avec les données temporelles
        output_path: chemin de sortie pour l'animation
    """
    if 'S(t)' not in data or 'C(t)' not in data:
        warnings.warn("Données insuffisantes pour l'animation")
        return
    
    S_data = data['S(t)']
    C_data = data['C(t)']
    T = len(S_data)
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Configuration initiale
    ax1.set_xlim(0, T)
    ax1.set_ylim(np.min(S_data)*1.1, np.max(S_data)*1.1)
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('S(t)')
    ax1.set_title('Signal global S(t)')
    ax1.grid(True, alpha=0.3)
    
    # Spirale polaire
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_ylim(0, 1.5)
    ax2.set_title('Évolution spiralée')
    
    # Lignes à animer
    line1, = ax1.plot([], [], 'b-', linewidth=2)
    line2, = ax2.plot([], [], 'r-', linewidth=2)
    point, = ax2.plot([], [], 'ro', markersize=8)
    
    # Fonction d'initialisation
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        point.set_data([], [])
        return line1, line2, point
    
    # Fonction d'animation
    def animate(frame):
        # Signal temporel
        t = np.arange(frame)
        line1.set_data(t, S_data[:frame])
        
        # Spirale
        theta = np.linspace(0, 2*np.pi*frame/100, frame)
        r = 0.5 + 0.5 * C_data[:frame]
        line2.set_data(theta, r)
        
        # Point actuel
        if frame > 0:
            point.set_data([theta[-1]], [r[-1]])
        
        return line1, line2, point
    
    # Créer l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=T, interval=50, blit=True)
    
    # Sauvegarder
    try:
        anim.save(output_path, writer='pillow', fps=20)
        print(f"Animation sauvegardée : {output_path}")
    except Exception as e:
        warnings.warn(f"Impossible de sauvegarder l'animation : {e}")
        plt.show()


# ============== MATRICE DE CORRÉLATION ==============

def generate_correlation_matrix(criteria_terms_mapping: Dict[str, List[str]]) -> plt.Figure:
    """
    Génère une matrice de corrélation critère ↔ termes.
    
    Args:
        criteria_terms_mapping: dictionnaire {critère: [termes]}
    
    Returns:
        Figure matplotlib
    """
    # Extraire tous les termes uniques
    all_terms = set()
    for terms in criteria_terms_mapping.values():
        all_terms.update(terms)
    all_terms = sorted(list(all_terms))
    
    # Créer la matrice binaire
    criteria = list(criteria_terms_mapping.keys())
    matrix = np.zeros((len(criteria), len(all_terms)))
    
    for i, criterion in enumerate(criteria):
        for term in criteria_terms_mapping[criterion]:
            if term in all_terms:
                j = all_terms.index(term)
                matrix[i, j] = 1
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Axes
    ax.set_xticks(np.arange(len(all_terms)))
    ax.set_yticks(np.arange(len(criteria)))
    ax.set_xticklabels(all_terms, rotation=45, ha='right')
    ax.set_yticklabels(criteria)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Lien critère-terme', rotation=270, labelpad=15)
    
    # Titre
    ax.set_title('Matrice de correspondance Critères ↔ Termes FPS', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grille
    ax.set_xticks(np.arange(len(all_terms)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(criteria)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    return fig


# ============== RAPPORT HTML ==============

def export_html_report(all_data: Dict[str, Any], output_path: str) -> None:
    """
    Génère un rapport HTML complet avec tous les résultats.
    
    Args:
        all_data: toutes les données et résultats
        output_path: chemin de sortie HTML
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rapport FPS - Analyse complète</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2E86AB;
                border-bottom: 3px solid #2E86AB;
                padding-bottom: 10px;
            }
            h2 {
                color: #6A4C93;
                margin-top: 30px;
            }
            .metric-box {
                display: inline-block;
                margin: 10px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #2E86AB;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #2E86AB;
            }
            .metric-label {
                font-size: 14px;
                color: #666;
            }
            .section {
                margin: 30px 0;
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .config-box {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                font-family: monospace;
                font-size: 12px;
                overflow-x: auto;
            }
            .footer {
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌀 Rapport d'analyse FPS</h1>
            
            <div class="section">
                <h2>📊 Métriques principales</h2>
                <div class="grid-container">
    """
    
    # Ajouter les métriques principales
    if 'metrics_summary' in all_data:
        for metric, value in all_data['metrics_summary'].items():
            if isinstance(value, (int, float)):
                # Formater intelligemment selon la valeur
                if abs(value) < 0.001 and value != 0:
                    # Notation scientifique pour les très petites valeurs
                    formatted_value = f"{value:.2e}"
                elif abs(value) >= 1000000:
                    # Notation scientifique pour les très grandes valeurs
                    formatted_value = f"{value:.2e}"
                elif int(value) == value:
                    # Entier
                    formatted_value = f"{int(value)}"
                else:
                    # Décimales normales
                    formatted_value = f"{value:.3f}"
                    
                html_content += f"""
                    <div class="metric-box">
                        <div class="metric-value">{formatted_value}</div>
                        <div class="metric-label">{metric}</div>
                    </div>
                """

    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Détection d'émergences</h2>
    """
    
    # Résumé des émergences
    if 'emergence_summary' in all_data:
        html_content += "<ul>"
        for event_type, count in all_data['emergence_summary'].items():
            html_content += f"<li><strong>{event_type}</strong> : {count} événements</li>"
        html_content += "</ul>"
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>⚙️ Configuration utilisée</h2>
                <div class="config-box">
    """
    
    # Configuration
    if 'config' in all_data:
        html_content += f"<pre>{json.dumps(all_data['config'], indent=2)}</pre>"
    
    html_content += """
                </div>
            </div>
            
            <div class="footer">
                <p>Généré le """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>FPS - Fractal Pulsating Spiral | © 2025 Gepetto & Andréa Gadal</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Écrire le fichier
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Rapport HTML généré : {output_path}")


# ============== UTILITAIRES ==============

def save_all_figures(figures: Dict[str, plt.Figure], output_dir: str) -> None:
    """
    Sauvegarde toutes les figures dans un dossier.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figures.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {output_path}")


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module visualize.py
    """
    print("=== Tests du module visualize.py ===\n")
    
    # Générer des données de test
    print("Test 1 - Génération de données synthétiques:")
    t = np.linspace(0, 100, 1000)
    
    # Signal FPS simulé
    S_fps = np.sin(2 * np.pi * t / 10) + 0.5 * np.sin(2 * np.pi * t / 3)
    C_fps = np.cos(2 * np.pi * t / 15)
    effort_fps = 0.5 + 0.3 * np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(len(t))
    
    # Créer un dictionnaire de données
    test_data = {
        'S(t)': S_fps,
        'C(t)': C_fps,
        'effort(t)': effort_fps,
        'cpu_step(t)': 0.01 + 0.005 * np.random.randn(len(t)),
        'entropy_S': 0.5 + 0.1 * np.sin(2 * np.pi * t / 30),
        'variance_d2S': 0.01 + 0.005 * np.random.randn(len(t)),
        'mean_abs_error': 0.2 * np.exp(-t/50),
        'effort_status': ['stable' if e < 0.7 else 'transitoire' if e < 0.9 else 'chronique' 
                         for e in effort_fps]
    }
    
    # Tester chaque fonction
    print("\nTest 2 - Évolution du signal:")
    fig1 = plot_signal_evolution(t, S_fps, "Test - Signal S(t)")
    
    print("\nTest 3 - Comparaison des strates:")
    An_test = np.array([1.0 + 0.1*np.sin(t/10), 0.8 + 0.2*np.cos(t/15), 1.2 - 0.1*np.sin(t/20)])
    fn_test = np.array([1.0 + 0.05*np.sin(t/25), 1.1 - 0.03*np.cos(t/30), 0.9 + 0.04*np.sin(t/35)])
    fig2 = plot_strata_comparison(t, An_test, fn_test)
    
    print("\nTest 4 - Tableau de bord:")
    fig3 = plot_metrics_dashboard(test_data)
    
    print("\nTest 5 - Grille empirique:")
    scores_test = {
        'Stabilité': 4,
        'Régulation': 3,
        'Fluidité': 5,
        'Résilience': 3,
        'Innovation': 4,
        'Coût CPU': 2,
        'Effort interne': 3
    }
    fig4 = create_empirical_grid_notebook(scores_test)
    
    print("\nTest 6 - Matrice de corrélation:")
    mapping_test = {
        'Stabilité': ['S(t)', 'C(t)', 'φₙ(t)'],
        'Régulation': ['Fₙ(t)', 'G(x)', 'γ(t)'],
        'Fluidité': ['γₙ(t)', 'σ(x)', 'envₙ(x,t)'],
        'Innovation': ['A_spiral(t)', 'Eₙ(t)', 'r(t)']
    }
    fig5 = generate_correlation_matrix(mapping_test)
    
    # Sauvegarder les figures
    print("\nTest 7 - Sauvegarde des figures:")
    figures = {
        'signal_evolution': fig1,
        'strata_comparison': fig2,
        'metrics_dashboard': fig3,
        'empirical_grid': fig4,
        'correlation_matrix': fig5
    }
    save_all_figures(figures, "test_visualizations")
    
    print("\n✅ Module visualize.py prêt à révéler la beauté de la danse FPS!")
    
    # Afficher une figure pour vérification
    plt.show()


def plot_adaptive_resilience(metrics_history: Union[Dict[str, List], List[Dict]], 
                           perturbation_type: str = 'none') -> plt.Figure:
    """
    Affiche la métrique de résilience adaptative unifiée.
    
    Utilise adaptive_resilience qui sélectionne automatiquement entre:
    - t_retour pour les perturbations ponctuelles (choc)
    - continuous_resilience pour les perturbations continues (sinus, bruit, rampe)
    
    Args:
        metrics_history: historique des métriques
        perturbation_type: type de perturbation (détecté automatiquement si adaptive_resilience présent)
    
    Returns:
        Figure matplotlib
    """
    # Convertir en format uniforme si nécessaire
    if isinstance(metrics_history, list) and len(metrics_history) > 0:
        keys = metrics_history[0].keys()
        history_dict = {k: [m.get(k, 0) for m in metrics_history] for k in keys}
    else:
        history_dict = metrics_history
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Priorité à adaptive_resilience si disponible
    if 'adaptive_resilience' in history_dict:
        data = history_dict['adaptive_resilience']
        scores = history_dict.get('adaptive_resilience_score', [3] * len(data))
        metric_name = "Résilience Adaptative"
        ylabel = "Score unifié [0-1]"
        description = "Métrique unifiée selon le type de perturbation"
        color = FPS_COLORS['primary']
        
        # Déterminer automatiquement le type depuis les scores
        if len(scores) > 0 and scores[0] is not None:
            # Si on a des scores, on peut déduire le type de perturbation utilisé
            perturbation_type = 'adaptatif'
    
    # Fallback sur les métriques individuelles
    elif perturbation_type == 'choc':
        # Perturbation ponctuelle : utiliser t_retour
        if 't_retour' in history_dict:
            data = history_dict['t_retour']
            # Normaliser t_retour en score [0-1]
            data = [1.0 / (1.0 + t) if t != 0 else 1.0 for t in data]
            metric_name = "Résilience (basée sur t_retour)"
            ylabel = "Score [0-1]"
            description = "Temps de récupération normalisé après perturbation ponctuelle"
            color = FPS_COLORS['warning']
        else:
            ax.text(0.5, 0.5, 'Données de résilience non disponibles', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
    
    elif perturbation_type in ['sinus', 'bruit', 'rampe']:
        # Perturbation continue : utiliser continuous_resilience
        if 'continuous_resilience' in history_dict:
            data = history_dict['continuous_resilience']
            metric_name = "Résilience Continue"
            ylabel = "Score [0-1]"
            description = "Capacité à maintenir la cohérence sous perturbation continue"
            color = FPS_COLORS['success']
        else:
            ax.text(0.5, 0.5, 'Données continuous_resilience non disponibles', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
    
    else:
        # Pas de perturbation ou type inconnu
        ax.text(0.5, 0.5, f'Type de perturbation "{perturbation_type}" non géré\n' + 
                'Utilisez adaptive_resilience pour une sélection automatique',
                transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Tracer la métrique
    time_steps = np.arange(len(data))
    ax.plot(time_steps, data, color=color, linewidth=2.5, alpha=0.8)
    
    # Ajouter une ligne de référence pour continuous_resilience
    if perturbation_type in ['sinus', 'bruit', 'rampe']:
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, 
                  label='Seuil acceptable (0.5)')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, 
                  label='Seuil excellent (0.8)')
        ax.set_ylim(-0.05, 1.05)
    
    # Mise en forme
    ax.set_title(f'{metric_name} - {description}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pas de temps')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # Statistiques dans un encadré
    mean_val = np.mean(data)
    std_val = np.std(data)
    final_val = data[-1] if len(data) > 0 else 0
    
    stats_text = f'Moyenne: {mean_val:.3f}\n'
    stats_text += f'Écart-type: {std_val:.3f}\n'
    stats_text += f'Valeur finale: {final_val:.3f}'
    
    # Ajouter interprétation pour continuous_resilience
    if perturbation_type in ['sinus', 'bruit', 'rampe']:
        if final_val >= 0.8:
            interpretation = "Excellente résilience"
            interp_color = 'green'
        elif final_val >= 0.5:
            interpretation = "Résilience acceptable"
            interp_color = 'orange'
        else:
            interpretation = "Résilience faible"
            interp_color = 'red'
        stats_text += f'\n\nInterprétation: {interpretation}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=10)
    
    # Légende si lignes de référence
    if perturbation_type in ['sinus', 'bruit', 'rampe']:
        ax.legend(loc='lower right')
    
    # Ajouter note sur le type de perturbation
    ax.text(0.98, 0.02, f'Perturbation: {perturbation_type}', 
           transform=ax.transAxes, ha='right', va='bottom',
           style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig

# Gepetto & Claude & Andréa Gadal 🌀