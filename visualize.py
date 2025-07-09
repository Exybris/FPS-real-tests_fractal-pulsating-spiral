"""
visualize.py - Visualisation complète du système FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module donne des yeux à la danse spiralée FPS :
- Évolution temporelle des signaux
- Comparaisons entre strates
- Diagrammes de phase
- Tableaux de bord interactifs
- Grille empirique avec notation visuelle
- Animations de l'évolution spiralée
- Comparaisons FPS vs Kuramoto
- Matrices de corrélation
- Rapports HTML complets

La visualisation est le miroir qui permet de voir l'invisible,
de comprendre l'émergence et de partager la beauté du système.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import defaultdict

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
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
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
    ax8 = fig.add_subplot(gs[3, :])  # Prend toute la largeur de la dernière ligne
    
    # Tracer En_mean et On_mean
    if 'En_mean(t)' in history_dict and 'On_mean(t)' in history_dict:
        t_array = np.arange(len(history_dict['En_mean(t)']))
        ax8.plot(t_array, history_dict['En_mean(t)'], 'g--', linewidth=2, 
                 label='Eₙ (attendu)', alpha=0.8)
        ax8.plot(t_array, history_dict['On_mean(t)'], 'b-', linewidth=2, 
                 label='Oₙ (observé)', alpha=0.8)
        
        # Ajouter In_mean si disponible
        if 'In_mean(t)' in history_dict:
            ax8.plot(t_array, history_dict['In_mean(t)'], 'r:', linewidth=2,
                     label='Iₙ (input)', alpha=0.8)
        
        # Ajouter An_mean et fn_mean si disponibles
        if 'An_mean(t)' in history_dict:
            ax8.plot(t_array, history_dict['An_mean(t)'], 'm-', linewidth=1.5,
                     label='Aₙ (amplitude)', alpha=0.7)
        
        if 'fn_mean(t)' in history_dict:
            # Créer un axe secondaire pour fn car échelle différente
            ax8_twin = ax8.twinx()
            ax8_twin.plot(t_array, history_dict['fn_mean(t)'], 'c-', linewidth=1.5,
                         label='fₙ (fréquence)', alpha=0.7)
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
                            color='orange', alpha=0.3, label='γ (latence)')
        
        ax8.set_title('Dynamique complète : Iₙ → Aₙ → Oₙ/Eₙ avec fₙ et γ', fontweight='bold')
        ax8.set_xlabel('Temps')
        ax8.set_ylabel('Amplitude')
        ax8.legend(loc='upper left')
        ax8.grid(True, alpha=0.3)
    
    # 9. Résumé statistique (déplacé)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculer les statistiques
    stats_text = "📊 Statistiques globales\n\n"
    
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
    
    # Titre global
    fig.suptitle('Tableau de bord FPS - Vue d\'ensemble', fontsize=16, fontweight='bold')
    
    return fig


# ============== GRILLE EMPIRIQUE ==============

def create_empirical_grid(scores_dict: Dict[str, int]) -> plt.Figure:
    """
    Crée une grille empirique avec notation visuelle (1-5).
    
    Args:
        scores_dict: dictionnaire {critère: note} avec notes de 1 à 5
    
    Returns:
        Figure matplotlib
    """
    
    # Définition des icônes et couleurs
    score_config = {
        1: {'icon': '✖', 'color': '#C73E1D', 'label': 'Rupture/Chaotique'},
        2: {'icon': '▲', 'color': '#FF6B35', 'label': 'Instable'},
        3: {'icon': '●', 'color': '#FFC43D', 'label': 'Fonctionnel'},
        4: {'icon': '✔', 'color': '#87BE3F', 'label': 'Harmonieux'},
        5: {'icon': '∞', 'color': '#2E86AB', 'label': 'FPS-idéal'}
    }
    
    # Critères dans l'ordre de la grille
    criteria = ['Stabilité', 'Régulation', 'Fluidité', 'Résilience', 
                'Innovation', 'Coût CPU', 'Effort interne']
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Créer la grille
    n_criteria = len(criteria)
    y_positions = np.arange(n_criteria)
    
    # Fond alternant
    for i in range(n_criteria):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, alpha=0.1, color='gray')
    
    # Placer les scores
    for i, criterion in enumerate(criteria):
        score = scores_dict.get(criterion, 3)  # Default à 3 si non défini
        config = score_config[score]
        
        # Nom du critère
        ax.text(0, i, criterion, fontsize=12, va='center', ha='left', 
                fontweight='bold')
        
        # Score visuel
        ax.text(0.5, i, config['icon'], fontsize=24, va='center', ha='center',
                color=config['color'], fontweight='bold')
        
        # Barre de progression
        ax.barh(i, score/5, left=0.6, height=0.6, 
                color=config['color'], alpha=0.6)
        
        # Valeur numérique
        ax.text(1.2, i, f"{score}/5", fontsize=11, va='center', ha='center')
        
        # Description
        ax.text(1.4, i, config['label'], fontsize=10, va='center', ha='left',
                style='italic', alpha=0.8)
    
    # Configuration des axes
    ax.set_xlim(-0.1, 2.5)
    ax.set_ylim(-0.5, n_criteria - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    
    # Titre
    ax.set_title('Grille d\'évaluation empirique FPS', fontsize=16, 
                 fontweight='bold', pad=20)
    
    # Légende des scores
    legend_y = -1.5
    for score, config in score_config.items():
        ax.text(0.2 + (score-1)*0.5, legend_y, config['icon'], 
                fontsize=20, ha='center', color=config['color'])
        ax.text(0.2 + (score-1)*0.5, legend_y - 0.3, str(score), 
                fontsize=10, ha='center')
    
    ax.text(0.2, legend_y - 0.6, 'Légende:', fontsize=10, fontweight='bold')
    
    # Cadre
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


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


# ============== COMPARAISON FPS VS KURAMOTO ==============

def plot_fps_vs_kuramoto(fps_data: Dict[str, np.ndarray], 
                         kuramoto_data: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Compare les résultats FPS et Kuramoto.
    
    Args:
        fps_data: données du run FPS
        kuramoto_data: données du run Kuramoto
    
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Signaux globaux
    ax1 = axes[0, 0]
    
    # Gérer les tableaux potentiellement vides ou de longueurs différentes
    if 'S(t)' in fps_data and len(fps_data['S(t)']) > 0:
        t_fps = np.arange(len(fps_data['S(t)']))
        ax1.plot(t_fps, fps_data['S(t)'], 'b-', linewidth=2, 
                 label='FPS', alpha=0.8)
    
    if 'S(t)' in kuramoto_data and len(kuramoto_data['S(t)']) > 0:
        t_kura = np.arange(len(kuramoto_data['S(t)']))
        ax1.plot(t_kura, kuramoto_data['S(t)'], 'r--', linewidth=2, 
                 label='Kuramoto', alpha=0.8)
    
    ax1.set_title('Signal global S(t)', fontweight='bold')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Amplitude')
    # Ajouter légende seulement s'il y a des courbes
    handles1, labels1 = ax1.get_legend_handles_labels()
    if handles1:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coefficient d'accord
    ax2 = axes[0, 1]
    
    if 'C(t)' in fps_data and len(fps_data['C(t)']) > 0:
        t_fps_c = np.arange(len(fps_data['C(t)']))
        ax2.plot(t_fps_c, fps_data['C(t)'], 'b-', linewidth=2, 
                 label='FPS', alpha=0.8)
    
    if 'C(t)' in kuramoto_data and len(kuramoto_data['C(t)']) > 0:
        t_kura_c = np.arange(len(kuramoto_data['C(t)']))
        ax2.plot(t_kura_c, kuramoto_data['C(t)'], 'r--', linewidth=2, 
                 label='Kuramoto', alpha=0.8)
    
    ax2.set_title('Coefficient d\'accord C(t)', fontweight='bold')
    ax2.set_xlabel('Temps')
    ax2.set_ylabel('Coefficient')
    ax2.set_ylim(-1.1, 1.1)
    # Légende conditionnelle
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Effort/CPU
    ax3 = axes[1, 0]
    
    # Compteur de courbes pour légende
    effort_plotted = False
    
    if 'effort(t)' in fps_data and len(fps_data.get('effort(t)', [])) > 0:
        t_fps_e = np.arange(len(fps_data['effort(t)']))
        ax3.plot(t_fps_e, fps_data['effort(t)'], 'b-', linewidth=2, 
                 label='Effort FPS', alpha=0.8)
        effort_plotted = True
    
    # Ajout effort Kuramoto s'il existe
    if 'effort(t)' in kuramoto_data and len(kuramoto_data.get('effort(t)', [])) > 0:
        t_kura_e = np.arange(len(kuramoto_data['effort(t)']))
        # Vérifier si les valeurs ne sont pas toutes nulles
        if np.any(kuramoto_data['effort(t)'] != 0):
            ax3.plot(t_kura_e, kuramoto_data['effort(t)'], 'r--', linewidth=2, 
                     label='Effort Kuramoto', alpha=0.8)
            effort_plotted = True
    
    # CPU sur axe y droit
    if 'cpu_step(t)' in fps_data or 'cpu_step(t)' in kuramoto_data:
        ax3_twin = ax3.twinx()
        
        if 'cpu_step(t)' in fps_data and len(fps_data.get('cpu_step(t)', [])) > 0:
            t_fps_cpu = np.arange(len(fps_data['cpu_step(t)']))
            ax3_twin.plot(t_fps_cpu, fps_data['cpu_step(t)'], 'b:', linewidth=2, 
                          label='CPU FPS', alpha=0.6)
        
        if 'cpu_step(t)' in kuramoto_data and len(kuramoto_data.get('cpu_step(t)', [])) > 0:
            t_kura_cpu = np.arange(len(kuramoto_data['cpu_step(t)']))
            ax3_twin.plot(t_kura_cpu, kuramoto_data['cpu_step(t)'], 'r:', linewidth=2, 
                          label='CPU Kuramoto', alpha=0.6)
        
        ax3_twin.set_ylabel('CPU (s)')
        # Légende pour axe CPU aussi
        handles_cpu, labels_cpu = ax3_twin.get_legend_handles_labels()
        if handles_cpu:
            ax3_twin.legend(loc='upper right')
    
    ax3.set_title('Effort et coût CPU', fontweight='bold')
    ax3.set_xlabel('Temps')
    ax3.set_ylabel('Effort')
    # Légende effort seulement s'il y a des courbes
    handles3, labels3 = ax3.get_legend_handles_labels()
    if handles3:
        ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Métriques comparatives
    ax4 = axes[1, 1]
    metrics_names = ['Mean S(t)', 'Std S(t)', 'Mean CPU', 'Final C(t)']
    
    # Calculer les métriques
    fps_metrics = []
    kura_metrics = []
    
    # Mean et Std S(t)
    if 'S(t)' in fps_data and len(fps_data['S(t)']) > 0:
        fps_metrics.extend([np.mean(fps_data['S(t)']), np.std(fps_data['S(t)'])])
    else:
        fps_metrics.extend([0, 0])
    
    if 'S(t)' in kuramoto_data and len(kuramoto_data['S(t)']) > 0:
        kura_metrics.extend([np.mean(kuramoto_data['S(t)']), np.std(kuramoto_data['S(t)'])])
    else:
        kura_metrics.extend([0, 0])
    
    # Mean CPU - convertir en microsecondes pour visibilité
    if 'cpu_step(t)' in fps_data and len(fps_data.get('cpu_step(t)', [])) > 0:
        fps_cpu_mean = np.mean(fps_data['cpu_step(t)']) * 1e6  # Convertir en μs
        fps_metrics.append(fps_cpu_mean)
    else:
        fps_metrics.append(0)
    
    if 'cpu_step(t)' in kuramoto_data and len(kuramoto_data.get('cpu_step(t)', [])) > 0:
        kura_cpu_mean = np.mean(kuramoto_data['cpu_step(t)']) * 1e6  # Convertir en μs
        kura_metrics.append(kura_cpu_mean)
    else:
        kura_metrics.append(0)
    
    # Mettre à jour le label pour indiquer les microsecondes
    metrics_names[2] = 'Mean CPU (μs)'
    
    # Final C(t)
    if 'C(t)' in fps_data and len(fps_data['C(t)']) > 0:
        fps_metrics.append(fps_data['C(t)'][-1])
    else:
        fps_metrics.append(0)
    
    if 'C(t)' in kuramoto_data and len(kuramoto_data['C(t)']) > 0:
        kura_metrics.append(kuramoto_data['C(t)'][-1])
    else:
        kura_metrics.append(0)
    
    # Barres comparatives
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax4.bar(x - width/2, fps_metrics, width, label='FPS', 
            color=FPS_COLORS['primary'], alpha=0.8)
    ax4.bar(x + width/2, kura_metrics, width, label='Kuramoto', 
            color=FPS_COLORS['danger'], alpha=0.8)
    
    ax4.set_title('Métriques comparatives', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Titre principal
    fig.suptitle('Comparaison FPS vs Kuramoto', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


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
    fig4 = create_empirical_grid(scores_test)
    
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
