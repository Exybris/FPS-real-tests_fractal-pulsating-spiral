import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def analyze_gamma_mode(mode_name, folder_name):
    """Analyser les métriques pour un mode de gamma donné"""
    base_path = Path(f'fps_pipeline_output/{folder_name}')
    
    # Lire le fichier CSV principal
    csv_path = base_path / 'logs' / f'batch_run_0_{folder_name.replace("gamma_", "").replace("_run", "_run")}_seed12345.csv'
    if not csv_path.exists():
        # Essayer un autre pattern
        csv_files = list((base_path / 'logs').glob('batch_run_0_*.csv'))
        if csv_files:
            csv_path = csv_files[0]
        else:
            print(f"Pas de CSV trouvé pour {mode_name}")
            return None
    
    # Lire les données CSV
    df = pd.read_csv(csv_path)
    
    # Lire le rapport de comparaison
    json_path = base_path / 'reports' / 'comparison_fps_vs_controls.json'
    with open(json_path, 'r') as f:
        comparison_data = json.load(f)
    
    # Extraire les métriques clés
    metrics = {
        'mode': mode_name,
        'gamma_mean_avg': df['gamma_mean(t)'].mean(),
        'gamma_mean_std': df['gamma_mean(t)'].std(),
        'gamma_mean_final': df['gamma_mean(t)'].iloc[-1],
        'fluidity_score': comparison_data['detailed_metrics']['fluidity']['fps_value'],
        'effort_mean': df['effort(t)'].mean(),
        'effort_std': df['effort(t)'].std(),
        'effort_max': df['effort(t)'].max(),
        'empirical_global_score': comparison_data['detailed_metrics']['global_score']['fps'],
        'synchronization_score': comparison_data['detailed_metrics']['synchronization']['fps_value'],
        'stability_score': comparison_data['detailed_metrics']['stability']['fps_value'],
        'resilience_score': comparison_data['detailed_metrics']['resilience']['fps_value'],
        'innovation_score': comparison_data['detailed_metrics']['innovation']['fps_value']
    }
    
    # Analyser l'évolution temporelle
    metrics['gamma_trend'] = np.polyfit(df.index, df['gamma_mean(t)'], 1)[0]  # Pente de la tendance
    
    return metrics

# Analyser tous les modes
results = []
for mode_name, folder_name in gamma_modes.items():
    print(f"\nAnalyse du mode {mode_name}...")
    mode_metrics = analyze_gamma_mode(mode_name, folder_name)
    if mode_metrics:
        results.append(mode_metrics)

# Créer un DataFrame avec les résultats
results_df = pd.DataFrame(results)

# Afficher les résultats
print("\n" + "="*80)
print("COMPARAISON DES PERFORMANCES ENTRE LES MODES DE GAMMA")
print("="*80)

# Trier par score global empirique
results_df_sorted = results_df.sort_values('empirical_global_score', ascending=False)

print("\n1. SCORES GLOBAUX EMPIRIQUES:")
for _, row in results_df_sorted.iterrows():
    print(f"   {row['mode']:20s}: {row['empirical_global_score']:.6f}")

print("\n2. FLUIDITY SCORES:")
for _, row in results_df_sorted.iterrows():
    print(f"   {row['mode']:20s}: {row['fluidity_score']:.6f}")

print("\n3. EFFORT STATISTICS:")
print(f"{'Mode':20s} {'Mean':>15s} {'Std':>15s} {'Max':>15s}")
print("-"*70)
for _, row in results_df_sorted.iterrows():
    print(f"{row['mode']:20s} {row['effort_mean']:15.2f} {row['effort_std']:15.2f} {row['effort_max']:15.2f}")

print("\n4. GAMMA_MEAN STATISTICS:")
print(f"{'Mode':20s} {'Mean':>12s} {'Std':>12s} {'Final':>12s} {'Trend':>12s}")
print("-"*70)
for _, row in results_df_sorted.iterrows():
    print(f"{row['mode']:20s} {row['gamma_mean_avg']:12.6f} {row['gamma_mean_std']:12.6f} {row['gamma_mean_final']:12.6f} {row['gamma_trend']:12.6e}")

# Analyse des corrélations
print("\n5. ANALYSE DES CORRELATIONS:")
print("\nCorrélation entre gamma_mean_avg et les scores:")
for metric in ['empirical_global_score', 'fluidity_score', 'synchronization_score', 'stability_score']:
    corr = results_df['gamma_mean_avg'].corr(results_df[metric])
    print(f"   gamma_mean_avg vs {metric}: {corr:.4f}")

print("\nCorrélation entre effort_mean et les scores:")
for metric in ['empirical_global_score', 'fluidity_score', 'synchronization_score', 'stability_score']:
    corr = results_df['effort_mean'].corr(results_df[metric])
    print(f"   effort_mean vs {metric}: {corr:.4f}")

# Vérifier si les variations de gamma_mean se reflètent dans les scores
print("\n6. VARIATIONS GAMMA_MEAN vs SCORES EMPIRIQUES:")
gamma_range = results_df['gamma_mean_avg'].max() - results_df['gamma_mean_avg'].min()
score_range = results_df['empirical_global_score'].max() - results_df['empirical_global_score'].min()
print(f"   Range gamma_mean_avg: {gamma_range:.6f}")
print(f"   Range empirical_global_score: {score_range:.6f}")
print(f"   Ratio variation: {score_range/gamma_range if gamma_range > 0 else 'N/A':.2f}")

# Créer des visualisations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Gamma mean vs Global Score
ax1 = axes[0, 0]
ax1.scatter(results_df['gamma_mean_avg'], results_df['empirical_global_score'])
for i, row in results_df.iterrows():
    ax1.annotate(row['mode'], (row['gamma_mean_avg'], row['empirical_global_score']), fontsize=8)
ax1.set_xlabel('Gamma Mean Average')
ax1.set_ylabel('Empirical Global Score')
ax1.set_title('Gamma Mean vs Global Score')

# Plot 2: Effort Mean vs Fluidity
ax2 = axes[0, 1]
ax2.scatter(results_df['effort_mean'], results_df['fluidity_score'])
for i, row in results_df.iterrows():
    ax2.annotate(row['mode'], (row['effort_mean'], row['fluidity_score']), fontsize=8)
ax2.set_xlabel('Effort Mean')
ax2.set_ylabel('Fluidity Score')
ax2.set_title('Effort vs Fluidity')

# Plot 3: Comparaison des scores par mode
ax3 = axes[1, 0]
modes = results_df_sorted['mode'].tolist()
scores = results_df_sorted['empirical_global_score'].tolist()
ax3.bar(modes, scores)
ax3.set_xlabel('Gamma Mode')
ax3.set_ylabel('Empirical Global Score')
ax3.set_title('Global Scores by Gamma Mode')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Evolution temporelle pour quelques modes
ax4 = axes[1, 1]
sample_modes = ['static', 'dynamic', 'sinusoidal']
for mode_name in sample_modes:
    if mode_name in gamma_modes:
        folder_name = gamma_modes[mode_name]
        csv_path = Path(f'fps_pipeline_output/{folder_name}/logs')
        csv_files = list(csv_path.glob('batch_run_0_*.csv'))
        if csv_files:
            df_temp = pd.read_csv(csv_files[0])
            ax4.plot(df_temp['t'][:100], df_temp['gamma_mean(t)'][:100], label=mode_name, alpha=0.7)
ax4.set_xlabel('Time')
ax4.set_ylabel('Gamma Mean(t)')
ax4.set_title('Gamma Mean Evolution (first 100 steps)')
ax4.legend()

plt.tight_layout()
plt.savefig('gamma_modes_analysis.png', dpi=150)
plt.close()

# Sauvegarder les résultats dans un fichier
results_df.to_csv('gamma_modes_comparison.csv', index=False)
print(f"\nRésultats sauvegardés dans gamma_modes_comparison.csv et gamma_modes_analysis.png") 