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

def analyze_detailed_metrics(mode_name, folder_name):
    """Analyser les métriques détaillées pour comprendre la stagnation"""
    base_path = Path(f'fps_pipeline_output/{folder_name}')
    
    # Lire le fichier CSV principal
    csv_files = list((base_path / 'logs').glob('batch_run_0_*.csv'))
    if not csv_files:
        return None
    
    df = pd.read_csv(csv_files[0])
    
    # Lire le rapport de comparaison
    json_path = base_path / 'reports' / 'comparison_fps_vs_controls.json'
    with open(json_path, 'r') as f:
        comparison_data = json.load(f)
    
    # Analyser les métriques temporelles détaillées
    metrics = {
        'mode': mode_name,
        # Gamma statistics
        'gamma_actual_min': df['gamma'].min(),
        'gamma_actual_max': df['gamma'].max(),
        'gamma_mean_min': df['gamma_mean(t)'].min(),
        'gamma_mean_max': df['gamma_mean(t)'].max(),
        'gamma_mean_range': df['gamma_mean(t)'].max() - df['gamma_mean(t)'].min(),
        
        # Effort temporal analysis
        'effort_min': df['effort(t)'].min(),
        'effort_max': df['effort(t)'].max(),
        'effort_median': df['effort(t)'].median(),
        'effort_q25': df['effort(t)'].quantile(0.25),
        'effort_q75': df['effort(t)'].quantile(0.75),
        'effort_above_1_percent': (df['effort(t)'] > 1.0).sum() / len(df) * 100,
        
        # Fluidity components (from the CSV)
        'mean_high_effort_avg': df['mean_high_effort'].mean(),
        'mean_high_effort_final': df['mean_high_effort'].iloc[-1],
        
        # Signal dynamics
        'signal_std': df['S(t)'].std(),
        'signal_range': df['S(t)'].max() - df['S(t)'].min(),
        
        # Empirical scores
        'fluidity_score': comparison_data['detailed_metrics']['fluidity']['fps_value'],
        'empirical_global_score': comparison_data['detailed_metrics']['global_score']['fps'],
        
        # CPU efficiency
        'cpu_efficiency_score': comparison_data['detailed_metrics']['cpu_efficiency']['fps_value'],
        'cpu_step_mean': df['cpu_step(t)'].mean()
    }
    
    # Calculer la distribution de l'effort
    effort_bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
    effort_hist, _ = np.histogram(df['effort(t)'], bins=effort_bins)
    for i, (low, high) in enumerate(zip(effort_bins[:-1], effort_bins[1:])):
        label = f'effort_{low}_{high if high != np.inf else "inf"}'
        metrics[label] = effort_hist[i] / len(df) * 100
    
    return metrics

# Analyser tous les modes
results = []
for mode_name, folder_name in gamma_modes.items():
    print(f"Analyse du mode {mode_name}...")
    mode_metrics = analyze_detailed_metrics(mode_name, folder_name)
    if mode_metrics:
        results.append(mode_metrics)

# Créer un DataFrame avec les résultats
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("ANALYSE DETAILLEE DE LA STAGNATION FLUIDITY/EFFORT")
print("="*80)

# Afficher les scores de fluidity (tous identiques ?)
print("\n1. FLUIDITY SCORES (PROBLEME DE STAGNATION):")
for _, row in results_df.iterrows():
    print(f"   {row['mode']:20s}: {row['fluidity_score']:.10f}")

print("\n2. GAMMA VARIATIONS REELLES:")
print(f"{'Mode':20s} {'gamma_mean_range':>15s} {'gamma_min':>10s} {'gamma_max':>10s}")
print("-"*65)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['gamma_mean_range']:15.6f} {row['gamma_mean_min']:10.6f} {row['gamma_mean_max']:10.6f}")

print("\n3. EFFORT DISTRIBUTION DETAILLEE:")
print(f"{'Mode':20s} {'Min':>10s} {'Q25':>10s} {'Median':>10s} {'Q75':>10s} {'Max':>10s}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['effort_min']:10.4f} {row['effort_q25']:10.4f} {row['effort_median']:10.4f} {row['effort_q75']:10.4f} {row['effort_max']:10.4f}")

print("\n4. EFFORT DISTRIBUTION PAR BINS (%):")
print(f"{'Mode':20s} {'0-0.5':>8s} {'0.5-1':>8s} {'1-2':>8s} {'2-5':>8s} {'5-10':>8s} {'>10':>8s}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['effort_0_0.5']:8.2f} {row['effort_0.5_1.0']:8.2f} {row['effort_1.0_2.0']:8.2f} {row['effort_2.0_5.0']:8.2f} {row['effort_5.0_10.0']:8.2f} {row['effort_10.0_inf']:8.2f}")

print("\n5. MEAN_HIGH_EFFORT ANALYSIS:")
print(f"{'Mode':20s} {'Average':>12s} {'Final':>12s}")
print("-"*45)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['mean_high_effort_avg']:12.6f} {row['mean_high_effort_final']:12.6f}")

# Analyser pourquoi fluidity est identique
print("\n6. ANALYSE DU PROBLEME DE FLUIDITY:")
print("   Toutes les valeurs de fluidity sont identiques, ce qui suggère:")
print("   - Soit le calcul de fluidity ne prend pas en compte gamma")
print("   - Soit il y a un problème dans la grille empirique")

# Vérifier les différences dans d'autres métriques
print("\n7. METRIQUES QUI VARIENT ENTRE MODES:")
varying_metrics = []
for col in results_df.columns:
    if col != 'mode' and results_df[col].std() > 0.001:
        varying_metrics.append((col, results_df[col].std()))

varying_metrics.sort(key=lambda x: x[1], reverse=True)
for metric, std in varying_metrics[:10]:
    print(f"   {metric:30s}: std = {std:.6f}")

# Créer des visualisations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Effort distribution par mode
ax1 = axes[0, 0]
modes = results_df['mode'].tolist()
effort_means = results_df['effort_median'].tolist()
ax1.bar(modes, effort_means)
ax1.set_xlabel('Gamma Mode')
ax1.set_ylabel('Effort Median')
ax1.set_title('Effort Median by Gamma Mode')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Gamma mean range
ax2 = axes[0, 1]
gamma_ranges = results_df['gamma_mean_range'].tolist()
ax2.bar(modes, gamma_ranges)
ax2.set_xlabel('Gamma Mode')
ax2.set_ylabel('Gamma Mean Range')
ax2.set_title('Gamma Mean Variation Range')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Signal dynamics
ax3 = axes[0, 2]
signal_stds = results_df['signal_std'].tolist()
ax3.bar(modes, signal_stds)
ax3.set_xlabel('Gamma Mode')
ax3.set_ylabel('Signal Std Dev')
ax3.set_title('Signal Variability by Mode')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Effort vs Global Score
ax4 = axes[1, 0]
ax4.scatter(results_df['effort_median'], results_df['empirical_global_score'])
for i, row in results_df.iterrows():
    ax4.annotate(row['mode'], (row['effort_median'], row['empirical_global_score']), fontsize=8)
ax4.set_xlabel('Effort Median')
ax4.set_ylabel('Empirical Global Score')
ax4.set_title('Effort vs Global Score')

# Plot 5: Gamma range vs Global Score
ax5 = axes[1, 1]
ax5.scatter(results_df['gamma_mean_range'], results_df['empirical_global_score'])
for i, row in results_df.iterrows():
    ax5.annotate(row['mode'], (row['gamma_mean_range'], row['empirical_global_score']), fontsize=8)
ax5.set_xlabel('Gamma Mean Range')
ax5.set_ylabel('Empirical Global Score')
ax5.set_title('Gamma Variation vs Global Score')

# Plot 6: CPU efficiency
ax6 = axes[1, 2]
cpu_scores = results_df['cpu_efficiency_score'].tolist()
ax6.bar(modes, cpu_scores)
ax6.set_xlabel('Gamma Mode')
ax6.set_ylabel('CPU Efficiency Score')
ax6.set_title('CPU Efficiency by Mode')
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('stagnation_analysis.png', dpi=150)
plt.close()

# Examiner la grille empirique
print("\n8. HYPOTHESE SUR LA GRILLE EMPIRIQUE:")
print("   La grille empirique pourrait ne pas capturer les variations de gamma correctement.")
print("   Vérifions les composants du score global qui varient:")

# Calculer les variations pour chaque composant
components = ['fluidity_score', 'cpu_efficiency_score', 'empirical_global_score']
for comp in components:
    values = results_df[comp].unique()
    print(f"\n   {comp}: {len(values)} valeurs uniques")
    if len(values) <= 7:
        print(f"      Valeurs: {sorted(values)}")

# Sauvegarder les résultats
results_df.to_csv('stagnation_analysis.csv', index=False)
print(f"\nRésultats détaillés sauvegardés dans stagnation_analysis.csv et stagnation_analysis.png") 