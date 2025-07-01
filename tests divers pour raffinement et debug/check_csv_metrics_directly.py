import pandas as pd
import numpy as np
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

def analyze_csv_metrics(mode_name, folder_name):
    """Analyser les métriques directement depuis les CSV"""
    base_path = Path(f'fps_pipeline_output/{folder_name}/logs')
    
    # Trouver tous les fichiers batch_run_*
    csv_files = list(base_path.glob('batch_run_*.csv'))
    
    if not csv_files:
        print(f"  Pas de fichiers CSV trouvés pour {mode_name}")
        return None
    
    all_metrics = []
    
    # Analyser chaque fichier batch_run
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Extraire les métriques qui nous intéressent
        metrics = {
            'file': csv_file.name,
            # Métriques qui devraient varier si elles prennent en compte S(t)
            'variance_d2S_mean': df['variance_d2S'].mean(),
            'variance_d2S_std': df['variance_d2S'].std(),
            'variance_d2S_min': df['variance_d2S'].min(),
            'variance_d2S_max': df['variance_d2S'].max(),
            
            # Synchronisation (basée sur C(t))
            'C_mean': df['C(t)'].mean(),
            'C_std': df['C(t)'].std(),
            'C_final': df['C(t)'].iloc[-1],
            
            # Continuous resilience
            'continuous_resilience_mean': df['continuous_resilience'].mean(),
            'continuous_resilience_std': df['continuous_resilience'].std(),
            'continuous_resilience_unique': df['continuous_resilience'].nunique(),
            
            # Signal S(t) pour vérifier qu'il varie bien
            'S_mean': df['S(t)'].mean(),
            'S_std': df['S(t)'].std(),
            'S_range': df['S(t)'].max() - df['S(t)'].min(),
            
            # Gamma pour confirmation
            'gamma_mean': df['gamma_mean(t)'].mean(),
            'gamma_range': df['gamma_mean(t)'].max() - df['gamma_mean(t)'].min()
        }
        
        all_metrics.append(metrics)
    
    # Moyenner sur tous les batch_run
    avg_metrics = {
        'mode': mode_name,
        'n_files': len(all_metrics),
        'variance_d2S_avg': np.mean([m['variance_d2S_mean'] for m in all_metrics]),
        'variance_d2S_range': np.mean([m['variance_d2S_max'] - m['variance_d2S_min'] for m in all_metrics]),
        'C_avg': np.mean([m['C_mean'] for m in all_metrics]),
        'C_std_avg': np.mean([m['C_std'] for m in all_metrics]),
        'C_final_avg': np.mean([m['C_final'] for m in all_metrics]),
        'continuous_resilience_avg': np.mean([m['continuous_resilience_mean'] for m in all_metrics]),
        'continuous_resilience_std_avg': np.mean([m['continuous_resilience_std'] for m in all_metrics]),
        'continuous_resilience_unique_avg': np.mean([m['continuous_resilience_unique'] for m in all_metrics]),
        'S_std_avg': np.mean([m['S_std'] for m in all_metrics]),
        'S_range_avg': np.mean([m['S_range'] for m in all_metrics]),
        'gamma_mean_avg': np.mean([m['gamma_mean'] for m in all_metrics]),
        'gamma_range_avg': np.mean([m['gamma_range'] for m in all_metrics])
    }
    
    return avg_metrics

# Analyser tous les modes
print("="*80)
print("VERIFICATION DIRECTE DES METRIQUES DANS LES CSV")
print("="*80)

results = []
for mode_name, folder_name in gamma_modes.items():
    print(f"\nAnalyse du mode {mode_name}...")
    metrics = analyze_csv_metrics(mode_name, folder_name)
    if metrics:
        results.append(metrics)

# Créer un DataFrame pour l'analyse
results_df = pd.DataFrame(results)

# 1. Vérifier variance_d2S (base de fluidity)
print("\n1. VARIANCE_D2S (base du calcul de fluidity):")
print(f"{'Mode':20s} {'Moyenne':>12s} {'Range':>12s}")
print("-"*50)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['variance_d2S_avg']:12.2f} {row['variance_d2S_range']:12.2f}")

print(f"\nVariation entre modes:")
print(f"  Min: {results_df['variance_d2S_avg'].min():.2f}")
print(f"  Max: {results_df['variance_d2S_avg'].max():.2f}")
print(f"  Std: {results_df['variance_d2S_avg'].std():.2f}")

# 2. Vérifier C(t) (base de synchronization)
print("\n2. C(t) (base du calcul de synchronization):")
print(f"{'Mode':20s} {'Moyenne':>12s} {'Std':>12s} {'Final':>12s}")
print("-"*60)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['C_avg']:12.6f} {row['C_std_avg']:12.6f} {row['C_final_avg']:12.6f}")

print(f"\nVariation de C_avg entre modes:")
print(f"  Min: {results_df['C_avg'].min():.6f}")
print(f"  Max: {results_df['C_avg'].max():.6f}")
print(f"  Std: {results_df['C_avg'].std():.6f}")

# 3. Vérifier continuous_resilience
print("\n3. CONTINUOUS_RESILIENCE:")
print(f"{'Mode':20s} {'Moyenne':>12s} {'Std':>12s} {'# Uniques':>12s}")
print("-"*60)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['continuous_resilience_avg']:12.6f} {row['continuous_resilience_std_avg']:12.6f} {row['continuous_resilience_unique_avg']:12.1f}")

print(f"\nProblème détecté: continuous_resilience est-il toujours 1.0?")
print(f"  Toutes les moyennes sont: {results_df['continuous_resilience_avg'].unique()}")

# 4. Vérifier que S(t) varie bien entre modes
print("\n4. SIGNAL S(t) (devrait varier selon gamma):")
print(f"{'Mode':20s} {'Std':>12s} {'Range':>12s}")
print("-"*50)
for _, row in results_df.iterrows():
    print(f"{row['mode']:20s} {row['S_std_avg']:12.6f} {row['S_range_avg']:12.6f}")

print(f"\nVariation de S_std entre modes:")
print(f"  Min: {results_df['S_std_avg'].min():.6f}")
print(f"  Max: {results_df['S_std_avg'].max():.6f}")
print(f"  Ratio Max/Min: {results_df['S_std_avg'].max() / results_df['S_std_avg'].min():.2f}")

# 5. Corrélations
print("\n5. CORRELATIONS AVEC GAMMA:")
print(f"  gamma_mean vs variance_d2S: {results_df['gamma_mean_avg'].corr(results_df['variance_d2S_avg']):.4f}")
print(f"  gamma_mean vs C_avg: {results_df['gamma_mean_avg'].corr(results_df['C_avg']):.4f}")
print(f"  gamma_mean vs S_std: {results_df['gamma_mean_avg'].corr(results_df['S_std_avg']):.4f}")
print(f"  gamma_range vs variance_d2S: {results_df['gamma_range_avg'].corr(results_df['variance_d2S_avg']):.4f}")

# 6. Diagnostic du problème
print("\n6. DIAGNOSTIC DU PROBLEME:")
print("\nPour fluidity:")
if results_df['variance_d2S_avg'].std() < 10:
    print("  ❌ variance_d2S varie peu entre modes")
    print("  → La formule 1/(1+variance_d2S) donnera des valeurs très proches")
else:
    print("  ✓ variance_d2S varie entre modes")
    print("  → Mais la formule 1/(1+variance_d2S) écrase les différences")

print(f"\nExemple de calcul fluidity:")
for _, row in results_df.iterrows():
    fluidity = 1.0 / (1.0 + row['variance_d2S_avg'])
    print(f"  {row['mode']:20s}: 1/(1+{row['variance_d2S_avg']:.1f}) = {fluidity:.6f}")

print("\nPour synchronization:")
if results_df['C_avg'].std() < 0.001:
    print("  ❌ C(t) ne varie presque pas entre modes")
else:
    print("  ✓ C(t) varie entre modes mais très peu")

print("\nPour continuous_resilience:")
if results_df['continuous_resilience_unique_avg'].mean() == 1:
    print("  ❌ continuous_resilience est une constante (toujours 1.0)")
else:
    print("  ✓ continuous_resilience varie dans le temps")

# Sauvegarder les résultats
results_df.to_csv('csv_metrics_analysis.csv', index=False)
print(f"\nRésultats sauvegardés dans csv_metrics_analysis.csv") 