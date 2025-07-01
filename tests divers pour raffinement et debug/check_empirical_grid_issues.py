import json
import pandas as pd
from pathlib import Path
import numpy as np

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

def analyze_empirical_grid_calculation(mode_name, folder_name):
    """Analyser comment les scores empiriques sont calculés pour chaque mode"""
    base_path = Path(f'fps_pipeline_output/{folder_name}')
    
    # Lire le rapport de comparaison
    json_path = base_path / 'reports' / 'comparison_fps_vs_controls.json'
    with open(json_path, 'r') as f:
        comparison_data = json.load(f)
    
    # Lire un CSV pour obtenir les métriques brutes
    csv_files = list((base_path / 'logs').glob('batch_run_0_*.csv'))
    if not csv_files:
        return None
    
    df = pd.read_csv(csv_files[0])
    
    # Analyser les composants du score empirique
    detailed_metrics = comparison_data['detailed_metrics']
    
    return {
        'mode': mode_name,
        # Scores individuels
        'synchronization': detailed_metrics['synchronization']['fps_value'],
        'stability': detailed_metrics['stability']['fps_value'],
        'resilience': detailed_metrics['resilience']['fps_value'],
        'continuous_resilience': detailed_metrics['continuous_resilience']['fps_value'],
        'innovation': detailed_metrics['innovation']['fps_value'],
        'fluidity': detailed_metrics['fluidity']['fps_value'],
        'cpu_efficiency': detailed_metrics['cpu_efficiency']['fps_value'],
        'global_score': detailed_metrics['global_score']['fps'],
        
        # Métriques brutes qui devraient influencer les scores
        'gamma_mean_avg': df['gamma_mean(t)'].mean(),
        'effort_mean': df['effort(t)'].mean(),
        'variance_d2S': df['variance_d2S'].mean(),
        'mean_high_effort': df['mean_high_effort'].mean(),
        
        # Métriques de stabilité
        'std_S': df['S(t)'].std(),
        'entropy_S': df['entropy_S'].mean(),
        
        # Métriques temporelles
        't_retour_mean': df['t_retour'].mean(),
        'continuous_resilience_mean': df['continuous_resilience'].mean()
    }

# Analyser tous les modes
results = []
for mode_name, folder_name in gamma_modes.items():
    print(f"Analyse du mode {mode_name}...")
    mode_analysis = analyze_empirical_grid_calculation(mode_name, folder_name)
    if mode_analysis:
        results.append(mode_analysis)

# Créer un DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("ANALYSE DES PROBLEMES DE LA GRILLE EMPIRIQUE")
print("="*80)

# 1. Identifier les métriques qui ne varient pas
print("\n1. METRIQUES QUI NE VARIENT PAS (PROBLEME!):")
for col in results_df.columns:
    if col != 'mode':
        unique_values = results_df[col].nunique()
        if unique_values == 1:
            print(f"   {col:30s}: TOUJOURS {results_df[col].iloc[0]:.6f}")

# 2. Métriques qui varient peu
print("\n2. METRIQUES QUI VARIENT PEU (std < 0.01):")
for col in results_df.columns:
    if col != 'mode':
        std_val = results_df[col].std()
        if 0 < std_val < 0.01:
            print(f"   {col:30s}: std = {std_val:.6f}")

# 3. Corrélations entre métriques brutes et scores
print("\n3. CORRELATIONS ENTRE METRIQUES BRUTES ET SCORES EMPIRIQUES:")
raw_metrics = ['gamma_mean_avg', 'effort_mean', 'variance_d2S', 'mean_high_effort', 'std_S']
score_metrics = ['synchronization', 'stability', 'resilience', 'innovation', 'fluidity', 'global_score']

print(f"\n{'Métrique brute':20s} | {'Score':20s} | {'Corrélation':>12s}")
print("-"*60)

for raw in raw_metrics:
    for score in score_metrics:
        if results_df[raw].std() > 0 and results_df[score].std() > 0:
            corr = results_df[raw].corr(results_df[score])
            if abs(corr) > 0.3:  # Seulement les corrélations significatives
                print(f"{raw:20s} | {score:20s} | {corr:12.4f}")

# 4. Analyse du problème de fluidity
print("\n4. ANALYSE DU PROBLEME DE FLUIDITY:")
print(f"   Fluidity est basé sur variance_d2S")
print(f"   Range variance_d2S: {results_df['variance_d2S'].min():.6f} - {results_df['variance_d2S'].max():.6f}")
print(f"   Std variance_d2S: {results_df['variance_d2S'].std():.6f}")
print(f"   Mais fluidity score est toujours: {results_df['fluidity'].iloc[0]:.6f}")
print(f"\n   DIAGNOSTIC: La formule fluidity = 1/(1+variance_d2S) ne capture pas les variations")
print(f"   car variance_d2S est très petite (~0.01) pour tous les modes")

# 5. Proposer des corrections
print("\n5. PROPOSITIONS DE CORRECTION:")
print("\n   a) Pour FLUIDITY:")
print("      - Inclure gamma dans le calcul: fluidity = f(variance_d2S, gamma_stability)")
print("      - Ou utiliser une métrique plus sensible que variance_d2S")
print("      - Ou ajuster la formule de normalisation")

print("\n   b) Pour EFFORT:")
print("      - L'effort varie bien entre modes mais n'est pas dans la grille empirique")
print("      - Ajouter un critère 'efficiency' basé sur effort_mean")

# 6. Vérifier quelles métriques contribuent au score global
print("\n6. CONTRIBUTION AU SCORE GLOBAL:")
# Calculer quelle proportion du score global vient de chaque métrique
# En supposant une moyenne simple (ce qui semble être le cas)
score_components = ['synchronization', 'stability', 'resilience', 'continuous_resilience', 
                   'innovation', 'fluidity', 'cpu_efficiency']

for mode in results_df['mode'].unique():
    mode_data = results_df[results_df['mode'] == mode].iloc[0]
    print(f"\n   Mode {mode}:")
    
    # Normaliser chaque composant par rapport aux autres modes
    for comp in score_components:
        comp_values = results_df[comp].values
        if comp_values.std() > 0:
            normalized = (mode_data[comp] - comp_values.min()) / (comp_values.max() - comp_values.min())
        else:
            normalized = 0.5
        
        print(f"      {comp:20s}: {normalized:.3f} (raw: {mode_data[comp]:.3f})")

# Sauvegarder l'analyse
results_df.to_csv('empirical_grid_analysis.csv', index=False)
print(f"\nAnalyse sauvegardée dans empirical_grid_analysis.csv") 