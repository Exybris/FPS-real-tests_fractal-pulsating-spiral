#!/usr/bin/env python3
"""
Analyse des corrélations entre architecture temporelle et performance FPS.

Explore si certaines signatures temporelles (ratios de tau, cohérence)
prédisent les moments d'excellence du système.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_fps_data(csv_path):
    """Charge et nettoie les données FPS."""
    try:
        df = pd.read_csv(csv_path)
        
        # Convertir les colonnes numériques
        numeric_cols = ['t', 'S(t)', 'fluidity', 'entropy_S', 'temporal_coherence',
                       'adaptive_resilience', 'effort(t)', 'cpu_step(t)',
                       'tau_S', 'tau_gamma', 'tau_A_mean', 'tau_f_mean']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Erreur chargement CSV: {e}")
        return None


def compute_performance_score(df):
    """
    Calcule un score de performance composite basé sur les métriques clés.
    
    Performance = pondération de fluidité, résilience, innovation, 
                  effort optimal, cpu efficient
    """
    # Normaliser chaque métrique entre 0 et 1
    scores = {}
    
    if 'fluidity' in df.columns:
        scores['fluidity'] = df['fluidity']  # Déjà 0-1
    
    if 'adaptive_resilience' in df.columns:
        scores['resilience'] = df['adaptive_resilience']  # Déjà 0-1
    
    if 'entropy_S' in df.columns:
        scores['innovation'] = df['entropy_S']  # Déjà 0-1
    
    if 'effort(t)' in df.columns:
        # Effort optimal = ni trop bas ni trop haut (courbe en cloche)
        effort_norm = (df['effort(t)'] - df['effort(t)'].min()) / (df['effort(t)'].max() - df['effort(t)'].min() + 1e-10)
        scores['effort_optimal'] = 1 - np.abs(effort_norm - 0.5) * 2  # Pic à 0.5
    
    if 'cpu_step(t)' in df.columns:
        # CPU efficace = faible coût
        cpu_norm = (df['cpu_step(t)'] - df['cpu_step(t)'].min()) / (df['cpu_step(t)'].max() - df['cpu_step(t)'].min() + 1e-10)
        scores['cpu_efficient'] = 1 - cpu_norm
    
    # Score composite
    weights = {
        'fluidity': 0.25,
        'resilience': 0.25, 
        'innovation': 0.2,
        'effort_optimal': 0.15,
        'cpu_efficient': 0.15
    }
    
    performance = np.zeros(len(df))
    total_weight = 0
    
    for metric, weight in weights.items():
        if metric in scores:
            performance += weight * scores[metric].fillna(0)
            total_weight += weight
    
    if total_weight > 0:
        performance = performance / total_weight
    
    return performance


def compute_temporal_features(df):
    """Calcule des features temporelles dérivées."""
    features = {}
    
    # Ratios de tau (révélateurs d'architecture)
    if all(col in df.columns for col in ['tau_gamma', 'tau_S']):
        features['ratio_gamma_S'] = df['tau_gamma'] / (df['tau_S'] + 1e-10)
    
    if all(col in df.columns for col in ['tau_f_mean', 'tau_A_mean']):
        features['ratio_f_A'] = df['tau_f_mean'] / (df['tau_A_mean'] + 1e-10)
    
    if all(col in df.columns for col in ['tau_f_mean', 'tau_S']):
        features['ratio_structure_surface'] = df['tau_f_mean'] / (df['tau_S'] + 1e-10)
    
    # Cohérence temporelle globale
    if 'temporal_coherence' in df.columns:
        features['coherence'] = df['temporal_coherence']
    
    # Stabilité des tau (variance glissante)
    tau_cols = ['tau_S', 'tau_gamma', 'tau_A_mean', 'tau_f_mean']
    available_taus = [col for col in tau_cols if col in df.columns]
    
    if available_taus:
        # Variance des tau (indicateur de stabilité architecture)
        tau_matrix = df[available_taus].values
        features['tau_stability'] = 1 / (1 + np.var(tau_matrix, axis=1))
        
        # Moyenne harmonique des tau (échelle temporelle globale)
        tau_matrix_safe = np.maximum(tau_matrix, 1e-10)
        features['tau_harmonic_mean'] = len(available_taus) / np.sum(1/tau_matrix_safe, axis=1)
    
    return pd.DataFrame(features, index=df.index)


def analyze_correlations(df, performance, temporal_features):
    """Analyse les corrélations entre features temporelles et performance."""
    
    print("=== ANALYSE DES CORRÉLATIONS TEMPORELLES-PERFORMANCE ===\n")
    
    # Combiner toutes les features
    all_features = temporal_features.copy()
    all_features['performance'] = performance
    
    # Matrice de corrélation
    corr_matrix = all_features.corr()
    perf_corrs = corr_matrix['performance'].drop('performance').sort_values(key=abs, ascending=False)
    
    print("🎯 Corrélations avec la Performance (ordre décroissant):")
    for feature, corr in perf_corrs.items():
        strength = "🔥" if abs(corr) > 0.5 else "⚡" if abs(corr) > 0.3 else "💫"
        print(f"  {strength} {feature:25}: {corr:+.4f}")
    
    # Analyse des moments d'excellence
    perf_percentiles = np.percentile(performance, [75, 90, 95])
    excellence_threshold = perf_percentiles[1]  # Top 10%
    
    excellent_moments = performance >= excellence_threshold
    print(f"\n🌟 Moments d'excellence identifiés: {excellent_moments.sum()}/{len(performance)} ({excellent_moments.mean()*100:.1f}%)")
    
    if excellent_moments.sum() > 5:  # Assez de données
        print(f"\n🔍 Caractéristiques des moments d'excellence:")
        
        for feature in temporal_features.columns:
            vals_excellent = temporal_features[feature][excellent_moments]
            vals_normal = temporal_features[feature][~excellent_moments]
            
            if len(vals_excellent) > 0 and len(vals_normal) > 0:
                mean_excellent = vals_excellent.mean()
                mean_normal = vals_normal.mean()
                
                # Test statistique
                try:
                    t_stat, p_value = stats.ttest_ind(vals_excellent, vals_normal)
                    significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                except:
                    significant = ""
                
                ratio = mean_excellent / (mean_normal + 1e-10)
                print(f"  {feature:25}: {mean_excellent:.4f} vs {mean_normal:.4f} (x{ratio:.2f}) {significant}")
    
    return corr_matrix, perf_corrs


def find_optimal_temporal_signatures(df, performance, temporal_features):
    """Identifie les signatures temporelles optimales par clustering."""
    
    print(f"\n🎨 RECHERCHE DE SIGNATURES TEMPORELLES OPTIMALES\n")
    
    # Préparer les données pour clustering
    features_for_clustering = temporal_features.dropna()
    perf_for_clustering = performance[features_for_clustering.index]
    
    if len(features_for_clustering) < 20:
        print("⚠️ Pas assez de données pour clustering")
        return None
    
    # Normalisation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)
    
    # K-means avec différents K
    best_k = 3
    silhouette_scores = []
    
    for k in range(2, min(8, len(features_for_clustering)//5)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            from sklearn.metrics import silhouette_score
            score = silhouette_score(features_scaled, labels)
            silhouette_scores.append((k, score))
            
            if score == max(s[1] for s in silhouette_scores):
                best_k = k
        except:
            continue
    
    # Clustering optimal
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    print(f"📊 Clustering en {best_k} signatures temporelles")
    
    # Analyser chaque cluster
    cluster_stats = []
    for cluster_id in range(best_k):
        mask = clusters == cluster_id
        cluster_perf = perf_for_clustering[mask]
        cluster_features = features_for_clustering[mask]
        
        stats_dict = {
            'cluster_id': int(cluster_id),
            'size': int(mask.sum()),
            'mean_performance': float(cluster_perf.mean()),
            'std_performance': float(cluster_perf.std()),
        }
        
        # Caractéristiques moyennes du cluster
        for col in features_for_clustering.columns:
            stats_dict[f'mean_{col}'] = float(cluster_features[col].mean())
        
        cluster_stats.append(stats_dict)
    
    # Trier par performance
    cluster_stats.sort(key=lambda x: x['mean_performance'], reverse=True)
    
    print(f"\n🏆 Signatures ordonnées par performance:")
    for i, stats in enumerate(cluster_stats):
        performance_score = stats['mean_performance']
        size = stats['size']
        
        print(f"\n  📋 Signature #{stats['cluster_id']} ({size} moments, perf={performance_score:.3f}):")
        
        # Top 3 caractéristiques
        feature_values = [(k.replace('mean_', ''), v) for k, v in stats.items() if k.startswith('mean_') and not k == 'mean_performance']
        
        for feature, value in feature_values[:4]:
            print(f"    {feature:20}: {value:.4f}")
    
    return cluster_stats


def plot_temporal_performance_analysis(df, performance, temporal_features, output_dir="./"):
    """Génère des visualisations de l'analyse."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse Corrélations Temporelles-Performance FPS', fontsize=16)
    
    # 1. Performance au cours du temps
    axes[0,0].plot(df['t'], performance, alpha=0.7, color='purple')
    axes[0,0].set_title('Performance au cours du temps')
    axes[0,0].set_xlabel('Temps (s)')
    axes[0,0].set_ylabel('Performance composite')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Tau évolution
    tau_cols = ['tau_S', 'tau_gamma', 'tau_A_mean', 'tau_f_mean']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (col, color) in enumerate(zip(tau_cols, colors)):
        if col in df.columns:
            axes[0,1].plot(df['t'], df[col], label=col, color=color, alpha=0.8)
    
    axes[0,1].set_title('Évolution des Tau')
    axes[0,1].set_xlabel('Temps (s)')
    axes[0,1].set_ylabel('Tau (s)')
    axes[0,1].legend()
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Corrélation performance vs cohérence temporelle
    if 'coherence' in temporal_features.columns:
        axes[0,2].scatter(temporal_features['coherence'], performance, alpha=0.6, s=20)
        axes[0,2].set_title('Performance vs Cohérence Temporelle')
        axes[0,2].set_xlabel('Cohérence temporelle')
        axes[0,2].set_ylabel('Performance')
        axes[0,2].grid(True, alpha=0.3)
    
    # 4. Ratios tau vs performance
    if 'ratio_structure_surface' in temporal_features.columns:
        axes[1,0].scatter(temporal_features['ratio_structure_surface'], performance, alpha=0.6, s=20, color='orange')
        axes[1,0].set_title('Performance vs Ratio Structure/Surface')
        axes[1,0].set_xlabel('tau_f_mean / tau_S')
        axes[1,0].set_ylabel('Performance')
        axes[1,0].set_xscale('log')
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Heatmap des corrélations
    all_features = temporal_features.copy()
    all_features['performance'] = performance
    corr_matrix = all_features.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[1,1], cbar_kws={'shrink': 0.8})
    axes[1,1].set_title('Matrice de Corrélations')
    
    # 6. Distribution performance avec tau stability
    if 'tau_stability' in temporal_features.columns:
        # Diviser en quartiles de stabilité
        stability_quartiles = pd.qcut(temporal_features['tau_stability'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = stability_quartiles == quartile
            axes[1,2].hist(performance[mask], alpha=0.7, label=f'Stabilité {quartile}', bins=20)
        
        axes[1,2].set_title('Performance par Quartile de Stabilité Tau')
        axes[1,2].set_xlabel('Performance')
        axes[1,2].set_ylabel('Fréquence')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'temporal_performance_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphiques sauvés: {output_file}")
    plt.show()


def main():
    """Fonction principale d'analyse."""
    
    print("🔍 ANALYSE CORRÉLATIONS TEMPORELLES-PERFORMANCE FPS\n")
    
    # Trouver le dernier CSV
    csvs = sorted(Path('logs').glob('run_*_FPS_seed*.csv'))
    
    if not csvs:
        print("❌ Aucun CSV FPS trouvé dans logs/")
        return
    
    csv_path = csvs[-1]
    print(f"📈 Analyse: {csv_path.name}")
    
    # Charger données
    df = load_fps_data(csv_path)
    if df is None or len(df) < 50:
        print("❌ Données insuffisantes")
        return
    
    print(f"✅ {len(df)} points de données chargés")
    
    # Calculer performance et features temporelles
    performance = compute_performance_score(df)
    temporal_features = compute_temporal_features(df)
    
    print(f"📊 {len(temporal_features.columns)} features temporelles calculées")
    print(f"🎯 Performance moyenne: {performance.mean():.3f} ± {performance.std():.3f}")
    
    # Analyses
    corr_matrix, perf_corrs = analyze_correlations(df, performance, temporal_features)
    cluster_stats = find_optimal_temporal_signatures(df, performance, temporal_features)
    
    # Visualisations
    plot_temporal_performance_analysis(df, performance, temporal_features)
    
    # Sauvegarde résultats
    results = {
        'csv_analyzed': str(csv_path),
        'performance_stats': {
            'mean': float(performance.mean()),
            'std': float(performance.std()),
            'min': float(performance.min()),
            'max': float(performance.max())
        },
        'correlations': {k: float(v) for k, v in perf_corrs.items()},
        'cluster_signatures': cluster_stats if cluster_stats else []
    }
    
    results_file = Path('temporal_performance_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Résultats sauvés: {results_file}")
    print(f"\n✨ Analyse terminée ! Découvertes fascinantes sur l'architecture temporelle de FPS !")


if __name__ == "__main__":
    main() 