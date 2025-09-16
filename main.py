"""
main.py - Orchestrateur principal du pipeline FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module orchestre TOUS les autres modules du pipeline FPS
sans jamais les court-circuiter, en respectant parfaitement
leurs interfaces et structures de données.

Fonctionnalités :
- Validation complète via validate_config.py
- Exécution via simulate.py (FPS/Kuramoto/Neutral)
- Exploration via explore.py
- Analyse batch via analyze.py
- Visualisation via visualize.py
- Gestion complète des erreurs et données
- Pipeline complet end-to-end

PRINCIPE : Ce module est un ORCHESTRATEUR pur.
Il ne fait QUE coordonner les autres modules.
Aucune logique métier n'est implémentée ici.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import argparse
import os
import json
import sys
import traceback
from datetime import datetime
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
from utils import deep_convert

# Imports conditionnels avec gestion d'erreurs
try:
    import numpy as np
except ImportError:
    print("❌ NumPy non installé. Installez avec : pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sans interface pour serveurs
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️  Matplotlib non installé. Visualisations désactivées.")
    print("   Pour activer : pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

# Import des modules FPS avec vérification
FPS_MODULES = {}
for module_name in ['validate_config', 'simulate', 'explore', 'analyze', 'visualize', 'utils']:
    try:
        FPS_MODULES[module_name] = __import__(module_name)
        print(f"✓ Module {module_name} importé")
    except ImportError as e:
        print(f"❌ Module {module_name} manquant : {e}")
        sys.exit(1)


def check_prerequisites() -> bool:
    """
    Vérifie que tous les prérequis sont satisfaits.
    
    Returns:
        bool: True si tout est OK
    """
    print("\n🔧 Vérification des prérequis...")
    
    # Vérifier les modules FPS
    required_modules = ['validate_config', 'simulate', 'explore', 'analyze', 'utils']
    if MATPLOTLIB_AVAILABLE:
        required_modules.append('visualize')
    
    missing = []
    for module in required_modules:
        if module not in FPS_MODULES:
            missing.append(module)
    
    if missing:
        print(f"❌ Modules manquants : {missing}")
        return False
    
    # Vérifier les dépendances Python
    try:
        import scipy
        print("✓ SciPy disponible")
    except ImportError:
        print("⚠️  SciPy recommandé mais non critique")
    
    try:
        import pandas
        print("✓ Pandas disponible")
    except ImportError:
        print("⚠️  Pandas recommandé pour l'analyse")
    
    print("✅ Prérequis validés")
    return True


def validate_configuration(config_path: str) -> Tuple[bool, Dict]:
    """
    Valide la configuration via validate_config.py.
    
    Args:
        config_path: chemin vers config.json
    
    Returns:
        Tuple[bool, Dict]: (valid, config_dict)
    """
    print(f"\n📋 Validation de la configuration : {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ Fichier de configuration non trouvé : {config_path}")
        return False, {}
    
    try:
        # Utiliser validate_config.py
        print(f"  → Appel validate_config({config_path})...")
        errors, warnings = FPS_MODULES['validate_config'].validate_config(config_path)
        
        print(f"  → Validation terminée. Errors: {len(errors) if errors else 0}, Warnings: {len(warnings) if warnings else 0}")
        
        # Debug : afficher le type et contenu des erreurs
        if errors:
            print(f"  → Type errors: {type(errors)}")
            print(f"  → Contenu errors: {errors}")
        
        # Afficher les erreurs
        if errors:
            print("❌ Erreurs de configuration :")
            for i, error in enumerate(errors):
                print(f"  - [{i}] {error}")
            return False, {}
        
        # Afficher les warnings
        if warnings:
            print("⚠️  Avertissements :")
            for i, warning in enumerate(warnings):
                print(f"  - [{i}] {warning}")
        
        # Charger la config si validation OK
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Configuration validée")
        return True, config
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation : {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def setup_environment(config: Dict) -> Dict[str, str]:
    """
    Configure l'environnement de travail via utils.py.
    
    Args:
        config: configuration validée
    
    Returns:
        Dict: chemins des dossiers créés
    """
    print("\n📁 Configuration de l'environnement...")
    
    try:
        # Utiliser utils.py pour créer la structure
        dirs = FPS_MODULES['utils'].setup_directories("fps_pipeline_output")
        
        # Logger la configuration
        run_id = FPS_MODULES['utils'].generate_run_id("pipeline")
        FPS_MODULES['utils'].log_config_and_meta(config, run_id, dirs['configs'])
        
        dirs['pipeline_run_id'] = run_id
        
        print(f"✅ Environnement configuré : {dirs['base']}")
        return dirs
        
    except Exception as e:
        print(f"❌ Erreur configuration environnement : {e}")
        raise


def execute_simulations(config_path: str, config: Dict, dirs: Dict) -> Dict[str, Any]:
    """
    Execute les simulations via simulate.py.
    
    Args:
        config_path: chemin config
        config: configuration
        dirs: dossiers de travail
    
    Returns:
        Dict: résultats des simulations
    """
    print("\n🔬 Exécution des simulations...")
    
    results = {}
    modes = ['FPS']
    
    # Ajouter les modes contrôles selon config
    if config.get('analysis', {}).get('compare_kuramoto', True):
        modes.append('Kuramoto')
    modes.append('neutral')
    
    for mode in modes:
        print(f"\n  → Simulation {mode}...")
        
        try:
            # Valider la config avant l'exécution
            if mode == 'FPS':
                # Vérifications spécifiques FPS
                N = config['system']['N']
                if N <= 0:
                    raise ValueError(f"N doit être > 0, reçu: {N}")
            
            # CORRECTION: Utiliser la même logique que les batch runs pour garantir cohérence
            # Créer un fichier config temporaire avec deep_convert pour cohérence
            temp_config_path = os.path.join(dirs['configs'], f'config_{mode.lower()}_main.json')
            with open(temp_config_path, 'w') as f:
                json.dump(deep_convert(config), f, indent=2)
            
            # Utiliser simulate.py avec le config temporaire 
            result = FPS_MODULES['simulate'].run_simulation(temp_config_path, mode)
            
            # Copier le fichier log dans le dossier du pipeline
            if 'logs' in result and isinstance(result['logs'], str):
                original_log = result['logs']
                if os.path.exists(original_log):
                    # Vérifier la taille du fichier
                    file_size = os.path.getsize(original_log)
                    if file_size > 0:
                        # Copier dans le dossier logs du pipeline
                        import shutil
                        pipeline_log_dir = os.path.join(dirs['base'], 'logs')
                        os.makedirs(pipeline_log_dir, exist_ok=True)
                        
                        new_log_path = os.path.join(pipeline_log_dir, os.path.basename(original_log))
                        shutil.copy2(original_log, new_log_path)
                        result['copied_log'] = new_log_path
                        print(f"    ✓ {mode} terminé : {result['run_id']} (log copié, {file_size} bytes)")
                    else:
                        print(f"    ⚠️ {mode} terminé : {result['run_id']} (log vide !)")
                        result['status'] = 'warning'
                        result['warning'] = 'Log file is empty'
                else:
                    print(f"    ⚠️ {mode} terminé : {result['run_id']} (log non trouvé: {original_log})")
                    result['status'] = 'warning'
                    result['warning'] = 'Log file not found'
            else:
                print(f"    ✓ {mode} terminé : {result['run_id']}")
            
            results[mode.lower()] = result
            
        except Exception as e:
            print(f"    ❌ Erreur {mode} : {e}")
            import traceback
            traceback.print_exc()
            results[mode.lower()] = {'status': 'error', 'error': str(e)}
    
    return deep_convert(results)


def run_exploration_analysis(results: Dict, config: Dict, dirs: Dict) -> Dict[str, Any]:
    """
    Lance l'exploration via explore.py.
    
    Args:
        results: résultats des simulations
        config: configuration
        dirs: dossiers
    
    Returns:
        Dict: résultats d'exploration
    """
    print("\n🔍 Exploration et détection d'émergences...")
    
    exploration_results = {}
    
    for mode, result in results.items():
        if result.get('status') == 'error':
            continue
        
        print(f"  → Exploration {mode}...")
        
        try:
            # Utiliser le log copié ou original
            log_path = result.get('copied_log', result.get('logs'))
            if not log_path:
                print(f"    ⚠️ Log non trouvé pour {mode}")
                continue
                
            # Vérifier si c'est un fichier qui existe
            if isinstance(log_path, str) and not os.path.exists(log_path):
                # Essayer dans le dossier logs
                if not log_path.startswith('logs/'):
                    log_path = os.path.join('logs', os.path.basename(log_path))
                if not os.path.exists(log_path):
                    print(f"    ⚠️ Fichier log non trouvé: {log_path}")
                    continue
            
            # Créer dossier de sortie pour ce mode
            output_dir = os.path.join(dirs['reports'], f"{mode}_exploration")
            
            # Utiliser explore.py
            exploration_result = FPS_MODULES['explore'].run_exploration(
                log_path, output_dir, config
            )
            
            exploration_results[mode] = exploration_result
            print(f"    ✓ {mode} exploré : {exploration_result.get('total_events', 0)} événements")
            
        except Exception as e:
            print(f"    ❌ Erreur exploration {mode} : {e}")
            exploration_results[mode] = {'status': 'error', 'error': str(e)}
    
    return deep_convert(exploration_results)


def run_batch_analysis(config_path: str, config: Dict, dirs: Dict) -> Optional[Dict]:
    """
    Execute l'analyse batch via analyze.py si configuré.
    
    Args:
        config_path: chemin config
        config: configuration
        dirs: dossiers
    
    Returns:
        Optional[Dict]: résultats batch ou None
    """
    batch_size = config.get('validation', {}).get('batch_size', 1)
    if batch_size <= 1:
        print("\n📊 Analyse batch désactivée (batch_size <= 1)")
        return None
    
    print(f"\n📊 Analyse batch : {batch_size} runs...")
    
    try:
        # Exécuter le batch dans le dossier logs du pipeline
        batch_log_dir = dirs['logs']
        batch_logs = []
        
        # Exécuter les simulations une par une
        for i in range(batch_size):
            print(f"  → Run {i+1}/{batch_size}...")
            
            # Modifier la seed pour chaque run
            config_copy = json.loads(json.dumps(config))  # Deep copy
            config_copy['system']['seed'] = config['system']['seed'] + i
            
            # Créer un fichier config temporaire
            temp_config_path = os.path.join(dirs['configs'], f'config_batch_{i}.json')
            with open(temp_config_path, 'w') as f:
                json.dump(deep_convert(config_copy), f, indent=2)
            
            # Exécuter la simulation
            result = FPS_MODULES['simulate'].run_simulation(temp_config_path, 'FPS')
            
            if result.get('logs'):
                # Copier le log dans le dossier du pipeline
                original_log = result['logs']
                if os.path.exists(original_log):
                    new_log_path = os.path.join(batch_log_dir, f'batch_run_{i}_{os.path.basename(original_log)}')
                    import shutil
                    shutil.copy2(original_log, new_log_path)
                    batch_logs.append(new_log_path)
                    print(f"    ✓ Log copié : {os.path.basename(new_log_path)}")
        
        if len(batch_logs) >= 2:
            print(f"  → Analyse de {len(batch_logs)} runs...")
            
            # Utiliser analyze.py avec les bons chemins
            analysis_result = FPS_MODULES['analyze'].analyze_criteria_and_refine(
                batch_logs, config
            )
            
            # Sauvegarder la config raffinée
            if analysis_result.get('refinements'):
                refined_path = os.path.join(dirs['configs'], 'config_refined.json')
                with open(refined_path, 'w') as f:
                    json.dump(deep_convert(analysis_result['updated_config']), f, indent=2)
                print(f"  ✓ Config raffinée : {refined_path}")
            
            return analysis_result
        else:
            print("  ⚠️ Pas assez de logs créés pour l'analyse")
            return None
            
    except Exception as e:
        print(f"  ❌ Erreur analyse batch : {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_visualizations(results: Dict, config: Dict, dirs: Dict) -> Dict[str, str]:
    """
    Génère les visualisations via visualize.py.
    
    Args:
        results: résultats des simulations
        config: configuration
        dirs: dossiers
    
    Returns:
        Dict: chemins des figures générées
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\n📈 Visualisations désactivées (matplotlib manquant)")
        return {}
    
    print("\n📈 Génération des visualisations...")
    
    figures_paths = {}
    
    try:
        # Vérifier qu'on a au moins FPS
        fps_result = results.get('fps')
        if not fps_result or fps_result.get('status') == 'error':
            print("  ⚠️ Pas de résultats FPS valides pour visualisation")
            return {}
        
        # Préparer les données temporelles pour visualize.py
        # Selon simulate.py, on a S_history, cpu_steps, etc.
        
        # 1. Évolution du signal S(t)
        if 'S_history' in fps_result and len(fps_result['S_history']) > 0:
            print("  → Signal evolution...")
            
            # Reconstruire le temps avec la bonne dimension
            dt = config.get('system', {}).get('dt', 0.05)
            n_points = len(fps_result['S_history'])
            t_array = np.linspace(0, (n_points - 1) * dt, n_points)
            
            fig1 = FPS_MODULES['visualize'].plot_signal_evolution(
                t_array, 
                np.array(fps_result['S_history']),
                "Évolution FPS - Signal S(t)"
            )
            path1 = os.path.join(dirs['figures'], 'signal_evolution_fps.png')
            fig1.savefig(path1, dpi=150, bbox_inches='tight')
            figures_paths['signal_evolution'] = path1
            plt.close(fig1)
        
        # 2. Dashboard métriques (si on a l'historique)
        if 'history' in fps_result:
            print("  → Metrics dashboard...")
            
            fig2 = FPS_MODULES['visualize'].plot_metrics_dashboard(fps_result['history'])
            path2 = os.path.join(dirs['figures'], 'metrics_dashboard.png')
            fig2.savefig(path2, dpi=150, bbox_inches='tight')
            figures_paths['metrics_dashboard'] = path2
            plt.close(fig2)
        
        # 3. Comparaison avec Kuramoto si disponible
        kuramoto_result = results.get('kuramoto')
        if kuramoto_result and kuramoto_result.get('status') != 'error':
            print("  → Comparaison FPS vs Kuramoto...")
            
            # Préparer les données dans le format attendu par visualize.py
            fps_data_viz = {
                'S(t)': fps_result.get('S_history', []),
                'C(t)': [h.get('C(t)', 1.0) for h in fps_result.get('history', [])] if fps_result.get('history') else [],
                'effort(t)': fps_result.get('effort_history', []),
                'cpu_step(t)': fps_result.get('cpu_steps', [])
            }
            
            kuramoto_data_viz = {
                'S(t)': kuramoto_result.get('S_history', []),
                'C(t)': [h.get('C', 1.0) for h in kuramoto_result.get('history', [])] if kuramoto_result.get('history') else [],
                'cpu_step(t)': kuramoto_result.get('cpu_steps', [])
            }
            
            fig3 = FPS_MODULES['visualize'].plot_fps_vs_kuramoto(fps_data_viz, kuramoto_data_viz)
            path3 = os.path.join(dirs['figures'], 'fps_vs_kuramoto.png')
            fig3.savefig(path3, dpi=150, bbox_inches='tight')
            figures_paths['comparison'] = path3
            plt.close(fig3)
        
        # 4. Grille empirique
        print("  → Grille empirique...")
        scores = calculate_empirical_scores(
            fps_result.get('metrics', {}), 
            config,
            fps_result.get('history', [])  # NOUVEAU : historique pour fenêtres adaptatives
        )
        fig4 = FPS_MODULES['visualize'].create_empirical_grid(scores)
        path4 = os.path.join(dirs['figures'], 'empirical_grid.png')
        fig4.savefig(path4, dpi=150, bbox_inches='tight')
        figures_paths['empirical_grid'] = path4
        plt.close(fig4)
        
        # 5. Matrice critères-termes
        print("  → Matrice critères-termes...")
        mapping = get_criteria_terms_mapping()
        fig5 = FPS_MODULES['visualize'].generate_correlation_matrix(mapping)
        path5 = os.path.join(dirs['figures'], 'criteria_terms_matrix.png')
        fig5.savefig(path5, dpi=150, bbox_inches='tight')
        figures_paths['correlation_matrix'] = path5
        plt.close(fig5)
        
        print(f"  ✓ {len(figures_paths)} visualisations générées")
        
    except Exception as e:
        print(f"  ❌ Erreur visualisations : {e}")
        traceback.print_exc()
    
    return figures_paths


def generate_final_report(results: Dict, exploration_results: Dict, 
                         analysis_result: Optional[Dict], config: Dict, 
                         dirs: Dict) -> str:
    """
    Génère le rapport final HTML via visualize.py.
    
    Args:
        results: résultats simulations
        exploration_results: résultats exploration
        analysis_result: résultats analyse batch
        config: configuration
        dirs: dossiers
    
    Returns:
        str: chemin du rapport
    """
    print("\n📄 Génération du rapport final...")
    
    try:
        # Agrégation des données pour visualize.py
        all_data = {
            'fps_result': results.get('fps', {}),
            'kuramoto_result': results.get('kuramoto', {}),
            'neutral_result': results.get('neutral', {}),
            'exploration_results': exploration_results,
            'analysis_result': analysis_result,
            'config': config,
            'metrics_summary': results.get('fps', {}).get('metrics', {}),
            'emergence_summary': count_emergence_events(exploration_results),
            'pipeline_metadata': {
                'timestamp': datetime.now().isoformat(),
                'run_id': dirs.get('pipeline_run_id', 'unknown'),
                'fps_version': '1.3'
            }
        }
        
        # Utiliser visualize.py
        report_path = os.path.join(dirs['reports'], 'rapport_complet_fps.html')
        FPS_MODULES['visualize'].export_html_report(all_data, report_path)
        
        print(f"  ✓ Rapport généré : {report_path}")
        return report_path
        
    except Exception as e:
        print(f"  ❌ Erreur génération rapport : {e}")
        # Créer un rapport minimal en cas d'erreur
        minimal_report = create_minimal_report(results, dirs)
        return minimal_report


def create_minimal_report(results: Dict, dirs: Dict) -> str:
    """
    Crée un rapport minimal en cas d'erreur.
    """
    report_path = os.path.join(dirs['reports'], 'rapport_minimal.txt')
    
    with open(report_path, 'w') as f:
        f.write("RAPPORT FPS - VERSION MINIMALE\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Généré le : {datetime.now()}\n\n")
        
        f.write("RÉSULTATS DES SIMULATIONS :\n")
        for mode, result in results.items():
            if result.get('status') == 'error':
                f.write(f"  {mode}: ERREUR - {result.get('error', 'inconnue')}\n")
            else:
                f.write(f"  {mode}: OK - Run ID {result.get('run_id', 'unknown')}\n")
        
        f.write(f"\nTous les fichiers dans : {dirs['base']}\n")
    
    return report_path


# ============== FONCTIONS HELPER ==============

def calculate_empirical_scores(metrics: Dict, config: Dict = None, history: List[Dict] = None) -> Dict[str, int]:
    """
    Calcule les scores 1-5 pour la grille empirique en utilisant les fenêtres adaptatives.
    
    Signé: Claude, Gepetto & Andréa Gadal 🌀
    
    Args:
        metrics: métriques calculées (pour fallback)
        config: configuration (contient adaptive_windows)
        history: historique complet pour fenêtres adaptatives
    
    Returns:
        dict avec scores 1-5 pour chaque critère
    """
    # Si on a l'historique, utiliser le nouveau système adaptatif
    if history and len(history) >= 20:
        try:
            # Utiliser calculate_all_scores avec fenêtres adaptatives
            import metrics as metrics_module
            adaptive_scores = metrics_module.calculate_all_scores(history, config)
            current_scores = adaptive_scores.get('current', {})
            
            if current_scores:
                # Mapper vers la grille empirique
                return {
                    'Stabilité': int(round(current_scores.get('stability', 3))),
                    'Régulation': int(round(current_scores.get('regulation', 3))),
                    'Fluidité': int(round(current_scores.get('fluidity', 3))),
                    'Résilience': int(round(current_scores.get('resilience', 3))),
                    'Innovation': int(round(current_scores.get('innovation', 3))),
                    'Coût CPU': int(round(current_scores.get('cpu_cost', 3))),
                    'Effort interne': int(round(current_scores.get('effort', 3)))
                }
        except Exception as e:
            print(f"  ⚠️ Erreur système adaptatif, fallback : {e}")
    
    # Sinon, fallback sur l'ancien système (garde la compatibilité)
    scores = {}
    
    # Stabilité (basée sur std_S et max_median_ratio)
    std_s = metrics.get('std_S', float('inf'))
    if std_s < 0.5:
        scores['Stabilité'] = 5
    elif std_s < 1.0:
        scores['Stabilité'] = 4
    elif std_s < 2.0:
        scores['Stabilité'] = 3
    else:
        scores['Stabilité'] = 2
    
    # Régulation (basée sur final_mean_abs_error)
    error = metrics.get('final_mean_abs_error', float('inf'))
    if error < 0.1:
        scores['Régulation'] = 5
    elif error < 0.5:
        scores['Régulation'] = 4
    elif error < 1.0:
        scores['Régulation'] = 3
    else:
        scores['Régulation'] = 2
    
    # Fluidité (basée sur la nouvelle métrique de fluidité)
    fluidity = metrics.get('final_fluidity', None)
    if fluidity is None:
        # Fallback : calculer depuis variance_d2S si fluidity n'est pas disponible
        var_d2s = metrics.get('final_variance_d2S', 175.0)
        x = var_d2s / 175.0  # Reference variance
        fluidity = 1 / (1 + np.exp(5.0 * (x - 1)))
    
    if fluidity >= 0.9:
        scores['Fluidité'] = 5
    elif fluidity >= 0.7:
        scores['Fluidité'] = 4
    elif fluidity >= 0.5:
        scores['Fluidité'] = 3
    elif fluidity >= 0.3:
        scores['Fluidité'] = 2
    else:
        scores['Fluidité'] = 1
    
    # Résilience - Utilise métrique adaptative unifiée
    adaptive_resilience_score = metrics.get('adaptive_resilience_score', None)
    
    if adaptive_resilience_score is not None:
        scores['Résilience'] = adaptive_resilience_score
    else:
        # Fallback : ancienne logique
        has_continuous_perturbation = False
        
        # Nouvelle structure avec input.perturbations
        input_cfg = config.get('system', {}).get('input', {}) if config else {}
        perturbations = input_cfg.get('perturbations', [])
        
        for pert in perturbations:
            if pert.get('type') in ['sinus', 'bruit', 'rampe']:
                has_continuous_perturbation = True
                break
        
        # Si pas trouvé dans la nouvelle structure, vérifier l'ancienne (pour compatibilité)
        if not has_continuous_perturbation and config:
            old_pert = config.get('system', {}).get('perturbation', {})
            if old_pert.get('type') in ['sinus', 'bruit', 'rampe']:
                has_continuous_perturbation = True
        
        if has_continuous_perturbation:
            cont_resilience = metrics.get('continuous_resilience_mean', metrics.get('continuous_resilience', 0))
            if cont_resilience >= 0.90:
                scores['Résilience'] = 5
            elif cont_resilience >= 0.75:
                scores['Résilience'] = 4
            elif cont_resilience >= 0.60:
                scores['Résilience'] = 3
            elif cont_resilience >= 0.40:
                scores['Résilience'] = 2
            else:
                scores['Résilience'] = 1
        else:
            t_retour = metrics.get('resilience_t_retour', float('inf'))
            if t_retour < 1.0:
                scores['Résilience'] = 5
            elif t_retour < 2.0:
                scores['Résilience'] = 4
            elif t_retour < 5.0:
                scores['Résilience'] = 3
            elif t_retour < 10.0:
                scores['Résilience'] = 2
            else:
                scores['Résilience'] = 1
    
    # Innovation (basée sur entropy_S moyen pour cohérence avec système adaptatif)
    # CORRECTION: utiliser la moyenne plutôt que la valeur finale pour éviter les artefacts
    entropy = metrics.get('entropy_S', metrics.get('final_entropy_S', 0))  # Priorité à la moyenne
    if entropy > 0.8:
        scores['Innovation'] = 5
    elif entropy > 0.6:
        scores['Innovation'] = 4
    elif entropy > 0.4:
        scores['Innovation'] = 3
    else:
        scores['Innovation'] = 2
    
    # Coût CPU (basé sur mean_cpu_step)
    cpu = metrics.get('mean_cpu_step', float('inf'))
    if cpu < 0.001:
        scores['Coût CPU'] = 5
    elif cpu < 0.01:
        scores['Coût CPU'] = 4
    elif cpu < 0.1:
        scores['Coût CPU'] = 3
    else:
        scores['Coût CPU'] = 2
    
    # Effort interne (basé sur mean_effort)
    effort = metrics.get('mean_effort', float('inf'))
    if effort < 0.5:
        scores['Effort interne'] = 5
    elif effort < 1.0:
        scores['Effort interne'] = 4
    elif effort < 2.0:
        scores['Effort interne'] = 3
    else:
        scores['Effort interne'] = 2
    
    return scores


def get_criteria_terms_mapping() -> Dict[str, List[str]]:
    """
    Retourne le mapping critères-termes FPS pour la matrice de corrélation.
    
    Cette fonction définit la correspondance entre les critères empiriques
    et les termes mathématiques du système FPS.
    """
    return {
        'Stabilité': ['S(t)', 'C(t)', 'φₙ(t)', 'L(t)', 'max_median_ratio'],
        'Régulation': ['Fₙ(t)', 'G(x)', 'γ(t)', 'Aₙ(t)', 'mean_abs_error'],
        'Fluidité': ['γₙ(t)', 'σ(x)', 'envₙ(x,t)', 'μₙ(t)', 'fluidity'],
        'Résilience': ['Aₙ(t)', 'G(x,t)', 'effort(t)', 'adaptive_resilience'],
        'Innovation': ['A_spiral(t)', 'Eₙ(t)', 'r(t)', 'entropy_S'],
        'Coût CPU': ['cpu_step(t)', 'N', 'T'],
        'Effort interne': ['effort(t)', 'd_effort/dt', 'mean_high_effort']
    }


def count_emergence_events(exploration_results: Dict) -> Dict[str, int]:
    """
    Compte les événements d'émergence détectés par explore.py.
    
    Args:
        exploration_results: résultats d'exploration
    
    Returns:
        Dict: comptage par type d'événement
    """
    summary = defaultdict(int)
    
    for mode, result in exploration_results.items():
        if result.get('status') == 'error':
            continue
        
        events = result.get('events', [])
        for event in events:
            event_type = event.get('event_type', 'unknown')
            summary[event_type] += 1
    
    return dict(summary)


# ============== PIPELINE PRINCIPAL ==============

def run_complete_pipeline(config_path: str, parallel: bool = False) -> bool:
    """
    Execute le pipeline complet FPS avec orchestration parfaite.
    
    Args:
        config_path: chemin vers config.json
        parallel: utilisation du parallélisme pour batch
    
    Returns:
        bool: True si succès complet
    """
    print("\n🌀 PIPELINE FPS COMPLET - ORCHESTRATION V1.3 🌀")
    print("=" * 60)
    
    try:
        # 1. Vérifications préalables
        if not check_prerequisites():
            return False
        
        # 2. Validation configuration via validate_config.py
        valid, config = validate_configuration(config_path)
        if not valid:
            return False
        
        # 3. Setup environnement via utils.py
        dirs = setup_environment(config)
        
        # 4. Exécution simulations via simulate.py
        results = execute_simulations(config_path, config, dirs)

        # Rapport de comparaison
        if all(mode in results for mode in ['fps', 'kuramoto', 'neutral']):
            # Importer le module de comparaison
            import compare_modes
    
            # Générer le rapport de comparaison
            comparison_path = os.path.join(dirs['reports'], 'comparison_fps_vs_controls.json')
            comparison_report = compare_modes.export_comparison_report(
                results['fps'],
                results['kuramoto'], 
                results['neutral'],
                comparison_path
            )
    
            print(f"\n📊 Rapport de comparaison généré :")
            print(f"  JSON : {comparison_path}")
            print(f"  TXT  : {comparison_path.replace('.json', '.txt')}")
            print(f"  Verdict : {comparison_report['summary']['overall_verdict']}")
        
        # Vérifier qu'on a au moins un résultat valide
        valid_results = {k: v for k, v in results.items() if v.get('status') != 'error'}
        if not valid_results:
            print("❌ Aucune simulation n'a réussi")
            return False
        
        # 5. Exploration via explore.py
        exploration_results = run_exploration_analysis(results, config, dirs)
        
        # 6. Analyse batch via analyze.py (optionnel)
        analysis_result = run_batch_analysis(config_path, config, dirs)
        
        # 7. Visualisations via visualize.py
        figures_paths = generate_visualizations(results, config, dirs)
        
        # 8. Rapport final via visualize.py
        report_path = generate_final_report(
            results, exploration_results, analysis_result, config, dirs
        )
        
        # 9. Résumé final
        print("\n" + "=" * 60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
        print("=" * 60)
        print(f"📂 Dossier principal : {dirs['base']}")
        print(f"📄 Rapport complet : {report_path}")
        print(f"📊 {len(valid_results)} simulations réussies")
        print(f"📈 {len(figures_paths)} visualisations générées")
        
        if analysis_result and analysis_result.get('refinements'):
            print(f"🔧 {len(analysis_result['refinements'])} raffinements appliqués")
        
        # Afficher les métriques finales FPS pour cohérence avec simulate.py
        fps_result = valid_results.get('fps')
        if fps_result and 'metrics' in fps_result:
            print(f"📊 Métriques finales FPS : {fps_result['metrics']}")
        
        print("\n🌀 La danse FPS s'achève en harmonie ! 🌀")
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE PIPELINE : {e}")
        traceback.print_exc()
        return False


def main():
    """Point d'entrée principal avec interface CLI complète."""
    
    parser = argparse.ArgumentParser(
        description='FPS - Fractal Pulsating Spiral v1.3 - Pipeline Complet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Pipeline complet (recommandé)
  python main.py complete --config config.json

  # Run simple avec mode spécifique
  python main.py run --config config.json --mode FPS

  # Batch de simulations en parallèle
  python main.py batch --config config.json --parallel

  # Analyse d'un batch existant
  python main.py analyze --config config.json

  # Comparaison FPS vs Kuramoto seulement
  python main.py compare --config config.json

  # Validation seule de la configuration
  python main.py validate --config config.json

        """
    )
    
    parser.add_argument('action', 
                        choices=['complete', 'run', 'batch', 'analyze', 'compare', 'validate'],
                        help='Action à effectuer')
    parser.add_argument('--config', default='config.json',
                        help='Fichier de configuration (défaut: config.json)')
    parser.add_argument('--mode', default='FPS', 
                        choices=['FPS', 'Kuramoto', 'neutral'],
                        help='Mode de simulation pour run simple')
    parser.add_argument('--parallel', action='store_true',
                        help='Exécution parallèle pour batch')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Affichage détaillé')
    
    args = parser.parse_args()
    
    # Configuration du niveau de détail
    if args.verbose:
        print("Mode verbose activé")
    
    # Exécution selon l'action demandée
    try:
        if args.action == 'complete':
            # Pipeline complet - RECOMMANDÉ
            success = run_complete_pipeline(args.config, args.parallel)
            sys.exit(0 if success else 1)
            
        elif args.action == 'validate':
            # Validation seule
            valid, config = validate_configuration(args.config)
            if valid:
                print("✅ Configuration valide")
                sys.exit(0)
            else:
                print("❌ Configuration invalide")
                sys.exit(1)
                
        elif args.action == 'run':
            # Run simple via simulate.py
            if not check_prerequisites():
                sys.exit(1)
            
            valid, config = validate_configuration(args.config)
            if not valid:
                sys.exit(1)
            
            print(f"\n🔬 Simulation {args.mode}...")
            result = FPS_MODULES['simulate'].run_simulation(args.config, args.mode)
            print(f"✅ Terminé : {result['run_id']}")
            
        elif args.action == 'batch':
            # Batch via utils.py
            if not check_prerequisites():
                sys.exit(1)
            
            valid, config = validate_configuration(args.config)
            if not valid:
                sys.exit(1)
            
            batch_size = config.get('validation', {}).get('batch_size', 5)
            configs = [args.config] * batch_size
            
            print(f"\n🔄 Batch de {batch_size} simulations...")
            results = FPS_MODULES['utils'].batch_runner(configs, args.parallel)
            
            success_count = sum(1 for r in results if r.get('status') == 'success')
            print(f"✅ Batch terminé : {success_count}/{len(results)} succès")
            
        elif args.action == 'analyze':
            # Analyse via analyze.py
            if not check_prerequisites():
                sys.exit(1)
            
            valid, config = validate_configuration(args.config)
            if not valid:
                sys.exit(1)
            
            # Chercher les logs récents
            logs = sorted(glob.glob('logs/run_*.csv'))[-5:]
            if len(logs) < 2:
                print("❌ Pas assez de logs pour l'analyse (minimum 2)")
                sys.exit(1)
            
            print(f"\n📊 Analyse de {len(logs)} runs...")
            result = FPS_MODULES['analyze'].analyze_criteria_and_refine(logs, config)
            
            if result.get('refinements'):
                print(f"✅ {len(result['refinements'])} raffinements appliqués")
            else:
                print("✅ Aucun raffinement nécessaire")
                
        elif args.action == 'compare':
            # Comparaison simple FPS vs Kuramoto
            if not check_prerequisites():
                sys.exit(1)
            
            valid, config = validate_configuration(args.config)
            if not valid:
                sys.exit(1)
            
            print("\n⚖️  Comparaison FPS vs Kuramoto...")
            
            fps_result = FPS_MODULES['simulate'].run_simulation(args.config, 'FPS')
            kura_result = FPS_MODULES['simulate'].run_simulation(args.config, 'Kuramoto')
            
            if MATPLOTLIB_AVAILABLE:
                fig = FPS_MODULES['visualize'].plot_fps_vs_kuramoto(fps_result, kura_result)
                output_path = f'comparison_{datetime.now():%Y%m%d_%H%M%S}.png'
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✅ Comparaison sauvegardée : {output_path}")
                plt.close(fig)
            else:
                print("✅ Comparaison terminée (matplotlib manquant pour visualisation)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()