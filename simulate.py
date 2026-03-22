"""
simulate.py – FPS Pipeline Simulation Core
Version exhaustive FPS
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Ce module orchestre toute la dynamique FPS/Kuramoto/Neutral, et doit toujours rester :
- **modulaire** : chaque fonction/étape isolable, falsifiable individuellement
- **extensible** : chaque critère, métrique, ou pipeline peut être augmenté/remplacé
- **falsifiable** : tout doit être logué, traçable, chaque hypothèse peut être testée, toute évolution loguée
- **strict** : tout ce qui est attendu ici, même si le module n'est pas encore codé, doit être détaillé dans ce fichier en code ou en commentaire.
---------------------------------------------------------------

**Pipeline FPS** – *Résumé*
Boucle temporelle orchestrant :
- Chargement et validation complète du config.json (validate_config.py)
- Initialisation strates, seeds, loggers (init.py)
- Gestion du mode (FPS, Kuramoto, Neutral)
- Boucle de simulation temps-réel (avec dynamique FPS complète OU alternatives)
- Calculs à chaque pas :
    - Input contextuel/perturbations, dynamique FPS (dynamics.py)
    - Régulation (regulation.py), feedback adaptatif G(x)
    - Mise à jour des états, calcul S(t), E(t), O(t), metrics (metrics.py)
    - Log metrics, backup d'état, gestion erreurs (utils.py)
- Gestion post-run :
    - Analyse & raffinements automatiques (analyze.py)
    - Exploration émergences, fractals, anomalies (explore.py)
    - Visualisation complète, dashboard, grille empirique (visualize.py)
    - Export de tous les logs, configs, seeds, changelog.txt
- Exécution en mode batch/auto si besoin (via batch_runner ou module export-batch)
"""

# --------- IMPORTS (TOUS MODULES DU PIPELINE FPS) -----------
import os, sys, time, json, traceback, numpy as np
import scipy
import csv
# Imports stricts
import init
import dynamics
import regulation
import metrics
import analyze
import explore
import visualize
import validate_config
import kuramoto
import perturbations
import utils
from utils import deep_convert, extract_best_pair_from_journal

def safe_float_conversion(value, default=0.0):
    """Convertit une valeur en float sûr."""
    try:
        if isinstance(value, str):
            return default
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    except:
        return default


# --------- PIPELINE PRINCIPALE : RUN_SIMULATION() -----------

def run_simulation(config_path, mode="FPS", strict=False):
    """
    Orchestration complète d'un run FPS/Kuramoto/Neutral.
    - Validation complète config.json (présence, structure, types, seuils, dépendances croisées…)
    - Initialisation strates, seeds, loggers
    - Branche pipeline selon le mode
    - À chaque step : dynamique complète (input, dynamique FPS, feedback, metrics, log, backup)
    - Gestion post-run (analyse, exploration, visualisation, logs, exports)
    """
    # ---- 0. Chargement et VALIDATION TOTALE du config ----
    config = init.load_config(config_path) if hasattr(init, "load_config") else json.load(open(config_path))
    errors, warnings = validate_config.validate_config(config_path) if hasattr(validate_config, "validate_config") else ([], [])
    if errors:  # Si il y a des erreurs (liste non vide)
        print("Config validation FAILED:")
        for e in errors: print("  -", e)
        sys.exit(1)
    if warnings:
        print("Config validation avec avertissements:")
        for w in warnings: print("  -", w)
    print("Config validation: OK")

    # ---- 1. SEED : reproductibilité, log dans seeds.txt ----
    SEED = config['system']['seed']
    # CORRECTION: Assurer cohérence seeds entre run principal et batch
    print(f"🌱 Initialisation seed: {SEED} pour mode {mode}")
    np.random.seed(SEED)
    import random; random.seed(SEED)
    
    # Log de la seed pour traçabilité
    if hasattr(utils, "log_seed"): 
        utils.log_seed(SEED)
    else: 
        with open("seeds.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | SEED = {SEED} | Mode = {mode}\n")
    
    # ---- 2. Initialisation : strates, état, logs, dirs ----
    state = init.init_strates(config)
    loggers = init.setup_logging(config, mode_suffix=mode.upper())
    
    if hasattr(utils, "log_config_and_meta"): 
        utils.log_config_and_meta(config, loggers['run_id'])
    
    # ---- 3. BRANCH par mode (FPS/Kuramoto/Neutral) ----
    if mode.lower() == "kuramoto":
        result = run_kuramoto_simulation(config, loggers)
    elif mode.lower() == "neutral":
        result = run_neutral_simulation(config, loggers)
    elif mode.lower() == "fps":
        result = run_fps_simulation(config, state, loggers, strict)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(2)
    
    # ---- 4. Post-run : Analyse, exploration, visualisation, export ----
    if hasattr(analyze, "analyze_criteria_and_refine"):
        analyze.analyze_criteria_and_refine([result['logs']], config)
    
    if hasattr(explore, "run_exploration"):
        explore.run_exploration(result['logs'], loggers['output_dir'], config)
    
    if hasattr(visualize, "plot_metrics_dashboard"):
        visualize.plot_metrics_dashboard(result['metrics'])
    
    # ---- 5. Exports finaux : rapport, changelog, backup ----
    if hasattr(utils, "log_end_of_run"): 
        utils.log_end_of_run(result['run_id'])
    
    # NOTE: Le fichier CSV sera fermé APRÈS la simulation dans run_fps_simulation
    # pour éviter la fermeture prématurée qui empêche l'enregistrement des données
    
    return result

# --- MODE FPS (Pipeline complet) ------------------------------------------
# --- MODE FPS (Pipeline complet) ------------------------------------------
def run_fps_simulation(config, state, loggers, strict=False):
    """
    Boucle principale FPS, version exhaustive :
    - À chaque pas : input contextuel, dynamique FPS (toutes formules), feedback/régulation, logs, backup
    - Gestion du mode statique/dynamique pour tous paramètres (voir config)
    - Log exhaustif des métriques (voir feuille de route/tableau structuré)
    - Gestion stricte erreurs, try/except, logs des exceptions
    - Sécurité backup état toutes les 100 steps
    - À la fin : export de tous les logs, résumé metrics, passage à l'analyse et à l'exploration
    """
    # -- PARAMÈTRES GLOBAUX
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    N = config['system']['N']
    run_id = loggers['run_id']
    t_array = np.arange(0, T, dt)
    backup_interval = 100
    # Chercher t_choc depuis la config de perturbations
    perturbations_cfg = config.get('system', {}).get('input', {}).get('perturbations', [])
    choc_pert = [p for p in perturbations_cfg if p.get('type') == 'choc']
    t_choc = choc_pert[0].get('t0', T/2) if choc_pert else None
    # Initialiser les signaux de feedback à zéro
    F_n_t_An = np.zeros(N)
    F_n_t_fn = np.zeros(N)

    # Historiques avec limite de mémoire
    MAX_HISTORY_SIZE = config.get('system', {}).get('max_history_size', 10000)
    history, cpu_steps, effort_history, S_history = [], [], [], []
    # Historiques supplémentaires pour métriques avancées
    C_history = []
    An_history = []  # Pour A_mean(t) et export individuel
    fn_history = []  # Pour f_mean(t) et export individuel
    En_history = []  # Pour mean_abs_error
    On_history = []  # Pour mean_abs_error
    
    # NOUVEAU S1: Historique des alignements En ≈ On
    history_align = []  # Stocke les timestamps où |E-O| < epsilon_E
    epsilon_E = config.get('regulation', {}).get('epsilon_E', 0.01)  # Seuil d'alignement
    
    # NOUVEAU : État adaptatif pour gamma et G
    adaptive_state = {
        'gamma_journal': None,
        'regulation_state': {}
    }

    # INITIALISER all_metrics ET t EN DEHORS DE LA BOUCLE
    all_metrics = {}
    t = 0
    
    # Fonction de rotation des historiques
    def rotate_history(hist_list, max_size):
        """Garde seulement les max_size derniers éléments."""
        if len(hist_list) > max_size:
            return hist_list[-max_size:]
        return hist_list
    
    # Préparer les fichiers individuels si N > 10
    individual_csv_writers = {}
    if N > 10 and config.get('analysis', {}).get('save_indiv_files', True):
        for n in range(N):
            # A_n_{id}.csv
            an_filename = os.path.join(loggers['output_dir'], f"A_n_{n}_{run_id}.csv")
            an_file = open(an_filename, 'w', newline='')
            an_writer = csv.writer(an_file)
            an_writer.writerow(['t', f'A_{n}(t)'])
            individual_csv_writers[f'A_{n}'] = {'file': an_file, 'writer': an_writer}
            
            # f_n_{id}.csv
            fn_filename = os.path.join(loggers['output_dir'], f"f_n_{n}_{run_id}.csv")
            fn_file = open(fn_filename, 'w', newline='')
            fn_writer = csv.writer(fn_file)
            fn_writer.writerow(['t', f'f_{n}(t)'])
            individual_csv_writers[f'f_{n}'] = {'file': fn_file, 'writer': fn_writer}
    
    print(f"DIAG strates: betas={[s['beta'] for s in state]}")
    print(f"DIAG strates: f0s={[s['f0'] for s in state]}")
    print(f"DIAG strates: A0s={[s['A0'] for s in state]}")

    # -- BOUCLE PRINCIPALE --
    try:
        for step, t in enumerate(t_array):
            step_start = time.perf_counter()

            if step == 1:
                print(f"DIAG step1: F_n_t_An={F_n_t_An}")
                print(f"DIAG step1: F_n_t_fn={F_n_t_fn}")
                print(f"DIAG step1: In_t_before={In_t if 'In_t' in dir() else 'N/A'}")
            
            # ----------- 1. INPUTS ET PERTURBATIONS -----------
            # Nouvelle architecture In(t)
            input_config = config.get('system', {}).get('input', {})
            
            # Calculer In(t) avec la nouvelle architecture
            try:
                input_cfg_with_T = input_config.copy()
                input_cfg_with_T['T'] = T
                In_t = perturbations.compute_In(t, input_cfg_with_T, state, history, dt)
                if np.isscalar(In_t):
                    In_t = np.full(N, In_t)
                
                # Logger les valeurs significatives
                if step % 100 == 0 and np.mean(In_t) > 0.01:
                    print(f"  📊 Input contextuel à t={t:.1f}: In_mean={np.mean(In_t):.3f}")
                    
            except Exception as e:
                print(f"⚠️ Erreur compute_In à t={t}: {e}")
                print(f"   input_config: {input_config}")
                print(f"   Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                # Fallback : utiliser un offset minimal
                In_t = np.full(N, 0.1)
            
            # ----------- 2. CALCULS DYNAMIQUE FPS --------------
            # Réinitialiser le chronomètre ici pour mesurer UNIQUEMENT la dynamique FPS
            # (perturbations déjà appliquées ; métriques et logging seront chronométrés après)
            core_start = time.perf_counter()
            
            # a) Amplitude, fréquence, phase, latence par strate (avec statique/dynamique, config.json)
            try:
                # IMPORTANT: Ajouter l'historique à la config pour compute_An avec enveloppe dynamique
                config_for_An = config.copy()
                config_for_An['history'] = history
                An_t = dynamics.compute_An(t, state, In_t, F_n_t_An, config_for_An)
                An_history.append(An_t.copy())                
                # Passer l'historique dans la config pour compute_fn
                config_with_history = config.copy()
                config_with_history['history'] = history
                fn_t = dynamics.compute_fn(t, state, An_t, F_n_t_fn, config_with_history)
                fn_history.append(fn_t.copy())
                # Calculer et stocker delta_fn pour l'historique
                delta_fn_t = np.zeros(N)
                for n in range(N):
                    S_i = dynamics.compute_S_i(t, n, history, state)
                    delta_fn_t[n] = dynamics.compute_delta_fn(t, state[n]['alpha'], S_i)
                phi_n_t = dynamics.compute_phi_n(t, state, config)
                
                # Calculer gamma global pour le logging
                latence_config = config.get('latence', {})
                gamma_mode = latence_config.get('gamma_mode', 'static')
                gamma_dynamic = latence_config.get('gamma_dynamic', {})
                k = gamma_dynamic.get('k', None)
                t0 = gamma_dynamic.get('t0', None)
                
                # CORRECTION : Calculer gamma adaptatif AVANT gamma_n_t
                if gamma_mode == 'adaptive_aware':
                    # Calculer gamma avec conscience de G
                    gamma_t, gamma_regime, gamma_journal = dynamics.compute_gamma_adaptive_aware(
                        t, state, history, config, adaptive_state['gamma_journal']
                    )
                    adaptive_state['gamma_journal'] = gamma_journal
                else:
                    # Modes classiques
                    gamma_t = dynamics.compute_gamma(t, gamma_mode, config.get('system', {}).get('T', 100), k, t0)
                    gamma_regime = 'classic'
                
                # Latence expressive gamma par strate (utilise gamma_t calculé)
                # NOUVEAU : Passer l'historique pour modulation locale basée sur t-1
                gamma_n_t = dynamics.compute_gamma_n(t, state, config, gamma_t,
                                                   An_array=An_t,  # An_t est déjà calculé
                                                   fn_array=fn_t,  # fn_t est déjà calculé  
                                                   history=history)
                    
            except Exception as e:
                print(f"❌ ERREUR CRITIQUE calculs dynamiques à t={t}: {e}")
                import traceback
                traceback.print_exc()
                # DÉBOGAGE : Détailler l'exception pour identifier la source
                print(f"🔍 DÉTAILS EXCEPTION:")
                print(f"  - Type: {type(e).__name__}")
                print(f"  - State: {len(state) if state else 'None'}")
                print(f"  - In_t: {In_t.shape if hasattr(In_t, 'shape') else type(In_t)}")
                print(f"  - Config keys: {list(config.keys())}")
                print(f"  - Enveloppe mode: {config.get('enveloppe', {}).get('env_mode', 'None')}")
                
                # En cas d'erreur, utiliser des valeurs de fallback mais alerter clairement
                print(f"🚨 UTILISATION VALEURS FALLBACK - ceci explique les problèmes de reproductibilité !")
                An_t = np.ones(N)
                fn_t = np.ones(N)
                phi_n_t = np.zeros(N)
                gamma_n_t = np.ones(N)
            
            # b) Calcul de phi adaptatif (alignement notebook)
            try:
                phi_mode = config.get('regulation', {}).get('phi_mode', 'fixed')
                
                if phi_mode == 'adaptive':
                    # Récupérer l'effort actuel depuis history
                    if len(history) > 0 and 'effort(t)' in history[-1]:
                        effort_current = history[-1].get('effort(t)', 0.0)
                    else:
                        effort_current = 0.0
                    
                    # Prendre les 20 derniers éléments pour phi_adaptive
                    effort_for_phi = effort_history[-20:] if len(effort_history) > 0 else [0.0]
                    
                    # Calculer phi adaptatif
                    phi_reg = dynamics.compute_phi_adaptive(effort_current, effort_for_phi, config)
                else:
                    # Mode fixe
                    phi_reg = config.get('regulation', {}).get('phi_fixed_value', 1.618)
            except Exception as e:
                print(f"⚠️ Erreur calcul phi adaptatif à t={t}: {e}")
                phi_reg = config.get('regulation', {}).get('phi_fixed_value', 1.618)
            
            # c) Sorties observée/attendue
            try:
                On_t = dynamics.compute_On(t, state, An_t, fn_t, phi_n_t, config) if hasattr(dynamics, 'compute_On') else An_t
                En_t = dynamics.compute_En(t, state, history, config, phi_reg, history_align, effort_history) if hasattr(dynamics, 'compute_En') else An_t
            except Exception as e:
                print(f"⚠️ Erreur compute On/En à t={t}: {e}")
                On_t = An_t
                En_t = An_t
            
            # d) Régulation/adaptation feedback
            try:                
                # NOUVEAU : Logger détaillé pour diagnostic
                debug_log_data = {
                    't': t,
                    'In_t': In_t.copy() if isinstance(In_t, np.ndarray) else In_t,
                    'An_t': An_t.copy() if isinstance(An_t, np.ndarray) else An_t,
                    'En_t': En_t.copy() if isinstance(En_t, np.ndarray) else En_t,
                    'On_t': On_t.copy() if isinstance(On_t, np.ndarray) else On_t,
                    'phi_n_t': phi_n_t.copy() if isinstance(phi_n_t, np.ndarray) else phi_n_t,  # NOUVEAU: phases
                    'fn_t': fn_t.copy() if isinstance(fn_t, np.ndarray) else fn_t,  # NOUVEAU: fréquences
                    'gamma_n_t': gamma_n_t.copy() if isinstance(gamma_n_t, np.ndarray) else gamma_n_t,  # NOUVEAU: gamma par strate
                    'erreur_n': (En_t - On_t).copy() if isinstance(En_t, np.ndarray) else (En_t - On_t),
                    'G_values': [],
                    'G_archs': []
                }
                
                # NOUVEAU : Calculer et logger r(t) si mode spirale activé
                dynamic_params = config.get('dynamic_parameters', {})
                if dynamic_params.get('dynamic_phi', False):
                    spiral_config = config.get('spiral', {})
                    phi = spiral_config.get('phi', 1.618)
                    epsilon = spiral_config.get('epsilon', 0.05)
                    omega = spiral_config.get('omega', 0.1)
                    theta = spiral_config.get('theta', 0.0)
                    r_t = dynamics.compute_r(t, phi, epsilon, omega, theta)
                    debug_log_data['r_t'] = r_t
                else:
                    debug_log_data['r_t'] = 1.618  # Valeur par défaut
                
                # NOUVEAU : Vérifier si on utilise G adaptatif
                regulation_config = config.get('regulation', {})
                G_arch_mode = regulation_config.get('G_arch', 'tanh')
                G_archs_used = []
                
                # Initialiser G_params par défaut pour éviter l'erreur dans le bloc except
                G_params = {
                    'lambda': regulation_config.get('lambda', 1.0),
                    'alpha': regulation_config.get('alpha', 1.0),
                    'beta': regulation_config.get('beta', 2.0)
                }
                
                for n in range(N):
                    beta_n = state[n]['beta']
                    error_n = En_t[n] - On_t[n]                    
                    # NOUVEAU : Calculer G selon le mode configuré
                    if G_arch_mode == 'adaptive_aware':
                        # G adaptatif
                        G_value, G_arch_used, G_params = dynamics.compute_G_adaptive_aware(
                            error_n, t, gamma_t, adaptive_state['regulation_state'], history, config, True, False
                        )
                        G_archs_used.append(G_arch_used)
                    else:
                        # G statique selon config
                        if config.get('regulation', {}).get('feedback_mode', 'simple') == 'simple':
                            G_value = error_n  # Pas de régulation
                            G_arch_used = 'identity'
                        else:
                            # Calculer G selon l'archétype configuré
                            G_arch = config.get('regulation', {}).get('G_arch', 'tanh')
                            G_params = {
                                'lambda': config.get('regulation', {}).get('lambda', 1.0),
                                'alpha': config.get('regulation', {}).get('alpha', 1.0),
                                'beta': config.get('regulation', {}).get('beta', 2.0)
                            }
                            G_value = regulation.compute_G(error_n, G_arch, G_params)
                            G_arch_used = G_arch
                        G_archs_used.append(G_arch_used)
                    
                    # Calculer le feedback avec G_value déjà calculé
                    F_n_t_fn[n] = state[n]['beta'] * gamma_t
                    F_n_t_An[n] = state[n]['beta'] * G_value
                    # Log G value per strate for analysis overlay
                    debug_log_data['G_values'].append(G_value)
                    debug_log_data['G_archs'].append(G_arch_used)
                
                # Archétype dominant pour le logging
                if G_archs_used:
                    G_arch_dominant = max(set(G_archs_used), key=G_archs_used.count)
                else:
                    G_arch_dominant = G_arch_mode

                if step == 0:
                    print(f"DIAG step0 post-reg: F_n_t_An={F_n_t_An}")
                    print(f"DIAG step0 post-reg: G_values={[debug_log_data['G_values']]}")
                
            except Exception as e:
                print(f"⚠️ Erreur régulation à t={t}: {e}")
            
            # d) Update état complet du système
            try:
                state = dynamics.update_state(state, An_t, fn_t, phi_n_t, gamma_n_t, F_n_t_fn, F_n_t_An) if hasattr(dynamics, 'update_state') else state
                # Mesurer le coût CPU « dynamique pur » (du core_start à la fin de update_state)
                cpu_step = metrics.compute_cpu_step(core_start, time.perf_counter(), N) if hasattr(metrics, 'compute_cpu_step') else 0.0
            except Exception as e:
                print(f"⚠️ Erreur update state à t={t}: {e}")
            
            # ----------- 3. SIGNAUX GLOBAUX ET MÉTRIQUES -------------
            try:
                # Préparer config enrichie pour mode étendu de S(t)
                # ALIGNEMENT NOTEBOOK: config complète + gamma_n_t pré-calculé
                config_for_S = config.copy()
                config_for_S['state'] = state
                config_for_S['history'] = history
                config_for_S['gamma'] = gamma_t
                S_t = dynamics.compute_S(t, An_t, fn_t, phi_n_t, config_for_S, gamma_n_t=gamma_n_t) if hasattr(dynamics, 'compute_S') else 0.0
                
                # NOUVEAU : Ajouter S(t) au debug log
                if 'debug_log_data' in locals():
                    debug_log_data['S_t'] = S_t
                    
                    # Écrire dans un fichier de debug séparé toutes les 10 itérations
                    if step % 10 == 0 and config.get('debug', {}).get('log_detailed', False):
                        debug_file = os.path.join(loggers['output_dir'], f"debug_detailed_{run_id}.csv")
                        
                        # Créer l'en-tête si c'est la première fois
                        if step == 0:
                            with open(debug_file, 'w') as f:
                                # En-tête avec toutes les strates + r(t) et fréquences
                                header = ['t', 'S_t', 'r_t']
                                for n in range(N):
                                    header.extend([f'In_{n}', f'An_{n}', f'En_{n}', f'On_{n}', f'phi_{n}', f'fn_{n}', f'gamma_{n}', f'erreur_{n}', f'G_{n}', f'G_arch_{n}'])
                                f.write(','.join(header) + '\n')
                        
                        # Écrire les données
                        with open(debug_file, 'a') as f:
                            row = [f"{debug_log_data['t']:.3f}", f"{debug_log_data['S_t']:.12g}", f"{debug_log_data['r_t']:.12g}"]
                            for n in range(N):
                                row.extend([
                                    f"{debug_log_data['In_t'][n] if hasattr(debug_log_data['In_t'], '__getitem__') else debug_log_data['In_t']:.12g}",
                                    f"{debug_log_data['An_t'][n]:.12g}",
                                    f"{debug_log_data['En_t'][n]:.12g}",
                                    f"{debug_log_data['On_t'][n]:.12g}",
                                    f"{debug_log_data['phi_n_t'][n]:.12g}",
                                    f"{debug_log_data['fn_t'][n]:.12g}",
                                    f"{debug_log_data['gamma_n_t'][n]:.12g}",
                                    f"{debug_log_data['erreur_n'][n]:.12g}",
                                    f"{debug_log_data['G_values'][n]:.12g}",
                                    f"{debug_log_data['G_archs'][n]}"
                                ])
                            f.write(','.join(row) + '\n')
                
                C_t = dynamics.compute_C(t, phi_n_t) if hasattr(dynamics, 'compute_C') else 0.0
                delta_fn_array = np.zeros(N)
                for n in range(N):
                    S_i = dynamics.compute_S_i(t, n, history, state)
                    delta_fn_array[n] = dynamics.compute_delta_fn(t, state[n]['alpha'], S_i)
                
                A_t = dynamics.compute_A(t, delta_fn_array) if hasattr(dynamics, 'compute_A') else 0.0
                A_spiral_t = dynamics.compute_A_spiral(t, C_t, A_t) if hasattr(dynamics, 'compute_A_spiral') else 0.0
                E_t = dynamics.compute_E(t, An_t) if hasattr(dynamics, 'compute_E') else 0.0
                # L(t) selon FPS_Paper: argmaxₙ |dAₙ(t)/dt| - nécessite historique An
                if hasattr(dynamics, 'compute_L') and len(An_history) >= 2:
                    L_t = dynamics.compute_L(t, An_history, dt)
                else:
                    L_t = 0
            except Exception as e:
                print(f"⚠️ Erreur signaux globaux à t={t}: {e}")
                S_t = 0.0
                C_t = 0.0
                A_t = 0.0
                A_spiral_t = 0.0
                E_t = 0.0
                L_t = 0.0
            
            # ----------- CALCUL DES MÉTRIQUES AVANCÉES -------------
            # Calcul des moyennes pour A_mean(t) et f_mean(t)
            A_mean_t = np.mean(An_t) if isinstance(An_t, np.ndarray) else An_t
            f_mean_t = np.mean(fn_t) if isinstance(fn_t, np.ndarray) else fn_t
            S_history.append(S_t)
            C_history.append(C_t)

            # Calcul de effort(t) avec deltas depuis history (aligné notebook)
            if len(history) > 0:
                delta_An = An_t - history[-1].get('An', An_t)
                delta_fn = fn_t - history[-1].get('fn', fn_t)
                delta_gamma_n = gamma_n_t - history[-1].get('gamma_n', gamma_n_t)
            else:
                delta_An = np.zeros(N)
                delta_fn = np.zeros(N)
                delta_gamma_n = np.zeros(N)
            
            effort_t = metrics.compute_effort(delta_An, delta_fn, delta_gamma_n,
                np.mean(An_t), np.mean(fn_t), gamma_t)

            effort_history.append(effort_t)
            effort_status = metrics.compute_effort_status(effort_t, effort_history, config) if hasattr(metrics, 'compute_effort_status') else "stable"
            
            # Calcul variance_d2S / fluidité (aligné notebook : pas de garde, compute_fluidity gère)
            variance_d2S = metrics.compute_variance_d2S(S_history, dt) if len(S_history) >= 3 else 0
            fluidity = metrics.compute_fluidity(variance_d2S)
            
            # Calcul entropy_S (innovation)
            if len(S_history) >= 10:
                window_size = min(50, len(S_history))
                S_window = S_history[-window_size:]
                entropy_S = metrics.compute_entropy_S(S_window, 1.0/dt) if hasattr(metrics, 'compute_entropy_S') else 0.5
            else:
                entropy_S = 0.1
            
            # === TAU MULTI-ÉCHELLES D'ABORD (tau_S requis par decorrelation après) ===
            if len(S_history) >= 20 and len(history) >= 20:
                try:
                    # Signaux alignés notebook : S, gamma, C
                    signals_for_tau = {
                        'S': S_history,
                        'gamma': [h.get('gamma', gamma_t) for h in history][-len(S_history):],
                        'C': C_history
                    }
                    if hasattr(metrics, 'compute_multiple_tau') and len(signals_for_tau) > 1:
                        multi_tau = metrics.compute_multiple_tau(signals_for_tau, dt)
                        tau_S = multi_tau.get('tau_S', dt)
                        tau_gamma = multi_tau.get('tau_gamma', dt)
                        tau_C = multi_tau.get('tau_C', dt)
                    else:
                        tau_S = tau_gamma = tau_C = dt
                    
                    # tau_A_mean et tau_f_mean via compute_tau_parameter (aligné notebook)
                    tau_A_mean = metrics.compute_tau_parameter(history, 'An_mean(t)') if hasattr(metrics, 'compute_tau_parameter') else dt
                    tau_f_mean = metrics.compute_tau_parameter(history, 'fn_mean(t)') if hasattr(metrics, 'compute_tau_parameter') else dt
                    
                except Exception:
                    tau_S = tau_gamma = tau_C = tau_A_mean = tau_f_mean = dt
            else:
                tau_S = tau_gamma = tau_C = tau_A_mean = tau_f_mean = dt
            
            # === COHÉRENCE TEMPORELLE + DÉCORRÉLATION (utilise tau_S calculé au-dessus) ===
            if len(S_history) >= 20:
                temporal_coherence = metrics.compute_temporal_coherence(S_history, dt) if hasattr(metrics, 'compute_temporal_coherence') else 0.5
                if hasattr(metrics, 'extract_decorrelation_metrics'):
                    decorrelation_time, autocorr_tau = metrics.extract_decorrelation_metrics(temporal_coherence, tau_S)
                else:
                    decorrelation_time = autocorr_tau = dt
            else:
                temporal_coherence = 0.5
                autocorr_tau = dt
                decorrelation_time = dt
            
            # Calcul mean_abs_error (régulation)
            mean_abs_error = metrics.compute_mean_abs_error(En_t, On_t) if hasattr(metrics, 'compute_mean_abs_error') else np.mean(np.abs(En_t - On_t))
            
            # NOUVEAU S1: Détecter alignement En ≈ On
            if mean_abs_error < epsilon_E:
                history_align.append(t)
                
                # Calculer et logger lambda_dyn pour le debug
                lambda_E = config.get('regulation', {}).get('lambda_E', 0.05)
                n_alignments = len(history_align)
                lambda_dyn = lambda_E
                
                if config.get('debug', {}).get('log_alignments', False):
                    print(f"[S1] Alignement #{n_alignments} à t={t:.2f}: |E-O|={mean_abs_error:.4f} < {epsilon_E}, λ_dyn={lambda_dyn:.4f}")
            
            # Calcul mean_high_effort (effort chronique) - nécessite historique
            if len(effort_history) >= 10:
                mean_high_effort = metrics.compute_mean_high_effort(effort_history, 80) if hasattr(metrics, 'compute_mean_high_effort') else np.percentile(effort_history[-100:], 80)
            else:
                mean_high_effort = 0
            
            # Calcul d_effort_dt (effort transitoire)
            if len(effort_history) >= 2:
                d_effort_dt = metrics.compute_d_effort_dt(effort_history, dt) if hasattr(metrics, 'compute_d_effort_dt') else (effort_history[-1] - effort_history[-2]) / dt
            else:
                d_effort_dt = 0.0
            
            # Calcul t_retour (résilience) - après perturbation
            if t_choc is not None and t > t_choc and len(S_history) > int(t_choc/dt):
                t_retour = metrics.compute_t_retour(S_history, int(t_choc/dt), dt, 0.95) if hasattr(metrics, 'compute_t_retour') else 0.0
            else:
                t_retour = 0.0
            
            # Calcul résilience continue - pour perturbations non-ponctuelles
            # Check for perturbations in the new structure
            perturbations_list = config.get('system', {}).get('input', {}).get('perturbations', [])
            perturbation_active = len(perturbations_list) > 0 and any(p.get('type', 'none') != 'none' for p in perturbations_list)
            if len(C_history) >= 20 and len(S_history) >= 20:
                continuous_resilience = metrics.compute_continuous_resilience(
                    C_history, S_history, True
                ) if hasattr(metrics, 'compute_continuous_resilience') else 1.0
            else:
                continuous_resilience = 1.0
            
            # Calcul max_median_ratio (stabilité)
            if len(S_history) >= 10:
                max_median_ratio = metrics.compute_max_median_ratio(S_history) if hasattr(metrics, 'compute_max_median_ratio') else 1.0
            else:
                max_median_ratio = 1.0
            
            # NOUVEAU: Calcul de la résilience adaptative
            adaptive_resilience = 0.0
            adaptive_resilience_score = 3
            if hasattr(metrics, 'compute_adaptive_resilience'):
                # Créer un dict temporaire avec les métriques actuelles
                current_metrics = {
                    't_retour': t_retour,
                    'continuous_resilience': continuous_resilience,
                    'continuous_resilience_mean': np.mean([h.get('continuous_resilience', 1.0) for h in history[-100:] if 'continuous_resilience' in h]) if len(history) > 0 else continuous_resilience
                }
                
                # Calculer la résilience adaptative
                resilience_result = metrics.compute_adaptive_resilience(
                    config, current_metrics, C_history, S_history, 
                    int(t_choc/dt) if t_choc is not None and t > t_choc else None, dt
                )
                
                adaptive_resilience = resilience_result.get('value', 0.0)
                adaptive_resilience_score = resilience_result.get('score', 3)
            
            # NOUVEAU: Calcul des moyennes En, On et gamma pour le logging
            En_mean_t = np.mean(En_t) if isinstance(En_t, np.ndarray) else En_t
            On_mean_t = np.mean(On_t) if isinstance(On_t, np.ndarray) else On_t
            gamma_mean_t = np.mean(gamma_n_t) if isinstance(gamma_n_t, np.ndarray) else gamma_n_t[0] if len(gamma_n_t) > 0 else 1.0
            In_mean_t = np.mean(In_t) if isinstance(In_t, np.ndarray) else In_t
            
            # CRÉER all_metrics ICI
            all_metrics = {
                't': t,
                'S(t)': S_t,
                'C(t)': C_t,
                'A_spiral(t)': A_spiral_t,
                'E(t)': E_t,
                'L(t)': L_t,
                'cpu_step(t)': cpu_step,
                'effort(t)': effort_t,
                'A_mean(t)': A_mean_t,
                'f_mean(t)': f_mean_t,
                'variance_d2S': variance_d2S,
                'fluidity': fluidity,  # Nouvelle métrique de fluidité
                'entropy_S': entropy_S,
                'temporal_coherence': temporal_coherence,  # Cohérence temporelle
                'autocorr_tau': autocorr_tau,  # Temps de décorrélation
                'decorrelation_time': decorrelation_time,  # Alias pour cohérence
                'tau_S': tau_S,  # Tau du signal global (surface fluide)
                'tau_gamma': tau_gamma,  # Tau de gamma (structure profonde)
                'tau_A_mean': tau_A_mean,  # Tau des amplitudes moyennes
                'tau_f_mean': tau_f_mean,  # Tau des fréquences moyennes
                'mean_abs_error': mean_abs_error,
                'mean_high_effort': mean_high_effort,
                'd_effort_dt': d_effort_dt,
                't_retour': t_retour,
                'max_median_ratio': max_median_ratio,
                'continuous_resilience': continuous_resilience,
                'adaptive_resilience': adaptive_resilience,
                'adaptive_resilience_score': adaptive_resilience_score,
                'En_mean(t)': En_mean_t,
                'On_mean(t)': On_mean_t,
                'gamma': gamma_t,  # Ajouter gamma global
                'gamma_mean(t)': gamma_mean_t,
                'phi': phi_reg,  # Phi adaptatif de régulation (alignement notebook)
                'In_mean(t)': In_mean_t,
                'An_mean(t)': A_mean_t,
                'fn_mean(t)': f_mean_t,
                'G_arch_used': G_arch_dominant if 'G_arch_dominant' in locals() else 'tanh',
            }

            all_metrics['tau_C'] = tau_C
            # Extraire la meilleure paire (γ, G) depuis le gamma_journal
            # ALIGNEMENT NOTEBOOK: utilise gamma_journal au lieu de regulation_memory
            best_gamma, best_G, best_score = extract_best_pair_from_journal(
                adaptive_state.get('gamma_journal')
            )
            
            all_metrics['best_pair_gamma'] = float(best_gamma) if best_gamma is not None else float('nan')
            all_metrics['best_pair_G'] = str(best_G) if best_G is not None else ''
            all_metrics['best_pair_score'] = float(best_score) if best_score is not None else float('nan')
            
            # Appliquer safe_float_conversion à toutes les métriques
            # SAUF les champs textuels et ceux où NaN est intentionnel
            skip_safe_convert = {'effort_status', 'G_arch_used', 'best_pair_G', 
                                 'best_pair_gamma', 'best_pair_score', 'tau_A_mean', 'tau_f_mean'}
            for key in all_metrics:
                if key in skip_safe_convert:
                    continue
                all_metrics[key] = safe_float_conversion(all_metrics[key])
            
            # ----------- 4. VÉRIFICATION NaN/Inf SYSTÉMATIQUE -------------
            # Champs où NaN est intentionnel (= pas de données disponibles)
            nan_ok_fields = {'best_pair_gamma', 'best_pair_score', 'tau_A_mean', 'tau_f_mean'}
            nan_inf_detected = False
            for metric_name, metric_value in all_metrics.items():
                if metric_name == 't' or metric_name in nan_ok_fields:
                    continue
                if isinstance(metric_value, (int, float)) and (np.isnan(metric_value) or np.isinf(metric_value)):
                    nan_inf_detected = True
                    alert_msg = f"ALERTE : NaN/Inf détecté à t={t} pour {metric_name}={metric_value}"
                    print(alert_msg)
                    os.makedirs(loggers['output_dir'], exist_ok=True)
                    with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                        alert_file.write(f"{alert_msg}\n")
                    if np.isnan(metric_value):
                        all_metrics[metric_name] = 0.0
                    elif np.isinf(metric_value):
                        all_metrics[metric_name] = 1e6 if metric_value > 0 else -1e6
            
            # Si trop de NaN/Inf, option d'arrêter le run
            if nan_inf_detected and config.get('system', {}).get('stop_on_nan_inf', False):
                raise ValueError(f"NaN/Inf détecté à t={t}, arrêt du run")
            
            # ----------- 5. LOGGING DE TOUTES LES MÉTRIQUES ------------
            metrics_dict = all_metrics.copy()
            metrics_dict['effort_status'] = effort_status
            
            # Créer la ligne de log dans l'ordre des colonnes de config
            log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics_order:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    # Formater correctement les valeurs
                    if isinstance(value, str):
                        row_data.append(value)
                    elif isinstance(value, (int, float)):
                        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                            row_data.append('')  # NaN/Inf intentionnel → cellule vide dans le CSV
                        else:
                            row_data.append(f"{value:.12g}" if value != int(value) else str(int(value)))
                    else:
                        row_data.append(str(value))
                else:
                    row_data.append("0.0")
            
            if hasattr(loggers['csv_writer'], 'writerow'):
                loggers['csv_writer'].writerow(row_data)
                # Flush toutes les 100 itérations pour éviter la perte de données
                if step % 100 == 0 and 'csv_file' in loggers:
                    loggers['csv_file'].flush()
            
            # ----------- 6. EXPORT FICHIERS INDIVIDUELS SI N > 10 ------------
            if N > 10 and individual_csv_writers:
                for n in range(N):
                    # Export A_n(t)
                    if f'A_{n}' in individual_csv_writers:
                        individual_csv_writers[f'A_{n}']['writer'].writerow([t, An_t[n]])
                    # Export f_n(t)
                    if f'f_{n}' in individual_csv_writers:
                        individual_csv_writers[f'f_{n}']['writer'].writerow([t, fn_t[n]])
            
            # ----------- 7. BACKUP AUTOMATIQUE ------------------------
            if step % backup_interval == 0 and step > 0:
                if hasattr(utils, "save_simulation_state"):
                    # Créer le dossier checkpoints s'il n'existe pas
                    checkpoint_dir = os.path.join(loggers.get('output_dir', '.'), 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}_backup_{step}.pkl")
                    utils.save_simulation_state(state, checkpoint_path)
            
            # NOUVEAU : Sauvegarder périodiquement les découvertes couplées
            # Pour T=50 avec dt=0.1, on a 500 steps, donc sauvegarder tous les 100 steps
            save_interval = min(1000, max(100, int(len(t_array) / 5)))  # Au moins 5 sauvegardes
            if gamma_mode == 'adaptive_aware' and step % save_interval == 0 and step > 0:
                if adaptive_state['gamma_journal'] is not None:
                    discoveries_path = os.path.join(
                        loggers.get('output_dir', '.'), 
                        f"discoveries_coupled_{run_id}_step{step}.json"
                    )
                    utils.save_coupled_discoveries(
                        adaptive_state['gamma_journal'], 
                        adaptive_state['regulation_state'], 
                        discoveries_path
                    )
                    if config.get('debug', {}).get('save_discoveries', False):
                        print(f"  💡 Découvertes sauvegardées: {discoveries_path}")
            
            # ----------- 8. HISTORIQUE POUR ANALYSE -------------------
            # Calculer G_values_array et S_contrib_t pour la visualisation par strate
            G_values_list = debug_log_data.get('G_values', [])
            G_values_array = np.array(G_values_list, dtype=float) if len(G_values_list) == N else np.zeros(N)
            # S_contrib_t = On_t * G_values_array (contribution de chaque strate au signal)
            S_contrib_t = (On_t * G_values_array) if isinstance(On_t, np.ndarray) else np.zeros(N)

            history.append({
                't': t, 'S': S_t, 'O': On_t.copy(), 'E': En_t.copy(), 'F': F_n_t_An,
                'gamma_n': gamma_n_t.copy(), 'An': An_t.copy(), 'fn': fn_t.copy(),
                'phi_n_t': phi_n_t,  # AJOUT PR: phases par strate (requis par visualize_stratum_patterns)
                'S_contrib': S_contrib_t,  # AJOUT PR: contribution de chaque strate à S(t)
                'G_values_array': G_values_array,  # AJOUT PR: G par strate (array numpy)
                'C': C_t, 'A_spiral': A_spiral_t, 'entropy_S': entropy_S,
                'delta_fn': delta_fn_t, 'S(t)': S_t, 'C(t)': C_t,
                'effort(t)': effort_t, 'cpu_step(t)': cpu_step,
                'A_mean(t)': A_mean_t, 'f_mean(t)': f_mean_t,
                'variance_d2S': variance_d2S, 'fluidity': fluidity,
                'mean_abs_error': mean_abs_error,
                'effort_status': effort_status,
                'En_mean(t)': En_mean_t,
                'On_mean(t)': On_mean_t,
                'gamma_mean(t)': gamma_mean_t,
                'In_mean(t)': In_mean_t,
                'In': In_t,
                'An_mean(t)': A_mean_t,
                'fn_mean(t)': f_mean_t,
                'continuous_resilience': continuous_resilience,
                'adaptive_resilience': adaptive_resilience,
                'adaptive_resilience_score': adaptive_resilience_score,
                'gamma': gamma_t,
                'phi': phi_reg,  # Phi adaptatif de régulation
                'G_arch_used': G_arch_dominant if 'G_arch_dominant' in locals() else 'tanh',
                'gamma_regime': gamma_regime if 'gamma_regime' in locals() else 'unknown',
                'best_pair': {'gamma': best_gamma, 'G_arch': best_G, 'score': best_score} if best_gamma is not None else {},
                'best_pair_gamma': float(best_gamma) if best_gamma is not None else None,
                'best_pair_G': str(best_G) if best_G is not None else None,
                'best_pair_score': float(best_score) if best_score is not None else float('nan'),
                'G_values': G_values_list,
                'G_archs': debug_log_data.get('G_archs', []),
                'tau_C': tau_C
            })
            cpu_steps.append(cpu_step)
            En_history.append(En_t.copy() if isinstance(En_t, np.ndarray) else En_t)
            On_history.append(On_t.copy() if isinstance(On_t, np.ndarray) else On_t)
            
            # Rotation des historiques pour éviter explosion mémoire
            if len(history) > MAX_HISTORY_SIZE:
                history = rotate_history(history, MAX_HISTORY_SIZE)
                cpu_steps = rotate_history(cpu_steps, MAX_HISTORY_SIZE)
                effort_history = rotate_history(effort_history, MAX_HISTORY_SIZE)
                S_history = rotate_history(S_history, MAX_HISTORY_SIZE)
                An_history = rotate_history(An_history, MAX_HISTORY_SIZE)
                fn_history = rotate_history(fn_history, MAX_HISTORY_SIZE)
                En_history = rotate_history(En_history, MAX_HISTORY_SIZE)
                On_history = rotate_history(On_history, MAX_HISTORY_SIZE)
                
                # Logger la rotation
                if step % 1000 == 0:
                    print(f"  📊 Rotation des historiques à t={t:.1f} (max_size={MAX_HISTORY_SIZE})")
            
            # ----------- 9. DÉTECTION MODE ALERTE (>3σ) ---------------
            # Vérifier si un critère non-déclencheur présente un écart > 3σ
            alert_sigma = config.get('validation', {}).get('alert_sigma', 3)
            if len(S_history) >= 100:  # Besoin d'historique pour calculer σ
                # Exemple pour entropy_S
                entropy_history = [h.get('entropy_S', 0.5) for h in history[-100:] if 'entropy_S' in h]
                if len(entropy_history) > 0:
                    entropy_mean = np.mean(entropy_history)
                    entropy_std = np.std(entropy_history)
                    if entropy_std > 0 and abs(entropy_S - entropy_mean) > alert_sigma * entropy_std:
                        alert_msg = f"MODE ALERTE : entropy_S={entropy_S:.4f} dévie de >{alert_sigma}σ à t={t}"
                        print(alert_msg)
                        with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                            alert_file.write(f"{alert_msg}\n")
                
                # Même vérification pour d'autres métriques non-déclencheuses
                for metric_name in ['variance_d2S', 'mean_high_effort', 'd_effort_dt']:
                    if metric_name in all_metrics:
                        metric_history = [h.get(metric_name, 0) for h in history[-100:] if metric_name in h]
                        if len(metric_history) > 10:
                            m_mean = np.mean(metric_history)
                            m_std = np.std(metric_history)
                            if m_std > 0 and abs(all_metrics[metric_name] - m_mean) > alert_sigma * m_std:
                                alert_msg = f"MODE ALERTE : {metric_name}={all_metrics[metric_name]:.4f} dévie de >{alert_sigma}σ à t={t}"
                                with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                                    alert_file.write(f"{alert_msg}\n")

            # Ajouter après chaque writerow
            if config.get('debug', False):
                print(f"[DEBUG] t={t:.2f}: S={S_t:.4f}, effort={effort_t:.4f}")

        # -- FERMETURE DES FICHIERS INDIVIDUELS --
        if N > 10 and individual_csv_writers:
            for key, writer_info in individual_csv_writers.items():
                writer_info['file'].close()
            print(f"Fichiers individuels A_n_*.csv et f_n_*.csv exportés pour N={N}")
        
        # -- SAUVEGARDE FINALE DES DÉCOUVERTES ADAPTATIVES --
        if gamma_mode == 'adaptive_aware' and adaptive_state['gamma_journal'] is not None:
            final_discoveries_path = os.path.join(
                loggers.get('output_dir', '.'), 
                f"discoveries_coupled_{run_id}_final.json"
            )
            utils.save_coupled_discoveries(
                adaptive_state['gamma_journal'], 
                adaptive_state['regulation_state'], 
                final_discoveries_path
            )
            print(f"  💡 Découvertes finales sauvegardées: {final_discoveries_path}")

        # -- EXPORTS ET SYNTHÈSE FINALE --
        logs = loggers['log_file']
        
        # Fermer le fichier CSV maintenant que la simulation est terminée
        if 'csv_file' in loggers and loggers['csv_file']:
            try:
                loggers['csv_file'].close()
                print(f"✓ Fichier CSV fermé : {logs}")
            except:
                pass
        
        # Calcul des statistiques finales pour le résumé
        # Calculer la moyenne de continuous_resilience depuis l'historique
        continuous_resilience_values = []
        for h in history:
            if 'continuous_resilience' in h:
                continuous_resilience_values.append(h['continuous_resilience'])
        continuous_resilience_mean = np.mean(continuous_resilience_values) if continuous_resilience_values else float(continuous_resilience) if continuous_resilience is not None else 1.0
        
        # Calculer moyennes sur l'historique pour cohérence avec système adaptatif
        entropy_history = [h.get('entropy_S', 0.5) for h in history if 'entropy_S' in h]
        mean_entropy_S = np.mean(entropy_history) if entropy_history else float(entropy_S) if entropy_S is not None else 0.0
        
        metrics_summary = {
            'mean_S': np.mean(S_history) if S_history else 0.0,
            'std_S': np.std(S_history) if S_history else 0.0,
            'mean_effort': np.mean(effort_history) if effort_history else 0.0,
            'max_effort': np.max(effort_history) if effort_history else 0.0,
            'mean_cpu_step': np.mean(cpu_steps) if cpu_steps else 0.0,
            'final_entropy_S': float(entropy_S) if entropy_S is not None else 0.0,
            'entropy_S': float(mean_entropy_S),  # NOUVEAU: moyenne pour cohérence avec système adaptatif
            'final_variance_d2S': float(variance_d2S) if variance_d2S is not None else 0.0,
            'final_fluidity': float(fluidity) if 'fluidity' in locals() and fluidity is not None else 0.0,
            'final_mean_abs_error': float(mean_abs_error) if mean_abs_error is not None else 0.0,
            'mean_C': float(np.mean(C_history)) if C_history else float('nan'),
            'resilience_t_retour': float(t_retour) if t_retour is not None else 0.0,
            'continuous_resilience': float(continuous_resilience) if continuous_resilience is not None else 1.0,
            'continuous_resilience_mean': float(continuous_resilience_mean),
            'adaptive_resilience': float(adaptive_resilience) if 'adaptive_resilience' in locals() else 0.0,
            'adaptive_resilience_score': int(adaptive_resilience_score) if 'adaptive_resilience_score' in locals() else 3,
            'stability_ratio': float(max_median_ratio) if max_median_ratio is not None else 1.0,
            'total_steps': len(t_array),
            'recorded_steps': len(S_history),
            'dt': float(dt),
            'N': int(N),
            'mode': 'FPS'
        }
        
        # Ne PAS appeler summarize_metrics avec le nom du fichier
        # Si on veut utiliser summarize_metrics, il faut lui passer history, pas logs
        # if hasattr(metrics, "summarize_metrics") and history:
        #     metrics_summary.update(metrics.summarize_metrics(history))
        
        # Calcul du checksum des logs pour intégrité
        if hasattr(utils, "compute_checksum") and os.path.exists(logs):
            checksum = utils.compute_checksum(logs)
            with open(os.path.join(loggers['output_dir'], f"checksum_{run_id}.txt"), "w") as f:
                f.write(f"Checksum for {logs}: {checksum}\n")
        
        # 👉 Vérifier que la simulation a couvert toutes les étapes prévues
        expected_steps = len(t_array)
        actual_steps = len(S_history)
        if actual_steps < expected_steps:
            msg = f"La simulation s'est terminée prématurément : {actual_steps}/{expected_steps} steps enregistrés"
            print(f"⚠️  {msg}")
            if strict:
                raise RuntimeError(msg)
        
        return deep_convert({
            'history': history,
            'logs': logs,
            'metrics': metrics_summary,
            'cpu_steps': cpu_steps,
            'effort_history': effort_history,
            'S_history': S_history,
            'run_id': run_id,
            'An_history': An_history,
            'fn_history': fn_history,
            'En_history': En_history,
            'On_history': On_history,
            'gamma_journal': adaptive_state.get('gamma_journal'),
        })

    except Exception as e:
        # -- LOG D'ERREUR CRITIQUE POUR POST-MORTEM --
        err_path = os.path.join(loggers['output_dir'], f"error_{run_id}.log")
        with open(err_path, "a") as f: 
            f.write(f"Error at t={t if 't' in locals() else 'unknown'}\n")
            f.write(traceback.format_exc())
        
        # Fermer les fichiers individuels en cas d'erreur
        if N > 10 and 'individual_csv_writers' in locals():
            for key, writer_info in individual_csv_writers.items():
                try:
                    writer_info['file'].close()
                except:
                    pass
        
        if hasattr(utils, "handle_crash_recovery"): 
            crash_state = {
                'strates': state,
                't': t if 't' in locals() else 0,
                'mode': config['system'].get('mode', 'FPS'),
                'error_info': str(e),
                'all_metrics': all_metrics  # Maintenant toujours défini
            }
            utils.handle_crash_recovery(crash_state, loggers, e)
        raise e

# --- MODE KURAMOTO (GROUPE CONTRÔLE) ---------------------------
def run_kuramoto_simulation(config, loggers):
    """
    Implémentation stricte de la dynamique Kuramoto (phase-only).
    - Utilise K=0.5, ωᵢ~U[0,1], phases initialisées random, N strates, dt, T comme FPS
    - Log les mêmes métriques que FPS pour comparaison
    """
    if hasattr(kuramoto, "run_kuramoto_simulation"):
        return kuramoto.run_kuramoto_simulation(config, loggers)
    else:
        # Implémentation minimale Kuramoto
        N = config['system']['N']
        T = config['system']['T']
        dt = config['system'].get('dt', 0.05)
        K = 0.5
        
        # Initialisation
        phases = np.random.uniform(0, 2*np.pi, N)
        frequencies = np.random.uniform(0, 1, N)
        t_array = np.arange(0, T, dt)
        
        history = []
        S_history = []
        C_history = []
        cpu_steps = []
        
        for t in t_array:
            step_start = time.perf_counter()
            
            # Équation Kuramoto
            dphases_dt = frequencies.copy()
            for i in range(N):
                coupling_sum = 0.0
                for j in range(N):
                    coupling_sum += np.sin(phases[j] - phases[i])
                dphases_dt[i] += (K / N) * coupling_sum
            
            # Mise à jour
            phases += dphases_dt * dt
            phases = phases % (2 * np.pi)
            
            # Ordre global (paramètre d'ordre de Kuramoto)
            order_param = np.abs(np.mean(np.exp(1j * phases)))
            
            # Cohérence des phases adjacentes (équivalent à C(t))
            if N > 1:
                C_t = np.mean([np.cos(phases[(i+1)%N] - phases[i]) for i in range(N)])
            else:
                C_t = 1.0
            
            # Signal global (somme des oscillateurs)
            S_t = np.sum(np.sin(phases))
            
            cpu_step = (time.perf_counter() - step_start) / N
            
            # Log
            metrics_dict = {
                't': t,
                'S(t)': S_t,
                'C(t)': C_t,
                'E(t)': order_param,
                'L(t)': 0,
                'cpu_step(t)': cpu_step,
                'effort(t)': 0.0,
                'A_mean(t)': 1.0,
                'f_mean(t)': np.mean(frequencies),
                'effort_status': 'stable'
            }
            
            # Écrire dans le CSV
            log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics_order:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    if isinstance(value, str):
                        row_data.append(value)
                    else:
                        row_data.append(f"{value:.12g}")
                else:
                    row_data.append("0.0")
            
            if hasattr(loggers['csv_writer'], 'writerow'):
                loggers['csv_writer'].writerow(row_data)
            
            history.append({'t': t, 'S': S_t, 'C': C_t, 'order': order_param})
            S_history.append(S_t)
            C_history.append(C_t)
            cpu_steps.append(cpu_step)
        
        # Fermer le fichier CSV
        if 'csv_file' in loggers and loggers['csv_file']:
            try:
                loggers['csv_file'].close()
            except:
                pass
        
        return deep_convert({
            "logs": loggers.get('log_file', 'kuramoto_log.csv'),
            "metrics": {
                'mean_S': float(np.mean(S_history)) if S_history else 0.0,
                'std_S': float(np.std(S_history)) if S_history else 0.0,
                'mean_C': float(np.mean(C_history)) if C_history else 0.0,
                'mean_cpu_step': float(np.mean(cpu_steps)) if cpu_steps else 0.0,
                'final_order': float(order_param) if order_param is not None else 0.0,
                'mode': 'Kuramoto'
            },
            "history": history,
            "run_id": loggers['run_id'],
            "S_history": S_history
        })

# --- MODE NEUTRAL (OSCs FIXES, PAS DE FEEDBACK) ----------------
def run_neutral_simulation(config, loggers):
    """
    Simulation neutre : phases/amplitudes fixes, sans rétroaction ni spiralisation.
    Sert de contrôle pour valider l'émergence spécifique de la FPS.
    """
    N = config['system']['N']
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    t_array = np.arange(0, T, dt)
    
    # Paramètres fixes
    amplitudes = np.ones(N)
    frequencies = np.linspace(0.8, 1.2, N)  # Fréquences légèrement différentes
    phases = np.zeros(N)
    
    history = []
    S_history = []
    cpu_steps = []
    
    for t in t_array:
        step_start = time.perf_counter()
        
        # Signal sans feedback
        S_t = np.sum(amplitudes * np.sin(2 * np.pi * frequencies * t + phases))
        
        # Pas de modulation, pas de feedback
        C_t = 1.0  # Cohérence fixe
        E_t = np.max(amplitudes)
        L_t = 0
        
        cpu_step = (time.perf_counter() - step_start) / N
        
        # Log
        metrics_dict = {
            't': t,
            'S(t)': S_t,
            'C(t)': C_t,
            'E(t)': E_t,
            'L(t)': L_t,
            'cpu_step(t)': cpu_step,
            'effort(t)': 0.0,
            'A_mean(t)': 1.0,
            'f_mean(t)': np.mean(frequencies),
            'effort_status': 'stable',
            'variance_d2S': 0.0,
            'entropy_S': 0.0,
            'mean_abs_error': 0.0
        }
        
        # Écrire dans le CSV
        log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
        row_data = []
        for metric in log_metrics_order:
            if metric in metrics_dict:
                value = metrics_dict[metric]
                if isinstance(value, str):
                    row_data.append(value)
                else:
                    row_data.append(f"{value:.12g}")
            else:
                row_data.append("0.0")
        
        if hasattr(loggers['csv_writer'], 'writerow'):
            loggers['csv_writer'].writerow(row_data)
        
        history.append({'t': t, 'S': S_t})
        S_history.append(S_t)
        cpu_steps.append(cpu_step)
    
    # Fermer le fichier CSV
    if 'csv_file' in loggers and loggers['csv_file']:
        try:
            loggers['csv_file'].close()
        except:
            pass
    
    return deep_convert({
        "logs": loggers.get('log_file', 'neutral_log.csv'),
        "metrics": {
            'mean_S': float(np.mean(S_history)) if S_history else 0.0,
            'std_S': float(np.std(S_history)) if S_history else 0.0,
            'mean_cpu_step': float(np.mean(cpu_steps)) if cpu_steps else 0.0,
            'mode': 'Neutral'
            },
        "history": history,
        "run_id": loggers['run_id'],
        "S_history": S_history
    })

# --------- AFFICHAGE TODO ET INCOMPLETS DU PIPELINE -------------
def list_todos():
    print("\n--- TODO FPS PIPELINE ---")
    print("Compléter/raffiner tous les modules suivants pour une version 100% rigoureuse et falsifiable :")
    print("  * Animation 3D de la spirale")
    print("  * Dashboard temps réel")
    print("- Documentation : README, matrice critère-terme, journal seuils\n")

# --------- ARGPARSE POUR EXECUTION CLI -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FPS Simulation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--mode", type=str, default="FPS", choices=["FPS", "Kuramoto", "neutral"], help="Simulation mode")
    parser.add_argument("--list-todos", action="store_true", help="List all TODO items")
    parser.add_argument("--strict", action="store_true", help="Interrompre si le run ne couvre pas tous les pas prévus (falsifiabilité stricte)")
    args = parser.parse_args()
    
    if args.list_todos:
        list_todos()
    else:
        result = run_simulation(args.config, args.mode, strict=args.strict)
        print(f"\nSimulation terminée : {result['run_id']}")
        print(f"Logs : {result['logs']}")
        print(f"Métriques finales : {result['metrics']}")

"""
-----------------------
FIN simulate.py FPS V1.3 (exhaustif, strict, explicite)
Ce fichier doit servir de **référence** pour toutes les évolutions du pipeline FPS. 
Chaque fonction à compléter/raffiner l'est dans les modules correspondants.
"""