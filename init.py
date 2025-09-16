import json
import random
import numpy as np
import pprint
from datetime import datetime
import os
import sys
from utils import deep_convert, generate_spiral_weights

# Import correct de validate_config
sys.path.append(os.path.dirname(__file__))
from validate_config import validate_config

"""
init.py - Initialisation et validation du système FPS/Kuramoto
Phase 1 du projet FPS - Version structurée et robuste

Fonctionnalités :
- Chargement du config.json exhaustif (phase 1)
- Initialisation des strates, dynamiques locales/globales
- Setup des logs, métriques, seeds
- Validation complète (blocs, dynamiques, métriques)
- Historique global sécurisé avec avertissement mémoire
- API d'extension pour variantes via post_init_callback
- Vérification stricte des matrices de poids
- Initialisation des paramètres dynamiques (gamma_n, mu_n, sigma_n)

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

def load_config(config_path='config.json'):
    """Charge le fichier de configuration JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)

def set_seed(seed):
    """Fixe la seed pour la reproductibilité."""
    # Vérifier si la seed est déjà initialisée pour éviter les doubles initialisations
    try:
        current_seed = np.random.get_state()[1][0]
        if current_seed == seed:
            print(f"🌱 Seed {seed} déjà initialisée, pas de double init")
            return
    except:
        pass  # Si erreur, procéder normalement
    
    np.random.seed(seed)
    random.seed(seed)
    print(f"🌱 Seed initialisée : {seed}")

def verify_weight_matrix(w, strate_id, epsilon: float = 1e-8, enforce_zero_sum: bool = True):
    """Vérifie la cohérence d'une ligne de poids.

    Paramètres
    ----------
    w : list[float]
        Ligne de la matrice de poids.
    strate_id : int
        Index de la strate courante.
    epsilon : float
        Tolérance numérique.
    enforce_zero_sum : bool, optional
        Lorsque *False* on n'impose pas Σw[i] = 0. Utile pour les extrémités
        d'une spirale ouverte (couplage non conservatif aux bords).
    """
    errors = []

    # Diagonale nulle
    if abs(w[strate_id]) > epsilon:
        errors.append(
            f"La diagonale w[{strate_id}][{strate_id}] = {w[strate_id]:.6f} doit être 0")

    # Conservation du signal si exigée
    if enforce_zero_sum:
        sum_w = sum(w)
        if abs(sum_w) > epsilon:
            errors.append(
                f"La somme des poids w[{strate_id}] = {sum_w:.6f} doit être 0 (conservation)")

    return len(errors) == 0, errors

def init_strates(config):
    """
    Initialise toutes les strates avec validation stricte.
    Inclut l'initialisation des paramètres dynamiques gamma_n, mu_n, sigma_n.
    """
    dynamic_params = config.get("dynamic_parameters", {})
    latence_config = config.get("latence", {})
    enveloppe_config = config.get("enveloppe", {})
    
    strates = []
    weight_errors = []
    
    # ----- NOUVEAU : génération automatique des poids spiralés -----
    coupling_cfg = config.get("coupling", {})
    coupling_type = str(coupling_cfg.get("type", "")).lower()
    spiral_mode = coupling_type in {"spiral", "ring"}
    mirror_mode = coupling_cfg.get("mirror", False)
    W_spiral = None
    if spiral_mode:
        N_total = len(config["strates"])
        c_val = coupling_cfg.get("c", 0.25)
        # Pour 'ring', on force closed=True si non précisé
        if coupling_type == "ring":
            closed_val = True
        else:
            closed_val = coupling_cfg.get("closed", False)

        W_spiral = generate_spiral_weights(N_total, c=c_val, closed=closed_val, mirror=mirror_mode)
        print(f"🔄 Génération matrice de poids '{coupling_type}' (c={c_val}, closed={closed_val}, mirror={mirror_mode})")
    
    for i, s in enumerate(config['strates']):
        # Sélection des poids : priorité au mode spiral s'il est activé
        if W_spiral is not None:
            w = W_spiral[i]
        else:
            w = s.get('w', [])
        
        # Vérification des poids (tolérance spéciale pour extrémités spirale ouverte)
        spiral_open = spiral_mode and not closed_val
        skip_edges = spiral_open and not mirror_mode and (i == 0 or i == N_total - 1)
        enforce_sum = not skip_edges
        is_valid, w_errors = verify_weight_matrix(w, i, enforce_zero_sum=enforce_sum)
        if not is_valid:
            weight_errors.extend([f"Strate {i}: {err}" for err in w_errors])
        
        # Détection des dynamiques pour cette strate
        dyn_phi = dynamic_params.get("dynamic_phi", False) or s.get("dynamic_phi", False)
        dyn_alpha = dynamic_params.get("dynamic_alpha", False) or s.get("dynamic_alpha", False)
        dyn_beta = dynamic_params.get("dynamic_beta", False) or s.get("dynamic_beta", False)
        
        # Initialisation gamma_n selon config
        if latence_config.get("gamma_n_mode") == "dynamic":
            gamma_n_init = 0.5  # Valeur initiale pour mode dynamique
            gamma_n_params = latence_config.get("gamma_n_dynamic", {"k_n": 2.0, "t0_n": 50})
        else:
            gamma_n_init = 1.0  # Mode statique
            gamma_n_params = {}
        
        # Initialisation mu_n selon config
        mu_n_init = enveloppe_config.get("mu_n", 0.0)
        
        # Initialisation sigma_n selon config
        if enveloppe_config.get("env_mode") == "dynamic":
            sigma_n_init = enveloppe_config.get("sigma_n_static", 0.1)
            sigma_n_params = enveloppe_config.get("sigma_n_dynamic", {
                "amp": 0.05, "freq": 1, "offset": 0.1, "T": 100
            })
        else:
            sigma_n_init = enveloppe_config.get("sigma_n_static", 0.1)
            sigma_n_params = {}
        
        # Structure complète de la strate
        strate = {
            'id': i,
            'A0': s['A0'],
            'f0': s['f0'],
            'phi': s.get('phi', 0.0),
            'alpha': s['alpha'],
            'beta': s['beta'],
            'k': s['k'],
            'x0': s['x0'],
            'w': w,
            # États dynamiques
            'An': s['A0'],
            'fn': s['f0'],
            'gamma_n': gamma_n_init,
            'gamma_n_params': gamma_n_params,
            'mu_n': mu_n_init,
            'sigma_n': sigma_n_init,
            'sigma_n_params': sigma_n_params,
            # États internes
            'En': s['A0'],
            'On': 0.0,
            'history': [],
            # Flags dynamiques
            'dynamic_phi': dyn_phi,
            'dynamic_alpha': dyn_alpha,
            'dynamic_beta': dyn_beta
        }
        strates.append(strate)
    
    # Rapport des erreurs de poids si présentes
    if weight_errors:
        print("\n❌ ERREUR - Matrices de poids invalides:")
        for err in weight_errors:
            print(f"  - {err}")
        print("\nLes poids DOIVENT respecter:")
        print("  - w[i][i] = 0 (pas d'auto-connexion)")
        print("  - Σw[i] = 0 (conservation du signal)")
        print("\nCorrection automatique appliquée...")
        
        # Corriger automatiquement les poids
        for strate in strates:
            w = strate['w']
            # Forcer la diagonale à zéro
            if strate['id'] < len(w):
                w[strate['id']] = 0.0
            # Ajuster pour que la somme soit nulle
            w_sum = sum(w)
            if abs(w_sum) > 1e-8:
                # Redistribuer l'écart sur tous les poids non-diagonaux
                non_diag_count = len(w) - 1
                if non_diag_count > 0:
                    correction = -w_sum / non_diag_count
                    for j in range(len(w)):
                        if j != strate['id']:
                            w[j] += correction
        # On continue mais on log l'avertissement
        os.makedirs("logs", exist_ok=True)
        with open("logs/weight_validation.txt", "a") as f:
            f.write(f"{datetime.now()} - Validation des poids:\n")
            for err in weight_errors:
                f.write(f"  {err}\n")
    
    return deep_convert(strates)

def setup_logging(config, log_dir="logs", mode_suffix=None):
    """
    Configure le système de logging avec gestion des dossiers.
    Retourne un dictionnaire avec la structure attendue par simulate.py.
    
    Args:
        config: Configuration du système
        log_dir: Dossier de logs (défaut: "logs")
        mode_suffix: Suffixe optionnel pour différencier les modes (ex: "FPS", "Kuramoto", "Neutral")
    """
    seed = config['system']['seed']
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Ajouter le suffixe si fourni
    if mode_suffix:
        run_id = f"run_{now}_{mode_suffix}_seed{seed}"
    else:
        run_id = f"run_{now}_seed{seed}"
    
    log_file = os.path.join(log_dir, f"{run_id}.csv")
    
    # Log de la seed
    with open(os.path.join(log_dir, "seeds.txt"), "a") as f:
        f.write(f"{now} | {mode_suffix or 'DEFAULT'} | SEED = {seed}\n")
    
    # Préparer le writer CSV
    csv_file = open(log_file, 'w', newline='')
    import csv
    csv_writer = csv.writer(csv_file)
    
    # Écrire les en-têtes selon la config
    log_metrics = config['system']['logging'].get('log_metrics', ['t'])
    csv_writer.writerow(log_metrics)
    
    # Structure de retour attendue par simulate.py
    return {
        'csv_writer': csv_writer,
        'csv_file': csv_file,  # Pour pouvoir fermer le fichier plus tard
        'run_id': run_id,
        'output_dir': log_dir,
        'log_file': log_file
    }

def prepare_log_files(log_path, metrics):
    """Prépare les fichiers de log avec les en-têtes."""
    with open(log_path, 'w') as f:
        f.write(','.join(metrics) + '\n')

def initialize_system(config, post_init_callback=None):
    """
    Crée et retourne la structure d'état du système.
    
    Args:
        config: Configuration complète du système
        post_init_callback: Fonction optionnelle appelée après l'initialisation.
                           Permet d'ajouter, patcher ou monitorer l'état sans toucher le core.
                           Signature: callback(system_state) -> None
    
    Exemple d'usage:
        def patch_for_kuramoto(system_state):
            system_state['mode'] = 'Kuramoto'
            system_state['kuramoto_specific'] = {...}
        
        system_state = initialize_system(config, post_init_callback=patch_for_kuramoto)
    """
    set_seed(config['system']['seed'])
    strates = init_strates(config)
    N = config['system']['N']
    T = config['system'].get('T', 100)
    
    # Estimation mémoire avec calcul précis
    n_metrics = len(config['system']['logging']['log_metrics'])
    estimated_points = N * T * n_metrics
    estimated_mb = (estimated_points * 8) / (1024 * 1024)  # 8 bytes par float64
    safe_limit = 5_000_000  # seuil adapté pour longues simulations
    
    if estimated_points > safe_limit:
        warning_msg = (
            f"⚠️  Attention : L'historique complet va contenir ~{estimated_points:,} points.\n"
            f"   Estimation mémoire : ~{estimated_mb:.1f} MB\n"
            f"   Cela peut dépasser la RAM sur une longue exécution !\n"
            f"   Paramètres actuels : N={N}, T={T}, {n_metrics} métriques\n"
            "   Options : réduire N/T/log_metrics ou activer la compression"
        )
        print(warning_msg)
        os.makedirs("logs", exist_ok=True)
        with open("logs/warnings.txt", "a") as wf:
            wf.write(f"{datetime.now()} - {warning_msg}\n")
        
        # Pause pour confirmation
        response = input("\nAppuyer sur Entrée pour continuer malgré tout, ou 'q' pour quitter: ")
        if response.lower() == 'q':
            exit(0)
    
    # Initialisation de l'historique global
    history = {m: [] for m in config['system']['logging']['log_metrics']}
    
    # Note sur les seuils théoriques
    print("\n📝 Note: Les seuils dans 'to_calibrate' sont des valeurs initiales théoriques.")
    print("   Ils seront ajustés après les 5 premiers runs de calibration.")
    
    # Construction de l'état système complet
    system_state = {
        'strates': strates,
        't': 0,
        'config': config,
        'mode': config['system'].get('mode', 'FPS'),
        'perturbation': config['system'].get('perturbation', {}),
        'logs': {},
        'run_id': None,
        'history': history,
        'exploration': config.get('exploration', {}),
        'dynamic_parameters': config.get('dynamic_parameters', {}),
        'regulation': config.get('regulation', {}),
        'latence': config.get('latence', {}),
        'enveloppe': config.get('enveloppe', {}),
        'validation': config.get('validation', {}),
        'analysis': config.get('analysis', {}),
        # Métadonnées
        'init_timestamp': datetime.now().isoformat(),
        'fps_version': '1.3',
        'weight_validation_passed': len(weight_errors) == 0 if 'weight_errors' in locals() else True
    }
    
    # Appel du callback si fourni
    if post_init_callback is not None:
        try:
            post_init_callback(system_state)
            print(f"✓ Post-init callback '{post_init_callback.__name__}' exécuté avec succès")
        except Exception as e:
            print(f"⚠️  Erreur dans post-init callback: {e}")
            os.makedirs("logs", exist_ok=True)
            with open("logs/warnings.txt", "a") as wf:
                wf.write(f"{datetime.now()} - Erreur post-init callback: {e}\n")
    
    return deep_convert(system_state)

# Exécution principale (test)
if __name__ == "__main__":
    config_path = 'config.json'
    
    # Validation complète AVANT le chargement
    try:
        errors, warnings = validate_config(config_path)
        if errors:
            print("❌ Erreurs de validation:")
            for e in errors:
                print(f"  - {e}")
            exit(1)
        if warnings:
            print("⚠️ Avertissements:")
            for w in warnings:
                print(f"  - {w}")
    except Exception as e:
        print(f"Erreur de validation config.json : {e}")
        exit(1)
    
    # Chargement de la config après validation
    config = load_config(config_path)
    
    # Setup logging avec la nouvelle structure
    loggers = setup_logging(config)
    
    # Exemple avec callback
    def example_callback(state):
        """Exemple de callback pour extension."""
        state['custom_extension'] = {
            'timestamp': datetime.now().isoformat(),
            'custom_params': {'test': True}
        }
        print("  → Callback: ajout de paramètres custom")
    
    # Initialisation avec callback
    system_state = initialize_system(config, post_init_callback=example_callback)
    
    print(f"\n✅ Initialisation réussie :")
    print(f"   - {len(system_state['strates'])} strates initialisées")
    print(f"   - Mode: {system_state['mode']}")
    print(f"   - Fichier log: {loggers['log_file']}")
    print(f"   - Run ID: {loggers['run_id']}")
    print(f"   - Validation des poids: {'✓' if system_state['weight_validation_passed'] else '⚠️'}")
    
    # Affichage détaillé pour contrôle visuel
    print("\nÉtat du système (aperçu):")
    print(f"  Strates[0]: gamma_n={system_state['strates'][0]['gamma_n']}, "
          f"mu_n={system_state['strates'][0]['mu_n']}, "
          f"sigma_n={system_state['strates'][0]['sigma_n']}")
    
    if 'custom_extension' in system_state:
        print(f"  Extensions: {list(system_state['custom_extension'].keys())}")
    
    # Fermer le fichier CSV de test
    if 'csv_file' in loggers:
        loggers['csv_file'].close()