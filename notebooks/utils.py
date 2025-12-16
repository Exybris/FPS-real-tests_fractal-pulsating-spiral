"""
utils.py - Fonctions utilitaires pour le système FPS
Version complète conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module contient toutes les fonctions transversales qui
facilitent l'orchestration du système FPS :

- Gestion des logs et fusion de données
- Sauvegarde et restauration d'états
- Exécution parallèle de runs
- Exports en formats multiples
- Génération d'identifiants uniques
- Gestion de la structure des dossiers

Chaque fonction est conçue pour la robustesse, la traçabilité
et la facilité d'utilisation dans l'écosystème FPS.

(c) 2025 Gepetto & Andréa Gadal & Claude (Anthropic) 🌀
"""

import os
import csv
import json
import pickle
import hashlib
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import warnings
import traceback
from pathlib import Path

# Import optionnel pour HDF5
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py non disponible - fonctionnalités HDF5 désactivées")


# ============== GESTION DES LOGS ==============

def merge_logs(log_files: List[str], output_path: str, 
               format: str = 'csv') -> str:
    """
    Fusionne plusieurs fichiers de logs CSV en un seul.
    
    Args:
        log_files: liste des chemins vers les fichiers CSV
        output_path: chemin de sortie
        format: format de sortie ('csv' ou 'parquet')
    
    Returns:
        str: chemin du fichier fusionné
    """
    print(f"🔄 Fusion de {len(log_files)} fichiers de logs...")
    
    # Charger tous les DataFrames
    dfs = []
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            # Ajouter une colonne avec le nom du fichier source
            df['source_file'] = os.path.basename(log_file)
            dfs.append(df)
            print(f"  ✓ Chargé: {os.path.basename(log_file)} ({len(df)} lignes)")
        except Exception as e:
            print(f"  ✗ Erreur avec {log_file}: {e}")
    
    if not dfs:
        raise ValueError("Aucun fichier de log valide trouvé")
    
    # Fusionner
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Trier par temps si la colonne existe
    if 't' in merged_df.columns:
        merged_df = merged_df.sort_values('t')
    
    # Sauvegarder
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                exist_ok=True)
    
    if format == 'csv':
        merged_df.to_csv(output_path, index=False)
    elif format == 'parquet' and 'pyarrow' in pd.io.parquet.get_engine('auto'):
        merged_df.to_parquet(output_path, index=False)
    else:
        # Fallback sur CSV
        output_path = output_path.replace('.parquet', '.csv')
        merged_df.to_csv(output_path, index=False)
    
    print(f"✅ Fusion terminée: {output_path} ({len(merged_df)} lignes totales)")
    return output_path


def log_seed(seed: int, seed_file: str = "seeds.txt") -> None:
    """
    Enregistre une seed utilisée avec timestamp.
    
    Args:
        seed: valeur de la seed
        seed_file: fichier de log des seeds
    """
    os.makedirs(os.path.dirname(seed_file) if os.path.dirname(seed_file) else ".", 
                exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(seed_file, 'a') as f:
        f.write(f"{timestamp} | SEED = {seed}\n")


def log_config_and_meta(config: Dict, run_id: str, 
                        output_dir: str = "logs") -> None:
    """
    Sauvegarde la configuration et les métadonnées d'un run.
    
    Args:
        config: configuration complète
        run_id: identifiant du run
        output_dir: dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder la config
    config_path = os.path.join(output_dir, f"config_{run_id}.json")
    with open(config_path, 'w') as f:
        json.dump(deep_convert(config), f, indent=2)
    
    # Créer un fichier de métadonnées
    meta_path = os.path.join(output_dir, f"meta_{run_id}.json")
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'fps_version': '1.3',
        'config_path': config_path,
        'system': {
            'N': config.get('system', {}).get('N'),
            'T': config.get('system', {}).get('T'),
            'mode': config.get('system', {}).get('mode'),
            'seed': config.get('system', {}).get('seed')
        }
    }
    
    with open(meta_path, 'w') as f:
        json.dump(deep_convert(metadata), f, indent=2)
    
    print(f"📝 Configuration et métadonnées sauvegardées pour {run_id}")


def log_end_of_run(run_id: str, summary: Optional[Dict] = None,
                   log_file: str = "runs_completed.txt") -> None:
    """
    Enregistre la fin d'un run avec résumé optionnel.
    
    Args:
        run_id: identifiant du run
        summary: résumé des résultats
        log_file: fichier de log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] Run terminé: {run_id}\n")
        
        if summary:
            f.write(f"  Résumé:\n")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    f.write(f"    - {key}: {value:.4f}\n")
                else:
                    f.write(f"    - {key}: {value}\n")


# ============== SAUVEGARDE ET RESTAURATION ==============

def save_simulation_state(state: Dict[str, Any], checkpoint_path: str) -> None:
    """
    Sauvegarde l'état complet de la simulation.
    
    Args:
        state: état du système (strates, historique, etc.)
        checkpoint_path: chemin du checkpoint
    """
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", 
                exist_ok=True)
    
    # Sauvegarder avec pickle (qui gère les types numpy)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # NE PAS créer de version JSON qui cause des erreurs
    # On peut créer juste un fichier d'info minimal
    info_path = checkpoint_path.replace('.pkl', '_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Checkpoint créé : {datetime.now().isoformat()}\n")
        f.write(f"Chemin : {checkpoint_path}\n")
        if 'strates' in state:
            f.write(f"Nombre de strates : {len(state.get('strates', []))}\n")
        if 't' in state:
            f.write(f"Temps actuel : {state.get('t', 0)}\n")
    
    print(f"💾 État sauvegardé: {checkpoint_path}")


def load_simulation_state(checkpoint_path: str) -> Dict[str, Any]:
    """
    Charge un état de simulation sauvegardé.
    
    Args:
        checkpoint_path: chemin du checkpoint
    
    Returns:
        Dict: état restauré
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)
    
    print(f"✅ État restauré depuis: {checkpoint_path}")
    return state


# ============== REPLAY ET ANALYSE ==============

def replay_from_logs(csv_path: str, start_t: float = 0, 
                     end_t: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Rejoue une simulation depuis les logs CSV.
    
    Args:
        csv_path: chemin vers le fichier CSV
        start_t: temps de début
        end_t: temps de fin (None = jusqu'à la fin)
    
    Returns:
        Dict: données rechargées
    """
    print(f"🔄 Replay depuis: {csv_path}")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    
    # Filtrer par temps si nécessaire
    if 't' in df.columns:
        if end_t is not None:
            df = df[(df['t'] >= start_t) & (df['t'] <= end_t)]
        else:
            df = df[df['t'] >= start_t]
    
    # Convertir en dictionnaire de arrays
    data = {}
    for col in df.columns:
        if col != 'effort_status':  # Exclure les colonnes non numériques
            try:
                data[col] = df[col].values.astype(float)
            except:
                # Garder comme string si non numérique
                data[col] = df[col].values
    
    print(f"  ✓ Chargé: {len(df)} pas de temps, {len(data)} métriques")
    return data


def compare_runs(run1_path: str, run2_path: str, 
                 metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compare deux runs sur des métriques spécifiques.
    
    Args:
        run1_path: chemin du premier run
        run2_path: chemin du second run
        metrics: liste des métriques à comparer
    
    Returns:
        Dict: comparaison des métriques
    """
    # Charger les données
    data1 = replay_from_logs(run1_path)
    data2 = replay_from_logs(run2_path)
    
    comparison = {}
    
    for metric in metrics:
        if metric in data1 and metric in data2:
            values1 = data1[metric]
            values2 = data2[metric]
            
            comparison[metric] = {
                'run1_mean': np.mean(values1),
                'run2_mean': np.mean(values2),
                'run1_std': np.std(values1),
                'run2_std': np.std(values2),
                'difference_mean': np.mean(values1) - np.mean(values2),
                'correlation': np.corrcoef(values1[:min(len(values1), len(values2))], 
                                          values2[:min(len(values1), len(values2))])[0, 1]
            }
    
    return deep_convert(comparison)


# ============== EXÉCUTION PARALLÈLE ==============

def run_single_simulation(args: Tuple[str, Dict]) -> Dict[str, Any]:
    """
    Fonction worker pour exécuter une simulation unique.
    
    Args:
        args: tuple (config_path, override_params)
    
    Returns:
        Dict: résultats de la simulation
    """
    config_path, override_params = args
    
    try:
        # Importer les modules nécessaires
        import simulate
        import init
        
        # Charger la config
        config = init.load_config(config_path)
        
        # Appliquer les overrides
        for key, value in override_params.items():
            keys = key.split('.')
            target = config
            for k in keys[:-1]:
                target = target[k]
            target[keys[-1]] = value
        
        # Lancer la simulation
        result = simulate.run_simulation(config_path, config['system'].get('mode', 'FPS'))
        
        return deep_convert({
            'status': 'success',
            'run_id': result['run_id'],
            'metrics': result['metrics']
        })
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def batch_runner(configs_list: List[Union[str, Tuple[str, Dict]]], 
                 parallel: bool = True, n_workers: Optional[int] = None) -> List[Dict]:
    """
    Execute un batch de simulations en parallèle ou séquentiellement.
    
    Args:
        configs_list: liste de configs ou tuples (config_path, overrides)
        parallel: exécution parallèle ou non
        n_workers: nombre de workers (None = nb de CPU)
    
    Returns:
        List[Dict]: résultats de toutes les simulations
    """
    print(f"\n🚀 Lancement batch: {len(configs_list)} simulations")
    
    # Normaliser les inputs
    normalized_configs = []
    for config in configs_list:
        if isinstance(config, str):
            normalized_configs.append((config, {}))
        else:
            normalized_configs.append(config)
    
    results = []
    
    if parallel:
        # Exécution parallèle
        n_workers = n_workers or cpu_count()
        print(f"  Mode parallèle avec {n_workers} workers")
        
        with Pool(n_workers) as pool:
            results = pool.map(run_single_simulation, normalized_configs)
    else:
        # Exécution séquentielle
        print("  Mode séquentiel")
        for config in normalized_configs:
            result = run_single_simulation(config)
            results.append(result)
            
            # Afficher le statut
            if result['status'] == 'success':
                print(f"  ✓ {result['run_id']}")
            else:
                print(f"  ✗ Erreur: {result['error']}")
    
    # Résumé
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n📊 Batch terminé: {success_count}/{len(results)} succès")
    
    return deep_convert(results)


# ============== EXPORT DE DONNÉES ==============

def export_to_hdf5(data_dict: Dict[str, np.ndarray], hdf5_path: str) -> None:
    """
    Exporte des données volumineuses en format HDF5.
    
    Args:
        data_dict: dictionnaire de données à exporter
        hdf5_path: chemin du fichier HDF5
    """
    if not HDF5_AVAILABLE:
        warnings.warn("HDF5 non disponible - export annulé")
        return
    
    os.makedirs(os.path.dirname(hdf5_path) if os.path.dirname(hdf5_path) else ".", 
                exist_ok=True)
    
    with h5py.File(hdf5_path, 'w') as f:
        # Métadonnées
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['fps_version'] = '1.3'
        
        # Données
        for key, data in data_dict.items():
            if isinstance(data, np.ndarray):
                f.create_dataset(key, data=data, compression='gzip')
            elif isinstance(data, (list, tuple)):
                f.create_dataset(key, data=np.array(data), compression='gzip')
            else:
                # Convertir en array si possible
                try:
                    f.create_dataset(key, data=np.array([data]))
                except:
                    # Stocker comme attribut si non convertible
                    f.attrs[key] = str(data)
    
    # Vérifier la taille
    file_size = os.path.getsize(hdf5_path) / (1024 * 1024)  # MB
    print(f"💾 Export HDF5: {hdf5_path} ({file_size:.1f} MB)")


# ============== GÉNÉRATION D'IDENTIFIANTS ==============

def generate_run_id(prefix: str = "run") -> str:
    """
    Génère un identifiant unique pour un run.
    
    Args:
        prefix: préfixe de l'identifiant
    
    Returns:
        str: identifiant unique
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ajouter un hash court pour unicité
    random_bytes = os.urandom(4)
    hash_suffix = hashlib.md5(random_bytes).hexdigest()[:6]
    
    return f"{prefix}_{timestamp}_{hash_suffix}"


# ============== GESTION DES DOSSIERS ==============

def setup_directories(base_dir: str = "fps_output") -> Dict[str, str]:
    """
    Crée la structure de dossiers pour les outputs FPS.
    
    Args:
        base_dir: dossier de base
    
    Returns:
        Dict: chemins créés
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    directories = {
        'base': run_dir,
        'logs': os.path.join(run_dir, 'logs'),
        'checkpoints': os.path.join(run_dir, 'checkpoints'),
        'figures': os.path.join(run_dir, 'figures'),
        'reports': os.path.join(run_dir, 'reports'),
        'configs': os.path.join(run_dir, 'configs')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"📁 Structure créée: {run_dir}")
    return directories


def archive_run(run_dir: str, archive_name: Optional[str] = None) -> str:
    """
    Archive un dossier de run complet.
    
    Args:
        run_dir: dossier à archiver
        archive_name: nom de l'archive (auto-généré si None)
    
    Returns:
        str: chemin de l'archive
    """
    if archive_name is None:
        archive_name = f"{os.path.basename(run_dir)}_archive"
    
    # Créer l'archive
    archive_path = shutil.make_archive(
        archive_name,
        'zip',
        os.path.dirname(run_dir),
        os.path.basename(run_dir)
    )
    
    print(f"📦 Archive créée: {archive_path}")
    return archive_path


# ============== CHECKSUM ET INTÉGRITÉ ==============

def compute_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calcule le checksum d'un fichier pour vérifier l'intégrité.
    
    Args:
        file_path: chemin du fichier
        algorithm: algorithme de hash ('md5', 'sha256', etc.)
    
    Returns:
        str: checksum hexadécimal
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def verify_data_integrity(data_dir: str, checksum_file: str = "checksums.txt") -> bool:
    """
    Vérifie l'intégrité des données d'un dossier.
    
    Args:
        data_dir: dossier contenant les données
        checksum_file: fichier de checksums
    
    Returns:
        bool: True si intégrité vérifiée
    """
    checksum_path = os.path.join(data_dir, checksum_file)
    
    if not os.path.exists(checksum_path):
        print("⚠️  Fichier de checksums non trouvé")
        return False
    
    # Lire les checksums attendus
    expected_checksums = {}
    with open(checksum_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('  ')
                if len(parts) == 2:
                    expected_checksums[parts[1]] = parts[0]
    
    # Vérifier chaque fichier
    all_valid = True
    for filename, expected in expected_checksums.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            actual = compute_checksum(file_path)
            if actual != expected:
                print(f"❌ Checksum invalide: {filename}")
                all_valid = False
            else:
                print(f"✓ {filename}")
        else:
            print(f"❌ Fichier manquant: {filename}")
            all_valid = False
    
    return all_valid


# ============== GESTION DES ERREURS ==============

def handle_crash_recovery(state: Dict[str, Any], loggers: Dict,
                         exception: Exception) -> None:
    """
    Gère la récupération après un crash.
    
    Args:
        state: état du système au moment du crash
        loggers: informations de logging
        exception: exception levée
    """
    crash_dir = "crash_recovery"
    os.makedirs(crash_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_id = f"crash_{timestamp}"
    
    # Fonction helper pour convertir les types numpy
    def convert_numpy_to_python(obj):
        """Convertit récursivement les types numpy en types Python natifs."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(item) for item in obj]
        else:
            return obj
    
    # Sauvegarder l'état avec pickle (qui gère les types numpy)
    state_path = os.path.join(crash_dir, f"{crash_id}_state.pkl")
    try:
        save_simulation_state(state, state_path)
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde état pickle: {e}")
    
    # Convertir l'état pour JSON
    state_for_json = deep_convert(state)
    
    # Sauvegarder les détails du crash
    crash_info = {
        'timestamp': timestamp,
        'run_id': loggers.get('run_id', 'unknown'),
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'traceback': traceback.format_exc(),
        't_current': float(state_for_json.get('t', 0)) if 't' in state_for_json else 'unknown',
        'n_strates': len(state_for_json.get('strates', [])),
        'mode': state_for_json.get('mode', 'unknown')
    }
    
    # Ajouter les métriques si disponibles
    if 'all_metrics' in state_for_json:
        crash_info['last_metrics'] = state_for_json['all_metrics']
    
    info_path = os.path.join(crash_dir, f"{crash_id}_info.json")
    try:
        with open(info_path, 'w') as f:
            json.dump(crash_info, f, indent=2)
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde JSON: {e}")
        # Essayer sans les métriques
        crash_info.pop('last_metrics', None)
        with open(info_path, 'w') as f:
            json.dump(crash_info, f, indent=2)
    
    print(f"\n🚨 Crash recovery:")
    print(f"  État sauvegardé: {state_path}")
    print(f"  Infos crash: {info_path}")
    print(f"  Pour reprendre: load_simulation_state('{state_path}')")


# ============== UTILITAIRES DIVERS ==============

def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en format lisible.
    
    Args:
        seconds: durée en secondes
    
    Returns:
        str: durée formatée (ex: "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs:.1f}s")
    
    return " ".join(parts)


def get_system_info() -> Dict[str, Any]:
    """
    Récupère des informations sur le système.
    
    Returns:
        Dict: informations système
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage_percent': psutil.disk_usage('/').percent
    }
    
    return info

def deep_convert(obj):
    """
    Convertit récursivement tous les np.ndarray en list et tous les types numpy en types Python natifs.
    À utiliser avant tout export JSON, logging de batchs ou rapport final.
    """
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert(x) for x in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def deep_convert_for_json(obj):
    """
    Convertit récursivement un objet Python pour le rendre sérialisable en JSON.
    Gère notamment les clés tuples en les convertissant en strings.
    
    Args:
        obj: objet à convertir
        
    Returns:
        objet converti compatible JSON
    """
    if isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            # Convertir les clés tuples en strings
            if isinstance(key, tuple):
                key_str = f"({','.join(str(k) for k in key)})"
                converted[key_str] = deep_convert_for_json(value)
            else:
                converted[str(key)] = deep_convert_for_json(value)
        return converted
    elif isinstance(obj, (list, tuple)):
        return [deep_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        # Objets custom
        return deep_convert_for_json(obj.__dict__)
    else:
        return obj

# ============== TOPOLOGIES / COUPLAGE ==============


def generate_spiral_weights(N: int,
                            c: float = 0.25,
                            closed: bool = False,
                            mirror: bool = False) -> List[List[float]]:
    """Generate an antisymmetric weight matrix producing a spiral-like coupling.

    Each strate i influences the next strate i+1 with +c, while the next strate
    feeds back −c on i (antisymmetry → Σ w[i] = 0 for every row).  If *closed*
    is True the last strate N-1 is connected back to 0 (ring); otherwise the
    extremities remain open, giving a genuine spiral.

    Parameters
    ----------
    N : int
        Number of strates.
    c : float, optional
        Coupling coefficient (>0).  Typical range 0.05 – 0.30.
    closed : bool, optional
        Whether to close the spiral into a ring (True) or keep it open (False).
    mirror : bool, optional
        Whether to conserve sum by adjusting edge weights.

    Returns
    -------
    List[List[float]]
        Weight matrix W where W[i][j] is the influence of j on i.
    """
    import numpy as _np  # local import avoids polluting public namespace

    if N <= 1:
        return [[0.0]]  # trivial case

    W = _np.zeros((N, N))

    # Forward couplings
    for i in range(N - 1):
        W[i, i + 1] = +c
        W[i + 1, i] = -c

    # Optionally close the ring
    if closed and N > 2:
        W[N - 1, 0] = +c
        W[0, N - 1] = -c
    elif mirror and N > 2:
        # Reflect the excess at the edges onto the adjacent inner nodes
        # so that every row sums to zero while keeping the open spiral direction.
        W[0, 1] -= c     # first row sum becomes zero
        W[N-1, N-2] += c # last row sum becomes zero

    # Convert to plain Python lists (to be JSON-serialisable)
    return W.tolist()

# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module utils.py
    """
    print("=== Tests du module utils.py ===\n")
    
    # Test 1: Génération d'ID
    print("Test 1 - Génération d'identifiants:")
    for i in range(3):
        run_id = generate_run_id()
        print(f"  ID {i+1}: {run_id}")
    
    # Test 2: Gestion des dossiers
    print("\nTest 2 - Structure de dossiers:")
    dirs = setup_directories("test_fps_output")
    for key, path in dirs.items():
        print(f"  {key}: {path}")
    
    # Test 3: Sauvegarde de seed
    print("\nTest 3 - Log de seed:")
    test_seed = 42
    log_seed(test_seed, os.path.join(dirs['logs'], "seeds.txt"))
    print(f"  Seed {test_seed} loguée")
    
    # Test 4: Sauvegarde d'état
    print("\nTest 4 - Sauvegarde/restauration d'état:")
    test_state = {
        't': 50.0,
        'strates': [{'id': 0, 'An': 1.0}, {'id': 1, 'An': 0.8}],
        'history': [{'t': 0, 'S': 0}, {'t': 1, 'S': 0.5}]
    }
    
    checkpoint_path = os.path.join(dirs['checkpoints'], "test_checkpoint.pkl")
    save_simulation_state(test_state, checkpoint_path)
    
    restored_state = load_simulation_state(checkpoint_path)
    print(f"  État restauré: t={restored_state['t']}, n_strates={len(restored_state['strates'])}")
    
    # Test 5: Configuration et métadonnées
    print("\nTest 5 - Log de configuration:")
    test_config = {
        'system': {'N': 3, 'T': 100, 'mode': 'FPS', 'seed': 42},
        'strates': [{'A0': 1.0, 'f0': 1.0}] * 3
    }
    log_config_and_meta(test_config, "test_run", dirs['configs'])
    
    # Test 6: Checksum
    print("\nTest 6 - Checksum:")
    test_file = checkpoint_path
    checksum = compute_checksum(test_file)
    print(f"  SHA256: {checksum[:32]}...")
    
    # Test 7: Formatage de durée
    print("\nTest 7 - Formatage de durée:")
    durations = [45.3, 125.7, 3665.2, 7200.0]
    for d in durations:
        print(f"  {d}s → {format_duration(d)}")
    
    # Test 8: Informations système
    print("\nTest 8 - Informations système:")
    try:
        sys_info = get_system_info()
        print(f"  Python: {sys_info['python_version']}")
        print(f"  CPUs: {sys_info['cpu_count']}")
        print(f"  RAM: {sys_info['memory_available_gb']:.1f}/{sys_info['memory_total_gb']:.1f} GB")
    except:
        print("  (psutil non disponible)")
    
    # Test 9: Archive
    print("\nTest 9 - Archivage:")
    archive_path = archive_run(dirs['base'])
    print(f"  Archive créée: {archive_path}")
    
    # Nettoyage
    shutil.rmtree("test_fps_output", ignore_errors=True)
    if os.path.exists(archive_path):
        os.remove(archive_path)
    
    print("\n✅ Module utils.py prêt à orchestrer la symphonie FPS")


# ============== FONCTIONS ADAPTATIVES ==============

def save_coupled_discoveries(gamma_journal: Dict, regulation_state: Dict, 
                           output_path: str) -> None:
    """
    Sauvegarde les découvertes couplées (γ, G) dans un fichier JSON.
    Si le fichier est trop gros (>15MB), le divise en plusieurs parties.
    
    Args:
        gamma_journal: journal des découvertes gamma
        regulation_state: état de la régulation G
        output_path: chemin de sortie (sera adapté si division nécessaire)
    """
    import os
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Préparer les données
    discoveries = {
        'timestamp': datetime.now().isoformat(),
        'gamma_discoveries': gamma_journal,
        'G_discoveries': regulation_state
    }
    
    # Convertir pour JSON (tuples -> strings, etc.)
    discoveries_serializable = deep_convert_for_json(discoveries)
    
    # Convertir en JSON pour vérifier la taille
    json_str = json.dumps(discoveries_serializable, indent=2, ensure_ascii=False)
    size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
    
    # Si petit fichier, sauvegarder normalement
    if size_mb < 15:  # Limite à 15MB pour garder une marge
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return
    
    # Si gros fichier, créer un dossier et diviser
    base_path = Path(output_path)
    folder_name = base_path.stem + "_parts"
    folder_path = base_path.parent / folder_name
    folder_path.mkdir(exist_ok=True)
    
    # Diviser les découvertes en chunks
    # Stratégie : diviser par états couplés
    gamma_states = discoveries_serializable['gamma_discoveries'].get('coupled_states', {})
    
    if gamma_states:
        # Calculer combien d'états par chunk pour rester sous 15MB
        total_states = len(gamma_states)
        estimated_states_per_chunk = max(1, int(total_states * 15 / size_mb))
        
        # Diviser les états
        states_items = list(gamma_states.items())
        chunk_num = 0
        
        for i in range(0, total_states, estimated_states_per_chunk):
            chunk_states = dict(states_items[i:i + estimated_states_per_chunk])
            
            # Créer un chunk avec métadonnées
            chunk_data = {
                'timestamp': discoveries_serializable['timestamp'],
                'chunk_info': {
                    'part': chunk_num + 1,
                    'total_parts': (total_states + estimated_states_per_chunk - 1) // estimated_states_per_chunk,
                    'states_in_chunk': len(chunk_states),
                    'total_states': total_states
                },
                'gamma_discoveries': {
                    **{k: v for k, v in discoveries_serializable['gamma_discoveries'].items() if k != 'coupled_states'},
                    'coupled_states': chunk_states
                }
            }
            
            # Ajouter les découvertes G seulement dans le premier chunk
            if chunk_num == 0:
                chunk_data['G_discoveries'] = discoveries_serializable['G_discoveries']
            
            # Sauvegarder le chunk
            chunk_path = folder_path / f"part_{chunk_num:03d}.json"
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            chunk_num += 1
        
        # Créer un fichier index
        index_data = {
            'timestamp': discoveries_serializable['timestamp'],
            'total_parts': chunk_num,
            'total_states': total_states,
            'folder': str(folder_path),
            'original_size_mb': round(size_mb, 2),
            'parts': [f"part_{i:03d}.json" for i in range(chunk_num)]
        }
        
        index_path = folder_path / "index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"  📂 Découvertes divisées en {chunk_num} parties dans : {folder_path}")
        print(f"     Taille originale : {size_mb:.1f}MB → ~{15:.1f}MB par partie")
    
    else:
        # Si pas de states à diviser, sauvegarder tel quel avec warning
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"  ⚠️  Fichier volumineux ({size_mb:.1f}MB) sauvegardé sans division : {output_path}")


def select_representative_strata(N, config=None, n_strata_to_show=None):
    """
    Sélectionne des strates représentatives pour visualisation.
    
    Args:
        N: Nombre total de strates
        config: Configuration (optionnel)
        n_strata_to_show: Nombre exact de strates à montrer (override config)
    
    Returns:
        list: Indices des strates sélectionnées, répartis uniformément
    """
    # Déterminer combien de strates à montrer
    if n_strata_to_show is not None:
        # Override explicite
        n_show = n_strata_to_show
    elif config and 'visualization' in config:
        # Depuis config (pourcentage ou nombre absolu)
        viz_config = config['visualization']
        
        if 'strata_sample_percent' in viz_config:
            # Pourcentage (ex: 0.2 = 20%)
            percent = viz_config['strata_sample_percent']
            n_show = max(1, int(N * percent))
        elif 'strata_sample_count' in viz_config:
            # Nombre absolu
            n_show = viz_config['strata_sample_count']
        else:
            # Défaut : 10% avec min 5, max 10
            n_show = max(5, min(10, N // 10))
    else:
        # Défaut : adaptatif selon N
        if N <= 10:
            n_show = N  # Tout montrer
        elif N <= 50:
            n_show = 5  # 10%
        else:
            n_show = max(5, min(10, N // 10))
    
    # Limiter entre 1 et N
    n_show = max(1, min(n_show, N))
    
    # Sélectionner les indices uniformément répartis
    if n_show == 1:
        indices = [0]
    elif n_show == 2:
        indices = [0, N-1]
    elif n_show >= N:
        indices = list(range(N))
    else:
        # Répartition uniforme : toujours inclure début et fin
        indices = [0]  # Toujours la première
        
        # Strates intermédiaires espacées uniformément
        step = (N - 1) / (n_show - 1)
        for i in range(1, n_show - 1):
            idx = int(round(i * step))
            indices.append(idx)
        
        indices.append(N - 1)  # Toujours la dernière
    
    return indices
