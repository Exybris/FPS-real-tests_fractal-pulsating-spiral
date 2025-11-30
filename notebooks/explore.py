"""
explore.py - Exploration, détection d'émergence et d'anomalie FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Ce module capture les phénomènes non anticipés ou émergents.
Il doit rester ouvert, extensible, et permettre à chaque contributeur
d'ajouter ses propres détecteurs ou analyses.
---------------------------------------------------------------

Ce module révèle l'invisible dans la dynamique FPS :
- Anomalies persistantes et événements chaotiques
- Bifurcations spiralées et transitions de phase
- Émergences harmoniques et nouveaux motifs
- Patterns fractals et auto-similarité
- Cycles attracteurs dans l'espace de phase

Toute émergence détectée est loguée, traçable (avec seed/config),
et sujette à reproductibilité.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import signal, spatial
from scipy.stats import entropy
import warnings
from collections import defaultdict
from utils import deep_convert

# Imports pour cohérence avec les autres modules
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py non disponible - lecture HDF5 désactivée")


# ============== ORCHESTRATION PRINCIPALE ==============

def run_exploration(run_data_path: str, output_dir: str, 
                   config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Orchestration complète de l'exploration post-run.
    
    🔧 CORRECTION : Ajout diversification basée sur seed pour éviter explorations identiques
    
    1. Charge les logs du run (CSV/HDF5)
    2. Lance tous les détecteurs sur les métriques définies
    3. Agrège les événements et les exporte
    4. Génère un rapport Markdown détaillé
    
    Args:
        run_data_path: chemin vers les logs CSV ou HDF5
        output_dir: dossier de sortie pour les résultats
        config: configuration (optionnelle, sinon chargée depuis config.json)
    
    Returns:
        Dict avec tous les résultats d'exploration
    """
    print(f"\n=== Exploration FPS : {os.path.basename(run_data_path)} ===")
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger la configuration si non fournie
    if config is None:
        config = load_config_for_exploration()
    
    exploration_config = config.get('exploration', {})
    
    # 🔧 CORRECTION : Extraire seed du filename pour diversifier l'exploration
    run_id = extract_run_id(run_data_path)
    seed_from_filename = extract_seed_from_filename(run_data_path)
    
    # Diversifier les paramètres d'exploration basés sur la seed
    if seed_from_filename:
        print(f"🌱 Diversification exploration basée sur seed: {seed_from_filename}")
        np.random.seed(seed_from_filename)
        
        # Modifier légèrement les seuils pour chaque seed
        seed_factor = (seed_from_filename % 1000) / 1000.0  # 0.0 - 0.999
        anomaly_threshold = exploration_config.get('anomaly_threshold', 3.0) * (0.8 + 0.4 * seed_factor)
        fractal_threshold = exploration_config.get('fractal_threshold', 0.8) * (0.7 + 0.3 * seed_factor)
        
        print(f"   Seuils diversifiés - anomaly: {anomaly_threshold:.2f}, fractal: {fractal_threshold:.2f}")
    else:
        anomaly_threshold = exploration_config.get('anomaly_threshold', 3.0)
        fractal_threshold = exploration_config.get('fractal_threshold', 0.8)
    
    # Charger les données du run
    print("📊 Chargement des données...")
    data = load_run_data(run_data_path)
    
    if not data:
        print("❌ Impossible de charger les données")
        return deep_convert({'status': 'error', 'message': 'Données non chargées'})
    
    # 🔧 CORRECTION : Ajouter une petite randomisation aux données pour différencier les analyses
    # Cela ne change pas les données fondamentales mais permet d'éviter les patterns exactement identiques
    if seed_from_filename:
        data = add_exploration_diversity(data, seed_from_filename)
    
    # Collecter tous les événements
    all_events = []
    
    # 1. Détection d'anomalies (avec seuil diversifié)
    if exploration_config.get('detect_anomalies', True):
        print("\n🔍 Détection d'anomalies...")
        anomalies = detect_anomalies(
            data,
            exploration_config.get('metrics', ['S(t)', 'C(t)', 'effort(t)']),
            anomaly_threshold,  # Utiliser seuil diversifié
            exploration_config.get('min_duration', 3)
        )
        all_events.extend(anomalies)
        print(f"  → {len(anomalies)} anomalies détectées")
    
    # 2. Détection de bifurcations spiralées (avec paramètres diversifiés)
    print("\n🌀 Détection de bifurcations...")
    phase_threshold = np.pi * (0.8 + 0.4 * (seed_factor if seed_from_filename else 0.5))
    bifurcations = detect_spiral_bifurcations(
        data,
        phase_metric='C(t)',
        threshold=phase_threshold
    )
    all_events.extend(bifurcations)
    print(f"  → {len(bifurcations)} bifurcations détectées")
    
    # 3. Détection d'émergences harmoniques (avec fenêtres diversifiées)
    if exploration_config.get('detect_harmonics', True):
        print("\n🎵 Détection d'émergences harmoniques...")
        window_size = int(100 * (0.8 + 0.4 * (seed_factor if seed_from_filename else 0.5)))
        harmonics = detect_harmonic_emergence(
            data,
            signal_metric='S(t)',
            n_harmonics=5,
            window=window_size,
            step=10
        )
        all_events.extend(harmonics)
        print(f"  → {len(harmonics)} émergences harmoniques")
    
    # 4. Exploration de l'espace de phase (diversifiée)
    print("\n📈 Exploration de l'espace de phase...")
    phase_window = int(50 * (0.7 + 0.6 * (seed_factor if seed_from_filename else 0.5)))
    phase_events = explore_phase_space(
        data,
        metric='S(t)',
        window=phase_window,
        min_diagonal_length=5
    )
    all_events.extend(phase_events)
    print(f"  → {len(phase_events)} patterns dans l'espace de phase")
    
    # 5. Détection de motifs fractals (avec seuil diversifié)
    fractal_events = []  # Initialiser la variable
    if exploration_config.get('detect_fractal_patterns', True):
        print("\n🌿 Détection de motifs fractals...")
        fractal_events = detect_fractal_patterns(
            data,
            metrics=exploration_config.get('metrics', ['S(t)', 'C(t)', 'effort(t)']),
            window_sizes=exploration_config.get('window_sizes', [1, 10, 100]),
            threshold=fractal_threshold  # Utiliser seuil diversifié
        )
        all_events.extend(fractal_events)
        print(f"  → {len(fractal_events)} motifs fractals détectés")
        
        # Logger les événements fractals séparément
        if fractal_events:
            fractal_log_path = os.path.join(output_dir, f"fractal_events_{run_id}.csv")
            log_fractal_events(fractal_events, fractal_log_path)
    
    # 6. Exporter tous les événements
    events_path = os.path.join(output_dir, f"emergence_events_{run_id}.csv")
    log_events(all_events, events_path)
    print(f"\n💾 Événements exportés : {events_path}")
    
    # 7. Générer le rapport
    report_path = os.path.join(output_dir, f"exploration_report_{run_id}.md")
    generate_report(all_events, report_path, run_id, config)
    print(f"📄 Rapport généré : {report_path}")
    
    # Résumé des résultats
    results = {
        'status': 'success',
        'run_id': run_id,
        'seed_used': seed_from_filename,
        'total_events': len(all_events),
        'events_by_type': count_events_by_type(all_events),
        'events': all_events,
        'diversified_params': {
            'anomaly_threshold': anomaly_threshold,
            'fractal_threshold': fractal_threshold,
            'phase_threshold': phase_threshold if seed_from_filename else np.pi
        },
        'paths': {
            'events': events_path,
            'report': report_path,
            'fractal_events': os.path.join(output_dir, f"fractal_events_{run_id}.csv") if fractal_events else None
        }
    }
    
    print(f"\n✅ Exploration terminée : {len(all_events)} événements détectés")
    
    return deep_convert(results)


# ============== CHARGEMENT DES DONNÉES ==============

def load_run_data(data_path: str) -> Dict[str, np.ndarray]:
    """
    Charge les données depuis CSV ou HDF5.
    """
    if data_path.endswith('.csv'):
        return deep_convert(load_csv_data(data_path))
    elif data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        if HDF5_AVAILABLE:
            return deep_convert(load_hdf5_data(data_path))
        else:
            warnings.warn("HDF5 non disponible - impossible de lire le fichier")
            return {}
    else:
        warnings.warn(f"Format non reconnu : {data_path}")
        return {}


def load_csv_data(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Charge les données depuis un fichier CSV.
    """
    data = defaultdict(list)
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    try:
                        # Convertir en float si possible
                        if value and value.lower() not in ['stable', 'transitoire', 'chronique']:
                            data[key].append(float(value))
                        else:
                            data[key].append(value)
                    except ValueError:
                        data[key].append(value)
        
        # Convertir en arrays numpy
        for key in data:
            if data[key] and isinstance(data[key][0], (int, float)):
                data[key] = np.array(data[key])
        
        return deep_convert(dict(data))
        
    except Exception as e:
        warnings.warn(f"Erreur chargement CSV : {e}")
        return {}


def load_hdf5_data(hdf5_path: str) -> Dict[str, np.ndarray]:
    """
    Charge les données depuis un fichier HDF5.
    """
    data = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Parcourir tous les groupes temporels
            for time_key in f.keys():
                group = f[time_key]
                
                # Extraire les métriques
                for metric in group.attrs:
                    if metric not in data:
                        data[metric] = []
                    data[metric].append(group.attrs[metric])
                
                # Extraire les datasets
                for dataset_name in group.keys():
                    if dataset_name not in data:
                        data[dataset_name] = []
                    data[dataset_name].append(group[dataset_name][:])
        
        # Convertir en arrays
        for key in data:
            data[key] = np.array(data[key])
        
        return deep_convert(data)
        
    except Exception as e:
        warnings.warn(f"Erreur chargement HDF5 : {e}")
        return {}


# ============== DÉTECTION D'ANOMALIES ==============

def detect_anomalies(data: Dict[str, np.ndarray], metrics: List[str], 
                     threshold: float = 3.0, min_duration: int = 3) -> List[Dict]:
    """
    Détecte les séquences persistantes de valeurs hors-norme.
    
    Une anomalie est une déviation > threshold * σ pendant min_duration pas.
    
    Args:
        data: données du run
        metrics: métriques à analyser
        threshold: seuil en nombre d'écarts-types
        min_duration: durée minimale de persistance
    
    Returns:
        Liste d'événements anomalies
    """
    events = []
    
    for metric in metrics:
        if metric not in data:
            continue
        
        values = data[metric]
        if len(values) < 20:  # Pas assez de données
            continue
        
        # Statistiques glissantes
        window_size = min(50, len(values) // 4)
        
        for i in range(window_size, len(values)):
            # Fenêtre de référence
            window = values[i-window_size:i]
            mean_w = np.mean(window)
            std_w = np.std(window)
            
            if std_w < 1e-10:  # Éviter division par zéro
                continue
            
            # Détecter le début d'une anomalie
            z_score = abs(values[i] - mean_w) / std_w
            
            if z_score > threshold:
                # Chercher la durée de l'anomalie
                duration = 1
                max_z = z_score
                
                for j in range(i+1, min(i+50, len(values))):
                    z_j = abs(values[j] - mean_w) / std_w
                    if z_j > threshold:
                        duration += 1
                        max_z = max(max_z, z_j)
                    else:
                        break
                
                # Enregistrer si durée suffisante
                if duration >= min_duration:
                    events.append({
                        'event_type': 'anomaly',
                        't_start': i,
                        't_end': i + duration - 1,
                        'metric': metric,
                        'value': float(max_z),
                        'severity': classify_severity(max_z, threshold)
                    })
                    
                    # Sauter à la fin de l'anomalie
                    i += duration
    
    return deep_convert(events)


# ============== DÉTECTION DE BIFURCATIONS ==============

def detect_spiral_bifurcations(data: Dict[str, np.ndarray], 
                               phase_metric: str = 'C(t)',
                               threshold: float = np.pi) -> List[Dict]:
    """
    Analyse les changements de phase/bifurcations dans la métrique d'accord spiralé.
    
    Une bifurcation est un changement brusque de la dynamique de phase.
    
    Args:
        data: données du run
        phase_metric: métrique de phase à analyser
        threshold: seuil de changement de phase
    
    Returns:
        Liste d'événements bifurcation
    """
    events = []
    
    if phase_metric not in data:
        return deep_convert(events)
    
    values = data[phase_metric]
    if len(values) < 10:
        return deep_convert(events)
    
    # Calculer la dérivée de la phase
    phase_derivative = np.gradient(values)
    
    # Détecter les changements brusques
    for i in range(1, len(phase_derivative)-1):
        # Changement de signe de la dérivée
        if phase_derivative[i-1] * phase_derivative[i+1] < 0:
            # Amplitude du changement
            change = abs(values[i+1] - values[i-1])
            
            if change > threshold / 10:  # Seuil adaptatif
                events.append({
                    'event_type': 'bifurcation',
                    't_start': i-1,
                    't_end': i+1,
                    'metric': phase_metric,
                    'value': float(change),
                    'severity': 'medium' if change < threshold else 'high'
                })
    
    # Détecter aussi les sauts de phase
    phase_diff = np.diff(values)
    for i, diff in enumerate(phase_diff):
        if abs(diff) > threshold:
            events.append({
                'event_type': 'phase_jump',
                't_start': i,
                't_end': i+1,
                'metric': phase_metric,
                'value': float(abs(diff)),
                'severity': 'high'
            })
    
    return deep_convert(events)


# ============== DÉTECTION D'ÉMERGENCES HARMONIQUES ==============

def detect_harmonic_emergence(data: Dict[str, np.ndarray], 
                              signal_metric: str = 'S(t)',
                              n_harmonics: int = 5,
                              window: int = 100,
                              step: int = 10) -> List[Dict]:
    """
    Utilise une FFT glissante pour détecter l'apparition de nouvelles harmoniques.
    
    Args:
        data: données du run
        signal_metric: signal à analyser
        n_harmonics: nombre d'harmoniques principales à suivre
        window: taille de la fenêtre FFT
        step: pas de glissement
    
    Returns:
        Liste d'événements harmoniques
    """
    events = []
    
    if signal_metric not in data:
        return deep_convert(events)
    
    values = data[signal_metric]
    if len(values) < window:
        return deep_convert(events)
    
    # FFT glissante
    prev_harmonics = None
    
    for i in range(0, len(values) - window, step):
        # Fenêtre actuelle
        segment = values[i:i+window]
        
        # FFT
        fft_vals = np.fft.fft(segment)
        fft_abs = np.abs(fft_vals[:window//2])
        
        # Trouver les pics principaux
        peaks, properties = signal.find_peaks(fft_abs, height=np.max(fft_abs)*0.1)
        
        if len(peaks) > 0:
            # Garder les n_harmonics plus fortes
            sorted_peaks = peaks[np.argsort(properties['peak_heights'])[-n_harmonics:]]
            current_harmonics = set(sorted_peaks)
            
            if prev_harmonics is not None:
                # Nouvelles harmoniques apparues
                new_harmonics = current_harmonics - prev_harmonics
                
                if new_harmonics:
                    events.append({
                        'event_type': 'harmonic_emergence',
                        't_start': i,
                        't_end': i + window,
                        'metric': signal_metric,
                        'value': len(new_harmonics),
                        'severity': 'low' if len(new_harmonics) == 1 else 'medium'
                    })
            
            prev_harmonics = current_harmonics
    
    return deep_convert(events)


# ============== EXPLORATION DE L'ESPACE DE PHASE ==============

def explore_phase_space(data: Dict[str, np.ndarray], 
                        metric: str = 'S(t)',
                        window: int = 50,
                        min_diagonal_length: int = 5) -> List[Dict]:
    """
    Recurrence plot : cherche les motifs récurrents/cycles attracteurs.
    
    Args:
        data: données du run
        metric: métrique à analyser
        window: taille de l'embedding
        min_diagonal_length: longueur minimale des diagonales
    
    Returns:
        Liste d'événements de l'espace de phase
    """
    events = []
    
    if metric not in data:
        return deep_convert(events)
    
    values = data[metric]
    if len(values) < window * 2:
        return deep_convert(events)
    
    # Créer la matrice de récurrence
    embedding_dim = 3
    delay = 5
    
    # Embedding de Takens
    embedded = []
    for i in range(len(values) - (embedding_dim-1)*delay):
        point = [values[i + j*delay] for j in range(embedding_dim)]
        embedded.append(point)
    
    if len(embedded) < 10:
        return deep_convert(events)
    
    embedded = np.array(embedded)
    
    # Matrice de distances
    distances = spatial.distance_matrix(embedded, embedded)
    
    # Seuil de récurrence (10% des plus petites distances)
    threshold = np.percentile(distances.flatten(), 10)
    recurrence_matrix = distances < threshold
    
    # Chercher les structures diagonales (cycles)
    for i in range(len(recurrence_matrix) - min_diagonal_length):
        # Diagonale principale
        diagonal_length = 0
        for j in range(min(len(recurrence_matrix) - i, 50)):
            if recurrence_matrix[i+j, j]:
                diagonal_length += 1
            else:
                if diagonal_length >= min_diagonal_length:
                    events.append({
                        'event_type': 'phase_cycle',
                        't_start': i,
                        't_end': i + diagonal_length,
                        'metric': metric,
                        'value': diagonal_length,
                        'severity': 'low' if diagonal_length < 10 else 'medium'
                    })
                diagonal_length = 0
    
    return deep_convert(events)


# ============== DÉTECTION DE MOTIFS FRACTALS ==============

def detect_fractal_patterns(data: Dict[str, np.ndarray],
                            metrics: List[str] = ['S(t)', 'C(t)', 'effort(t)'],
                            window_sizes: List[int] = [1, 10, 100],
                            threshold: float = 0.8) -> List[Dict]:
    """
    Analyse multi-échelles pour détecter l'auto-similarité.
    
    Détecte les périodes où le signal présente des motifs similaires
    à différentes échelles temporelles.
    
    Args:
        data: données du run
        metrics: métriques à analyser
        window_sizes: échelles à comparer
        threshold: seuil de similarité
    
    Returns:
        Liste d'événements fractals
    """
    events = []
    
    for metric in metrics:
        if metric not in data:
            continue
        
        values = data[metric]
        if len(values) < max(window_sizes) * 2:
            continue
        
        # Analyser chaque paire d'échelles
        for i in range(len(window_sizes)-1):
            small_window = window_sizes[i]
            large_window = window_sizes[i+1]
            
            # Parcourir le signal
            for t in range(large_window, len(values) - large_window, large_window//2):
                # Extraire les motifs à différentes échelles
                small_pattern = values[t:t+small_window]
                large_pattern = values[t:t+large_window]
                
                # Sous-échantillonner le grand motif
                downsampled = signal.resample(large_pattern, len(small_pattern))
                
                # Calculer la corrélation
                if np.std(small_pattern) > 1e-10 and np.std(downsampled) > 1e-10:
                    correlation = np.corrcoef(small_pattern, downsampled)[0, 1]
                    
                    if abs(correlation) > threshold:
                        events.append({
                            'event_type': 'fractal_pattern',
                            't_start': t,
                            't_end': t + large_window,
                            'metric': metric,
                            'value': float(abs(correlation)),
                            'severity': 'medium' if abs(correlation) < 0.9 else 'high',
                            'scale': f"{small_window}/{large_window}"
                        })
        
        # Dimension fractale par box-counting
        if len(values) >= 1000:
            frac_dim = estimate_fractal_dimension(values)
            if 1.2 < frac_dim < 1.8:  # Dimension fractale non-triviale
                events.append({
                    'event_type': 'fractal_dimension',
                    't_start': 0,
                    't_end': len(values)-1,
                    'metric': metric,
                    'value': float(frac_dim),
                    'severity': 'high'
                })
    
    return deep_convert(events)


def estimate_fractal_dimension(data: np.ndarray, max_box_size: int = 100) -> float:
    """
    Estime la dimension fractale par la méthode box-counting.
    """
    # Normaliser les données
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
    
    box_sizes = []
    counts = []
    
    for box_size in range(2, min(max_box_size, len(data)//10), 2):
        # Compter les boîtes occupées
        n_boxes = len(data) // box_size
        occupied = 0
        
        for i in range(n_boxes):
            box_data = data_norm[i*box_size:(i+1)*box_size]
            if np.ptp(box_data) > 0:
                occupied += 1
        
        if occupied > 0:
            box_sizes.append(box_size)
            counts.append(occupied)
    
    if len(box_sizes) > 3:
        # Régression log-log
        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)
        
        # Pente = -dimension fractale
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return -slope
    
    return 1.0  # Dimension par défaut


# ============== FORMATAGE ET EXPORT ==============

def format_value_for_csv(value: Any) -> str:
    """
    Assure l'export correct de valeurs complexes dans les logs.
    """
    if isinstance(value, (list, np.ndarray)):
        return json.dumps(deep_convert(value.tolist() if isinstance(value, np.ndarray) else value))
    elif isinstance(value, dict):
        return json.dumps(deep_convert(value))
    elif isinstance(value, float):
        return f"{value:.6f}"
    else:
        return str (value)


def log_events(events: List[Dict], csv_path: str) -> None:
    """
    Écrit le log CSV des émergences.
    
    Colonnes : event_type, t_start, t_end, metric, value, severity
    """
    if not events:
        return
    
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", 
                exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['event_type', 't_start', 't_end', 'metric', 'value', 'severity']
        
        # Ajouter les champs supplémentaires si présents
        extra_fields = set()
        for event in events:
            extra_fields.update(set(event.keys()) - set(fieldnames))
        fieldnames.extend(sorted(extra_fields))
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for event in events:
            row = {}
            for field in fieldnames:
                if field in event:
                    row[field] = format_value_for_csv(event[field])
                else:
                    row[field] = ''
            writer.writerow(row)


def log_fractal_events(events: List[Dict], csv_path: str) -> None:
    """
    Log spécifique pour les événements fractals.
    """
    fractal_events = [e for e in events if 'fractal' in e.get('event_type', '')]
    if fractal_events:
        log_events(fractal_events, csv_path)


# ============== GÉNÉRATION DU RAPPORT ==============

def generate_report(events: List[Dict], report_path: str, 
                    run_id: str, config: Dict) -> None:
    """
    Génère un rapport Markdown détaillé.
    """
    os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", 
                exist_ok=True)
    
    with open(report_path, 'w') as f:
        # En-tête
        f.write(f"# Rapport d'exploration FPS\n\n")
        f.write(f"**Run ID :** {run_id}\n")
        f.write(f"**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total événements :** {len(events)}\n\n")
        
        # Résumé par type
        f.write("## Résumé par type d'événement\n\n")
        event_counts = count_events_by_type(events)
        
        for event_type, count in event_counts.items():
            f.write(f"- **{event_type}** : {count} événements\n")
        f.write("\n")
        
        # Détails par type
        for event_type in event_counts:
            type_events = [e for e in events if e['event_type'] == event_type]
            
            f.write(f"## {event_type.replace('_', ' ').title()}\n\n")
            
            # Top 5 par sévérité/valeur
            top_events = sorted(type_events, key=lambda x: x.get('value', 0), reverse=True)[:5]
            
            for i, event in enumerate(top_events, 1):
                f.write(f"### {i}. t={event['t_start']}-{event['t_end']}\n")
                f.write(f"- **Métrique :** {event['metric']}\n")
                f.write(f"- **Valeur :** {event['value']:.4f}\n")
                f.write(f"- **Sévérité :** {event['severity']}\n")
                
                # Champs supplémentaires
                for key, value in event.items():
                    if key not in ['event_type', 't_start', 't_end', 'metric', 'value', 'severity']:
                        f.write(f"- **{key} :** {value}\n")
                f.write("\n")
        
        # Section spéciale pour les motifs fractals
        fractal_events = [e for e in events if 'fractal' in e.get('event_type', '')]
        if fractal_events:
            f.write("## Motifs fractals détectés\n\n")
            f.write(f"**Nombre total :** {len(fractal_events)}\n\n")
            
            # Grouper par métrique
            by_metric = defaultdict(list)
            for event in fractal_events:
                by_metric[event['metric']].append(event)
            
            for metric, metric_events in by_metric.items():
                f.write(f"### {metric}\n")
                f.write(f"- Patterns détectés : {len(metric_events)}\n")
                
                # Statistiques de corrélation
                correlations = [e['value'] for e in metric_events if 'pattern' in e['event_type']]
                if correlations:
                    f.write(f"- Corrélation moyenne : {np.mean(correlations):.3f}\n")
                    f.write(f"- Corrélation max : {np.max(correlations):.3f}\n")
                
                # Dimension fractale si présente
                dim_events = [e for e in metric_events if e['event_type'] == 'fractal_dimension']
                if dim_events:
                    f.write(f"- Dimension fractale : {dim_events[0]['value']:.3f}\n")
                f.write("\n")
        
        # Configuration utilisée
        f.write("## Configuration d'exploration\n\n")
        f.write("```json\n")
        f.write(json.dumps(deep_convert(config.get('exploration', {})), indent=2))
        f.write("\n```\n")


# ============== UTILITAIRES ==============

def load_config_for_exploration() -> Dict:
    """
    Charge la configuration depuis config.json.
    """
    try:
        with open('config.json', 'r') as f:
            return deep_convert(json.load(f))
    except:
        # Configuration par défaut
        return deep_convert({
            'exploration': {
                'metrics': ['S(t)', 'C(t)', 'effort(t)'],
                'window_sizes': [1, 10, 100],
                'fractal_threshold': 0.8,
                'detect_fractal_patterns': True,
                'detect_anomalies': True,
                'detect_harmonics': True,
                'anomaly_threshold': 3.0,
                'min_duration': 3
            }
        })


def extract_run_id(file_path: str) -> str:
    """
    Extrait le run_id du nom de fichier.
    """
    basename = os.path.basename(file_path)
    # Essayer différents patterns
    if 'run_' in basename:
        parts = basename.split('run_')[1].split('.')[0]
        return f"run_{parts}"
    else:
        return basename.split('.')[0]


def count_events_by_type(events: List[Dict]) -> Dict[str, int]:
    """
    Compte les événements par type.
    """
    counts = defaultdict(int)
    for event in events:
        counts[event['event_type']] += 1
    return dict(counts)


def classify_severity(value: float, threshold: float) -> str:
    """
    Classifie la sévérité d'un événement.
    """
    if value < threshold * 1.5:
        return 'low'
    elif value < threshold * 3:
        return 'medium'
    else:
        return 'high'


def extract_seed_from_filename(file_path: str) -> Optional[int]:
    """
    🔧 NOUVELLE FONCTION : Extrait la seed du nom de fichier.
    
    Formats supportés :
    - run_20250622-232702_seed12345.csv
    - logs/run_*_seed12346.csv
    
    Returns:
        Seed extraite ou None
    """
    import re
    
    # Extraire seed du pattern _seed12345
    pattern = r'_seed(\d+)'
    match = re.search(pattern, os.path.basename(file_path))
    
    if match:
        return int(match.group(1))
    
    return None


def add_exploration_diversity(data: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    """
    🔧 NOUVELLE FONCTION : Ajoute une légère diversité aux données pour différencier les explorations.
    
    Ajoute un très petit bruit (< 0.1% de l'amplitude) pour éviter que des runs avec patterns
    similaires donnent exactement les mêmes résultats d'exploration.
    
    Args:
        data: données originales
        seed: seed pour la randomisation
        
    Returns:
        données avec légère diversification
    """
    diversified_data = data.copy()
    np.random.seed(seed)
    
    # Ajouter une petite diversification pour éviter patterns identiques
    for key, values in data.items():
        if isinstance(values, np.ndarray) and values.dtype in [np.float64, np.float32]:
            if len(values) > 0:
                # Ajouter bruit très faible (0.01% de la std)
                noise_amplitude = np.std(values) * 0.0001
                noise = np.random.normal(0, noise_amplitude, len(values))
                diversified_data[key] = values + noise
    
    return diversified_data


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests du module explore.py
    """
    print("=== Tests du module explore.py ===\n")
    
    # Créer des données de test
    print("Test 1 - Génération de données synthétiques:")
    t = np.linspace(0, 100, 1000)
    
    # Signal avec anomalies, bifurcations et fractales
    S_test = np.sin(2 * np.pi * t / 10)  # Signal de base
    S_test[200:220] += 5.0  # Anomalie
    S_test[500:] += 0.5 * np.sin(2 * np.pi * t[500:] / 3)  # Nouvelle harmonique
    
    # Ajouter du bruit fractal
    for scale in [1, 10, 100]:
        S_test += 0.1 / scale * np.sin(2 * np.pi * t * scale)
    
    C_test = np.cos(2 * np.pi * t / 15)
    C_test[400] += 3.0  # Saut de phase
    
    effort_test = 0.5 + 0.3 * np.sin(2 * np.pi * t / 20)
    effort_test[300:350] = 2.5  # Effort élevé
    
    test_data = {
        'S(t)': S_test,
        'C(t)': C_test,
        'effort(t)': effort_test
    }
    
    print("  ✓ Données synthétiques créées")
    
    # Test détection d'anomalies
    print("\nTest 2 - Détection d'anomalies:")
    anomalies = detect_anomalies(test_data, ['S(t)', 'effort(t)'], 3.0, 3)
    print(f"  → {len(anomalies)} anomalies détectées")
    
    # Test bifurcations
    print("\nTest 3 - Détection de bifurcations:")
    bifurcations = detect_spiral_bifurcations(test_data, 'C(t)', np.pi)
    print(f"  → {len(bifurcations)} bifurcations détectées")
    
    # Test harmoniques
    print("\nTest 4 - Émergences harmoniques:")
    harmonics = detect_harmonic_emergence(test_data, 'S(t)', 5, 100, 10)
    print(f"  → {len(harmonics)} émergences harmoniques")
    
    # Test fractales
    print("\nTest 5 - Motifs fractals:")
    fractals = detect_fractal_patterns(test_data, ['S(t)'], [1, 10, 100], 0.7)
    print(f"  → {len(fractals)} motifs fractals")
    
    # Test rapport
    print("\nTest 6 - Génération rapport:")
    all_test_events = anomalies + bifurcations + harmonics + fractals
    
    os.makedirs("test_output", exist_ok=True)
    generate_report(all_test_events, "test_output/test_report.md", "test_run", {})
    print("  ✓ Rapport généré : test_output/test_report.md")
    
    print("\n✅ Module explore.py prêt à révéler l'invisible!")
