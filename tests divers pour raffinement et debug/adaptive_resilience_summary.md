# Implémentation de la Résilience Adaptative

## Vue d'ensemble

La résilience adaptative unifie intelligemment deux métriques de résilience selon le contexte :
- **t_retour** : pour les perturbations ponctuelles (choc)
- **continuous_resilience** : pour les perturbations continues (sinus, bruit, rampe)

## Modifications Apportées

### 1. Nouvelle Fonction dans `metrics.py`

```python
def compute_adaptive_resilience(config: Dict, metrics: Dict, 
                               C_history: List[float] = None, 
                               S_history: List[float] = None,
                               t_choc: int = None, dt: float = 0.05) -> Dict[str, Any]
```

**Caractéristiques** :
- Détecte automatiquement le type de perturbation depuis la config
- Sélectionne la métrique appropriée (t_retour ou continuous_resilience)
- Retourne un dictionnaire unifié avec :
  - `type` : 'punctual', 'continuous' ou 'none'
  - `value` : valeur normalisée [0, 1]
  - `score` : score empirique [1-5]
  - `metric_used` : métrique utilisée
  - `raw_value` : valeur brute originale

### 2. Intégration dans `simulate.py`

- Calcul de `adaptive_resilience` à chaque pas de temps
- Ajout dans `all_metrics` et l'historique
- Stockage du score adaptatif pour la grille empirique
- Ajout dans `metrics_summary` avec valeur et score

### 3. Mise à jour de `main.py`

- La grille empirique utilise maintenant `adaptive_resilience_score` directement
- Fallback intelligent sur l'ancienne logique si non disponible
- Mapping des termes mis à jour : `'adaptive_resilience'` au lieu de `'t_retour'`

### 4. Amélioration de `compare_modes.py`

- Utilise `adaptive_resilience` pour FPS si disponible
- Fallback sur t_retour pour compatibilité
- Kuramoto continue d'utiliser t_retour (pas de résilience adaptative)

### 5. Visualisation dans `visualize.py`

- `plot_adaptive_resilience()` priorise la métrique unifiée
- Affichage intelligent selon le type de perturbation détecté
- Normalisation automatique de t_retour en score [0, 1]
- Interprétation visuelle avec seuils colorés

### 6. Validation et Tests

- Ajout de `adaptive_resilience` dans `METRIQUES_VALIDES`
- Test unitaire `test_compute_adaptive_resilience()` ajouté
- Vérification des scores selon les seuils définis

## Avantages de l'Approche

1. **Unification** : Une seule métrique à surveiller au lieu de deux
2. **Intelligence** : Sélection automatique selon le contexte
3. **Comparabilité** : Scores normalisés [0, 1] et [1-5]
4. **Rétrocompatibilité** : Fallback sur les anciennes métriques si besoin
5. **Extensibilité** : Facile d'ajouter de nouveaux types de perturbations

## Seuils de Score

### Perturbations Continues (continuous_resilience)
- Score 5 : ≥ 0.90 (Excellence)
- Score 4 : ≥ 0.75 (Très bon)
- Score 3 : ≥ 0.60 (Bon)
- Score 2 : ≥ 0.40 (Acceptable)
- Score 1 : < 0.40 (Faible)

### Perturbations Ponctuelles (t_retour)
- Score 5 : < 1.0s (Très rapide)
- Score 4 : < 2.0s (Rapide)
- Score 3 : < 5.0s (Modéré)
- Score 2 : < 10.0s (Lent)
- Score 1 : ≥ 10.0s (Très lent)

## Utilisation

La métrique est calculée automatiquement dans le pipeline. Pour l'utiliser directement :

```python
import metrics

# Avec perturbation continue
config = {'system': {'input': {'perturbations': [{'type': 'sinus'}]}}}
current_metrics = {'continuous_resilience': 0.85}
result = metrics.compute_adaptive_resilience(config, current_metrics)
print(f"Score de résilience: {result['score']}/5")

# Avec perturbation ponctuelle
config = {'system': {'input': {'perturbations': [{'type': 'choc'}]}}}
current_metrics = {'t_retour': 1.5}
result = metrics.compute_adaptive_resilience(config, current_metrics)
print(f"Métrique utilisée: {result['metric_used']}")
```

## Impact sur les Rapports

- Les rapports HTML affichent maintenant "Résilience Adaptative"
- Les graphiques s'adaptent automatiquement au type de perturbation
- Les comparaisons entre modes restent cohérentes grâce à la normalisation
- La grille empirique reflète précisément la performance réelle

Cette implémentation offre une vision unifiée et intelligente de la résilience du système FPS, simplifiant l'analyse tout en conservant la précision nécessaire pour chaque type de perturbation. 