# Rapport d'exploration FPS

**Run ID :** run_20250709-232852_FPS_seed12349
**Date :** 2025-07-09 23:28:53
**Total événements :** 97

## Résumé par type d'événement

- **anomaly** : 74 événements
- **harmonic_emergence** : 4 événements
- **phase_cycle** : 11 événements
- **fractal_pattern** : 8 événements

## Anomaly

### 1. t=201-250
- **Métrique :** A_mean(t)
- **Valeur :** 21.5918
- **Sévérité :** high

### 2. t=301-350
- **Métrique :** A_mean(t)
- **Valeur :** 21.2568
- **Sévérité :** high

### 3. t=300-349
- **Métrique :** A_mean(t)
- **Valeur :** 20.6260
- **Sévérité :** high

### 4. t=300-315
- **Métrique :** effort(t)
- **Valeur :** 20.3133
- **Sévérité :** high

### 5. t=200-249
- **Métrique :** A_mean(t)
- **Valeur :** 20.2343
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=150-243
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=270-363
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1-17
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 2. t=2-17
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 3. t=8-17
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 4. t=3-11
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=4-12
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7300
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6857
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.6779
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.6747
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.6714
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 8

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.655
- Corrélation max : 0.664

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.730
- Corrélation max : 0.730

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.676
- Corrélation max : 0.686

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "C(t)",
    "A_mean(t)",
    "f_mean(t)",
    "entropy_S",
    "effort(t)",
    "mean_high_effort",
    "d_effort_dt",
    "mean_abs_error"
  ],
  "window_sizes": [
    1,
    10,
    100
  ],
  "fractal_threshold": 0.8,
  "detect_fractal_patterns": true,
  "detect_anomalies": true,
  "detect_harmonics": true,
  "recurrence_window": [
    1,
    10,
    100
  ],
  "anomaly_threshold": 3.0,
  "min_duration": 3
}
```
