# Rapport d'exploration FPS

**Run ID :** run_20250630-175310_seed12348
**Date :** 2025-06-30 17:53:10
**Total événements :** 101

## Résumé par type d'événement

- **anomaly** : 56 événements
- **harmonic_emergence** : 33 événements
- **phase_cycle** : 4 événements
- **fractal_pattern** : 8 événements

## Anomaly

### 1. t=297-346
- **Métrique :** A_mean(t)
- **Valeur :** 42.8197
- **Sévérité :** high

### 2. t=298-347
- **Métrique :** A_mean(t)
- **Valeur :** 39.8689
- **Sévérité :** high

### 3. t=299-348
- **Métrique :** A_mean(t)
- **Valeur :** 35.7150
- **Sévérité :** high

### 4. t=300-349
- **Métrique :** A_mean(t)
- **Valeur :** 28.5034
- **Sévérité :** high

### 5. t=301-350
- **Métrique :** A_mean(t)
- **Valeur :** 23.1499
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=320-413
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=340-433
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-13
- **Métrique :** S(t)
- **Valeur :** 12.0000
- **Sévérité :** medium

### 2. t=2-8
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=54-59
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_abs_error
- **Valeur :** 0.9119
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_abs_error
- **Valeur :** 0.9095
- **Sévérité :** high
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6840
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.6772
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** mean_abs_error
- **Valeur :** 0.6723
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 8

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.662
- Corrélation max : 0.662

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.672
- Corrélation max : 0.684

### mean_abs_error
- Patterns détectés : 3
- Corrélation moyenne : 0.831
- Corrélation max : 0.912

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
