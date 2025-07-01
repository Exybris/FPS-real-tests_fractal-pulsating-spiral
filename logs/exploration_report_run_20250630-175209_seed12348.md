# Rapport d'exploration FPS

**Run ID :** run_20250630-175209_seed12348
**Date :** 2025-06-30 17:52:09
**Total événements :** 187

## Résumé par type d'événement

- **anomaly** : 119 événements
- **harmonic_emergence** : 36 événements
- **phase_cycle** : 21 événements
- **fractal_pattern** : 11 événements

## Anomaly

### 1. t=216-265
- **Métrique :** mean_abs_error
- **Valeur :** 3600.5297
- **Sévérité :** high

### 2. t=217-266
- **Métrique :** mean_abs_error
- **Valeur :** 3232.3825
- **Sévérité :** high

### 3. t=218-267
- **Métrique :** mean_abs_error
- **Valeur :** 2731.4652
- **Sévérité :** high

### 4. t=219-268
- **Métrique :** mean_abs_error
- **Valeur :** 1842.0567
- **Sévérité :** high

### 5. t=220-269
- **Métrique :** mean_abs_error
- **Valeur :** 1482.5373
- **Sévérité :** high

## Harmonic Emergence

### 1. t=320-413
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=200-293
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=270-363
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=385-400
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 2. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 3. t=382-391
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 4. t=386-395
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 5. t=346-354
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=100-200
- **Métrique :** mean_abs_error
- **Valeur :** 0.9146
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9106
- **Sévérité :** high
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8806
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.7829
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.7505
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 11

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.688
- Corrélation max : 0.712

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.683
- Corrélation max : 0.717

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.783
- Corrélation max : 0.783

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.809
- Corrélation max : 0.911

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.915
- Corrélation max : 0.915

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
