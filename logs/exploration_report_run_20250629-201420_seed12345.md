# Rapport d'exploration FPS

**Run ID :** run_20250629-201420_seed12345
**Date :** 2025-06-29 20:14:21
**Total événements :** 145

## Résumé par type d'événement

- **anomaly** : 101 événements
- **harmonic_emergence** : 36 événements
- **fractal_pattern** : 8 événements

## Anomaly

### 1. t=76-125
- **Métrique :** mean_abs_error
- **Valeur :** 17951.5505
- **Sévérité :** high

### 2. t=77-126
- **Métrique :** mean_abs_error
- **Valeur :** 16940.3243
- **Sévérité :** high

### 3. t=75-124
- **Métrique :** mean_abs_error
- **Valeur :** 16735.5138
- **Sévérité :** high

### 4. t=78-127
- **Métrique :** mean_abs_error
- **Valeur :** 13259.0405
- **Sévérité :** high

### 5. t=79-128
- **Métrique :** mean_abs_error
- **Valeur :** 12049.4289
- **Sévérité :** high

## Harmonic Emergence

### 1. t=310-403
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=280-373
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=320-413
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8966
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=100-200
- **Métrique :** effort(t)
- **Valeur :** 0.7551
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.7548
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.7385
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7246
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 8

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.652
- Corrélation max : 0.652

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.747
- Corrélation max : 0.755

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.764
- Corrélation max : 0.897

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.671
- Corrélation max : 0.671

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
