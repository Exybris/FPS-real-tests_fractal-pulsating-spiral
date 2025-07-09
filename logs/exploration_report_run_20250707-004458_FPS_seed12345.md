# Rapport d'exploration FPS

**Run ID :** run_20250707-004458_FPS_seed12345
**Date :** 2025-07-07 00:44:59
**Total événements :** 36

## Résumé par type d'événement

- **anomaly** : 29 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=51-54
- **Métrique :** S(t)
- **Valeur :** 85.8752
- **Sévérité :** high

### 2. t=50-99
- **Métrique :** f_mean(t)
- **Valeur :** 51.7142
- **Sévérité :** high

### 3. t=52-54
- **Métrique :** S(t)
- **Valeur :** 16.0112
- **Sévérité :** high

### 4. t=251-300
- **Métrique :** mean_high_effort
- **Valeur :** 15.9455
- **Sévérité :** high

### 5. t=50-99
- **Métrique :** A_mean(t)
- **Valeur :** 13.4365
- **Sévérité :** high

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8946
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8923
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8635
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** entropy_S
- **Valeur :** 0.8456
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.7267
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.846
- Corrélation max : 0.846

### mean_high_effort
- Patterns détectés : 6
- Corrélation moyenne : 0.791
- Corrélation max : 0.895

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
