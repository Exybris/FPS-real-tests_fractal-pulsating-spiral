# Rapport d'exploration FPS

**Run ID :** run_20250707-192821_FPS_seed12345
**Date :** 2025-07-07 19:28:21
**Total événements :** 4

## Résumé par type d'événement

- **anomaly** : 4 événements

## Anomaly

### 1. t=45-94
- **Métrique :** A_mean(t)
- **Valeur :** 11.3711
- **Sévérité :** high

### 2. t=46-95
- **Métrique :** A_mean(t)
- **Valeur :** 9.6323
- **Sévérité :** high

### 3. t=94-96
- **Métrique :** S(t)
- **Valeur :** 5.5866
- **Sévérité :** medium

### 4. t=67-69
- **Métrique :** entropy_S
- **Valeur :** 3.0847
- **Sévérité :** low

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
