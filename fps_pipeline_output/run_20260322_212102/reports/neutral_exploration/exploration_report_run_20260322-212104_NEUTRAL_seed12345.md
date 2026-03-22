# Rapport d'exploration FPS

**Run ID :** run_20260322-212104_NEUTRAL_seed12345
**Date :** 2026-03-22 21:21:04
**Total événements :** 14

## Résumé par type d'événement

- **anomaly** : 6 événements
- **harmonic_emergence** : 8 événements

## Anomaly

### 1. t=216-219
- **Métrique :** S(t)
- **Valeur :** 5.0114
- **Sévérité :** medium

### 2. t=441-444
- **Métrique :** S(t)
- **Valeur :** 5.0114
- **Sévérité :** medium

### 3. t=437-439
- **Métrique :** S(t)
- **Valeur :** 4.9935
- **Sévérité :** medium

### 4. t=212-214
- **Métrique :** S(t)
- **Valeur :** 4.9935
- **Sévérité :** medium

### 5. t=217-219
- **Métrique :** S(t)
- **Valeur :** 4.6306
- **Sévérité :** medium

## Harmonic Emergence

### 1. t=170-263
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=220-313
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=400-493
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=150-243
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=200-293
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "C(t)",
    "An_mean(t)",
    "fn_mean(t)",
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
  "detect_fractal_patterns": true,
  "detect_anomalies": true,
  "detect_harmonics": true,
  "anomaly_threshold": 3.0,
  "fractal_threshold": 0.8,
  "min_duration": 3,
  "recurrence_window": [
    1,
    10,
    100
  ]
}
```
