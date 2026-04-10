# Rapport d'exploration FPS

**Run ID :** run_20260407-174942_NEUTRAL_seed12345
**Date :** 2026-04-07 17:50:03
**Total événements :** 36

## Résumé par type d'événement

- **anomaly** : 4 événements
- **harmonic_emergence** : 32 événements

## Anomaly

### 1. t=9887-9889
- **Métrique :** S(t)
- **Valeur :** 4.9011
- **Sévérité :** medium

### 2. t=7412-7414
- **Métrique :** S(t)
- **Valeur :** 4.9011
- **Sévérité :** medium

### 3. t=4937-4939
- **Métrique :** S(t)
- **Valeur :** 4.9011
- **Sévérité :** medium

### 4. t=2462-2464
- **Métrique :** S(t)
- **Valeur :** 4.9011
- **Sévérité :** medium

## Harmonic Emergence

### 1. t=2470-2563
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=4900-4993
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=7420-7513
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=9850-9943
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=10-103
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
