# Rapport d'exploration FPS

**Run ID :** run_20250609-030905_seed12345
**Date :** 2025-06-09 03:09:06
**Total événements :** 15

## Résumé par type d'événement

- **anomaly** : 10 événements
- **phase_cycle** : 5 événements

## Anomaly

### 1. t=92-94
- **Métrique :** S(t)
- **Valeur :** 4.8235
- **Sévérité :** medium

### 2. t=192-194
- **Métrique :** S(t)
- **Valeur :** 4.8235
- **Sévérité :** medium

### 3. t=292-294
- **Métrique :** S(t)
- **Valeur :** 4.8235
- **Sévérité :** medium

### 4. t=392-394
- **Métrique :** S(t)
- **Valeur :** 4.8235
- **Sévérité :** medium

### 5. t=492-494
- **Métrique :** S(t)
- **Valeur :** 4.8235
- **Sévérité :** medium

## Phase Cycle

### 1. t=40-63
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 2. t=140-163
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 3. t=240-263
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 4. t=340-363
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 5. t=440-463
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

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
