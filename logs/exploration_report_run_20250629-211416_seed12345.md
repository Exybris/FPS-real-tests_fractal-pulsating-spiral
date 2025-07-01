# Rapport d'exploration FPS

**Run ID :** run_20250629-211416_seed12345
**Date :** 2025-06-29 21:14:16
**Total événements :** 15

## Résumé par type d'événement

- **anomaly** : 11 événements
- **phase_cycle** : 4 événements

## Anomaly

### 1. t=54-99
- **Métrique :** mean_high_effort
- **Valeur :** 325.6920
- **Sévérité :** high

### 2. t=55-99
- **Métrique :** mean_high_effort
- **Valeur :** 114.4790
- **Sévérité :** high

### 3. t=56-99
- **Métrique :** mean_high_effort
- **Valeur :** 79.5625
- **Sévérité :** high

### 4. t=57-99
- **Métrique :** mean_high_effort
- **Valeur :** 67.0574
- **Sévérité :** high

### 5. t=58-99
- **Métrique :** mean_high_effort
- **Valeur :** 54.1396
- **Sévérité :** high

## Phase Cycle

### 1. t=57-62
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 2. t=58-63
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=59-64
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=65-70
- **Métrique :** S(t)
- **Valeur :** 5.0000
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
