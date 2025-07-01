# Rapport d'exploration FPS

**Run ID :** run_20250629-210156_seed12345
**Date :** 2025-06-29 21:01:56
**Total événements :** 61

## Résumé par type d'événement

- **anomaly** : 54 événements
- **harmonic_emergence** : 3 événements
- **phase_cycle** : 4 événements

## Anomaly

### 1. t=114-163
- **Métrique :** mean_high_effort
- **Valeur :** 183.4261
- **Sévérité :** high

### 2. t=115-164
- **Métrique :** mean_high_effort
- **Valeur :** 161.1003
- **Sévérité :** high

### 3. t=116-165
- **Métrique :** mean_high_effort
- **Valeur :** 135.7897
- **Sévérité :** high

### 4. t=117-166
- **Métrique :** mean_high_effort
- **Valeur :** 115.7061
- **Sévérité :** high

### 5. t=113-152
- **Métrique :** effort(t)
- **Valeur :** 102.7392
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=70-163
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=100-193
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=54-59
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
