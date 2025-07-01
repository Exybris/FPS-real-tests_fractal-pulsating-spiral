# Rapport d'exploration FPS

**Run ID :** run_20250629-210154_seed12345
**Date :** 2025-06-29 21:01:55
**Total événements :** 51

## Résumé par type d'événement

- **anomaly** : 43 événements
- **harmonic_emergence** : 6 événements
- **phase_cycle** : 2 événements

## Anomaly

### 1. t=95-144
- **Métrique :** effort(t)
- **Valeur :** 1124.2546
- **Sévérité :** high

### 2. t=96-145
- **Métrique :** effort(t)
- **Valeur :** 1107.0794
- **Sévérité :** high

### 3. t=97-146
- **Métrique :** effort(t)
- **Valeur :** 990.4147
- **Sévérité :** high

### 4. t=98-147
- **Métrique :** effort(t)
- **Valeur :** 886.9144
- **Sévérité :** high

### 5. t=103-152
- **Métrique :** effort(t)
- **Valeur :** 367.0264
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=60-153
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=70-163
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=80-173
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
