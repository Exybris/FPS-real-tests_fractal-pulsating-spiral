# Rapport d'exploration FPS

**Run ID :** run_20250629-210153_seed12345
**Date :** 2025-06-29 21:01:53
**Total événements :** 57

## Résumé par type d'événement

- **anomaly** : 47 événements
- **harmonic_emergence** : 8 événements
- **phase_cycle** : 2 événements

## Anomaly

### 1. t=95-144
- **Métrique :** effort(t)
- **Valeur :** 374.6621
- **Sévérité :** high

### 2. t=96-145
- **Métrique :** effort(t)
- **Valeur :** 374.0623
- **Sévérité :** high

### 3. t=100-125
- **Métrique :** d_effort_dt
- **Valeur :** 347.7294
- **Sévérité :** high

### 4. t=97-146
- **Métrique :** effort(t)
- **Valeur :** 328.2940
- **Sévérité :** high

### 5. t=101-125
- **Métrique :** d_effort_dt
- **Valeur :** 317.1040
- **Sévérité :** high

## Harmonic Emergence

### 1. t=100-193
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=30-123
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=60-153
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=141-147
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=144-149
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
