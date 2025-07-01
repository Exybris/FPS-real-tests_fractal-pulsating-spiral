# Rapport d'exploration FPS

**Run ID :** run_20250629-210157_seed12345
**Date :** 2025-06-29 21:01:57
**Total événements :** 58

## Résumé par type d'événement

- **anomaly** : 48 événements
- **harmonic_emergence** : 7 événements
- **phase_cycle** : 3 événements

## Anomaly

### 1. t=101-150
- **Métrique :** d_effort_dt
- **Valeur :** 909.3162
- **Sévérité :** high

### 2. t=95-144
- **Métrique :** effort(t)
- **Valeur :** 890.4597
- **Sévérité :** high

### 3. t=96-145
- **Métrique :** effort(t)
- **Valeur :** 860.9123
- **Sévérité :** high

### 4. t=97-146
- **Métrique :** effort(t)
- **Valeur :** 737.1692
- **Sévérité :** high

### 5. t=98-147
- **Métrique :** effort(t)
- **Valeur :** 647.1039
- **Sévérité :** high

## Harmonic Emergence

### 1. t=60-153
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=40-133
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=50-143
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

### 3. t=166-171
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
