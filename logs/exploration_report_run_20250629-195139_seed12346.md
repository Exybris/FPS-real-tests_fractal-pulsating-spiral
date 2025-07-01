# Rapport d'exploration FPS

**Run ID :** run_20250629-195139_seed12346
**Date :** 2025-06-29 19:51:40
**Total événements :** 131

## Résumé par type d'événement

- **anomaly** : 78 événements
- **harmonic_emergence** : 32 événements
- **phase_cycle** : 15 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=219-268
- **Métrique :** mean_high_effort
- **Valeur :** 75.2711
- **Sévérité :** high

### 2. t=220-269
- **Métrique :** mean_high_effort
- **Valeur :** 62.1485
- **Sévérité :** high

### 3. t=221-270
- **Métrique :** mean_high_effort
- **Valeur :** 44.8201
- **Sévérité :** high

### 4. t=283-332
- **Métrique :** mean_high_effort
- **Valeur :** 38.0779
- **Sévérité :** high

### 5. t=222-271
- **Métrique :** mean_high_effort
- **Valeur :** 37.0257
- **Sévérité :** high

## Harmonic Emergence

### 1. t=320-413
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=360-453
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 2. t=385-395
- **Métrique :** S(t)
- **Valeur :** 10.0000
- **Sévérité :** medium

### 3. t=348-357
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 4. t=351-359
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=350-357
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9258
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9064
- **Sévérité :** high
- **scale :** 10/100

### 3. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.7907
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7063
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6575
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.652
- Corrélation max : 0.652

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.791
- Corrélation max : 0.791

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.799
- Corrélation max : 0.926

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
