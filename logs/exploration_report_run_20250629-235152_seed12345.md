# Rapport d'exploration FPS

**Run ID :** run_20250629-235152_seed12345
**Date :** 2025-06-29 23:51:53
**Total événements :** 133

## Résumé par type d'événement

- **anomaly** : 75 événements
- **harmonic_emergence** : 35 événements
- **phase_cycle** : 11 événements
- **fractal_pattern** : 12 événements

## Anomaly

### 1. t=278-327
- **Métrique :** mean_high_effort
- **Valeur :** 188.5413
- **Sévérité :** high

### 2. t=279-328
- **Métrique :** mean_high_effort
- **Valeur :** 175.0465
- **Sévérité :** high

### 3. t=280-329
- **Métrique :** mean_high_effort
- **Valeur :** 160.9548
- **Sévérité :** high

### 4. t=281-330
- **Métrique :** mean_high_effort
- **Valeur :** 135.5149
- **Sévérité :** high

### 5. t=282-331
- **Métrique :** mean_high_effort
- **Valeur :** 121.6234
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=250-343
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=280-373
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=320-413
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=390-483
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-13
- **Métrique :** S(t)
- **Valeur :** 12.0000
- **Sévérité :** medium

### 2. t=350-359
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 3. t=349-357
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 4. t=351-359
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=342-348
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9305
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** d_effort_dt
- **Valeur :** 0.8740
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.8508
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.8211
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.6987
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 12

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.731
- Corrélation max : 0.851

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.673
- Corrélation max : 0.673

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.759
- Corrélation max : 0.821

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.734
- Corrélation max : 0.931

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.771
- Corrélation max : 0.874

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
