# Rapport d'exploration FPS

**Run ID :** run_20250706-151313_FPS_seed12349
**Date :** 2025-07-06 15:13:14
**Total événements :** 135

## Résumé par type d'événement

- **anomaly** : 81 événements
- **harmonic_emergence** : 31 événements
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

### 1. t=220-313
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=230-323
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=240-333
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=250-343
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=265-270
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 2. t=266-271
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=267-272
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=282-287
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=290-295
- **Métrique :** S(t)
- **Valeur :** 5.0000
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

### 5. t=300-400
- **Métrique :** entropy_S
- **Valeur :** 0.7021
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
- Corrélation moyenne : 0.702
- Corrélation max : 0.702

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
