# Rapport d'exploration FPS

**Run ID :** run_20250629-194915_seed12349
**Date :** 2025-06-29 19:49:16
**Total événements :** 116

## Résumé par type d'événement

- **anomaly** : 65 événements
- **harmonic_emergence** : 39 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=219-268
- **Métrique :** mean_high_effort
- **Valeur :** 52.1394
- **Sévérité :** high

### 2. t=220-269
- **Métrique :** mean_high_effort
- **Valeur :** 44.4835
- **Sévérité :** high

### 3. t=221-270
- **Métrique :** mean_high_effort
- **Valeur :** 38.4735
- **Sévérité :** high

### 4. t=222-271
- **Métrique :** mean_high_effort
- **Valeur :** 31.9549
- **Sévérité :** high

### 5. t=223-272
- **Métrique :** mean_high_effort
- **Valeur :** 28.6168
- **Sévérité :** high

## Harmonic Emergence

### 1. t=90-183
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=220-313
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=310-403
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=130-223
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=230-323
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-10
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 2. t=2-8
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9384
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8995
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8769
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.8113
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=200-300
- **Métrique :** entropy_S
- **Valeur :** 0.8014
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### S(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.750
- Corrélation max : 0.750

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.877
- Corrélation max : 0.877

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.734
- Corrélation max : 0.801

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.759
- Corrélation max : 0.811

### mean_high_effort
- Patterns détectés : 2
- Corrélation moyenne : 0.919
- Corrélation max : 0.938

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.667
- Corrélation max : 0.667

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
