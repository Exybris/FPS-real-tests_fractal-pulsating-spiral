# Rapport d'exploration FPS

**Run ID :** run_20250706-235102_FPS_seed12345
**Date :** 2025-07-06 23:51:03
**Total événements :** 296

## Résumé par type d'événement

- **anomaly** : 112 événements
- **harmonic_emergence** : 34 événements
- **phase_cycle** : 137 événements
- **fractal_pattern** : 13 événements

## Anomaly

### 1. t=271-320
- **Métrique :** mean_high_effort
- **Valeur :** 486.3903
- **Sévérité :** high

### 2. t=272-321
- **Métrique :** mean_high_effort
- **Valeur :** 449.5915
- **Sévérité :** high

### 3. t=273-322
- **Métrique :** mean_high_effort
- **Valeur :** 391.0241
- **Sévérité :** high

### 4. t=274-323
- **Métrique :** mean_high_effort
- **Valeur :** 333.5394
- **Sévérité :** high

### 5. t=275-324
- **Métrique :** mean_high_effort
- **Valeur :** 253.3533
- **Sévérité :** high

## Harmonic Emergence

### 1. t=390-483
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=250-343
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=260-353
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=290-383
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=360-453
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=2-18
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 2. t=295-310
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 3. t=3-17
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 4. t=350-364
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 5. t=4-17
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=100-200
- **Métrique :** mean_abs_error
- **Valeur :** 0.9261
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8661
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8450
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8420
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** d_effort_dt
- **Valeur :** 0.8233
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 13

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.842
- Corrélation max : 0.842

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.737
- Corrélation max : 0.820

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.667
- Corrélation max : 0.667

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.773
- Corrélation max : 0.866

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.756
- Corrélation max : 0.823

### mean_abs_error
- Patterns détectés : 2
- Corrélation moyenne : 0.842
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
