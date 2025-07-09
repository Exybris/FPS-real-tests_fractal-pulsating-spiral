# Rapport d'exploration FPS

**Run ID :** run_20250706-233152_FPS_seed12345
**Date :** 2025-07-06 23:31:52
**Total événements :** 125

## Résumé par type d'événement

- **anomaly** : 82 événements
- **harmonic_emergence** : 25 événements
- **phase_cycle** : 8 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=288-337
- **Métrique :** mean_high_effort
- **Valeur :** 75.8013
- **Sévérité :** high

### 2. t=289-338
- **Métrique :** mean_high_effort
- **Valeur :** 58.8720
- **Sévérité :** high

### 3. t=290-339
- **Métrique :** mean_high_effort
- **Valeur :** 48.9798
- **Sévérité :** high

### 4. t=291-340
- **Métrique :** mean_high_effort
- **Valeur :** 41.9412
- **Sévérité :** high

### 5. t=127-132
- **Métrique :** effort(t)
- **Valeur :** 41.6295
- **Sévérité :** high

## Harmonic Emergence

### 1. t=230-323
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=350-443
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=340-433
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=378-386
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 2. t=383-390
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 3. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 4. t=377-383
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 5. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9088
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9021
- **Sévérité :** high
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.8981
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.8336
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.7728
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.714
- Corrélation max : 0.714

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.747
- Corrélation max : 0.834

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.773
- Corrélation max : 0.773

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.817
- Corrélation max : 0.909

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
