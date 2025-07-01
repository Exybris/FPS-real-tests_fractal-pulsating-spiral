# Rapport d'exploration FPS

**Run ID :** run_20250623-183140_seed12347
**Date :** 2025-06-23 18:31:41
**Total événements :** 190

## Résumé par type d'événement

- **anomaly** : 154 événements
- **harmonic_emergence** : 28 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=76-125
- **Métrique :** mean_abs_error
- **Valeur :** 1778414.1429
- **Sévérité :** high

### 2. t=77-126
- **Métrique :** mean_abs_error
- **Valeur :** 1270560.3317
- **Sévérité :** high

### 3. t=78-127
- **Métrique :** mean_abs_error
- **Valeur :** 1048386.2524
- **Sévérité :** high

### 4. t=79-128
- **Métrique :** mean_abs_error
- **Valeur :** 917744.7891
- **Sévérité :** high

### 5. t=80-129
- **Métrique :** mean_abs_error
- **Valeur :** 829926.3333
- **Sévérité :** high

## Harmonic Emergence

### 1. t=230-323
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=330-423
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=340-433
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=30-123
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=100-105
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.8996
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8245
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.6936
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** entropy_S
- **Valeur :** 0.6853
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.6788
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.806
- Corrélation max : 0.900

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.682
- Corrélation max : 0.685

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.676
- Corrélation max : 0.676

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.657
- Corrélation max : 0.657

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
