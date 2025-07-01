# Rapport d'exploration FPS

**Run ID :** run_20250629-194200_seed12349
**Date :** 2025-06-29 19:42:01
**Total événements :** 155

## Résumé par type d'événement

- **anomaly** : 104 événements
- **harmonic_emergence** : 40 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 8 événements

## Anomaly

### 1. t=79-128
- **Métrique :** effort(t)
- **Valeur :** 61.6602
- **Sévérité :** high

### 2. t=300-349
- **Métrique :** mean_high_effort
- **Valeur :** 58.7018
- **Sévérité :** high

### 3. t=80-129
- **Métrique :** effort(t)
- **Valeur :** 56.0102
- **Sévérité :** high

### 4. t=303-352
- **Métrique :** mean_high_effort
- **Valeur :** 51.0452
- **Sévérité :** high

### 5. t=304-353
- **Métrique :** mean_high_effort
- **Valeur :** 47.1228
- **Sévérité :** high

## Harmonic Emergence

### 1. t=90-183
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=380-473
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=130-223
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=260-353
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=280-373
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.9490
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8887
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8331
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** entropy_S
- **Valeur :** 0.8048
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=200-300
- **Métrique :** mean_abs_error
- **Valeur :** 0.7848
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 8

### S(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.671
- Corrélation max : 0.671

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.833
- Corrélation max : 0.833

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.746
- Corrélation max : 0.805

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.766
- Corrélation max : 0.766

### mean_high_effort
- Patterns détectés : 2
- Corrélation moyenne : 0.919
- Corrélation max : 0.949

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.785
- Corrélation max : 0.785

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
