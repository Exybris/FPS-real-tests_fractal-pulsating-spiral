# Rapport d'exploration FPS

**Run ID :** run_20250713-223253_FPS_seed12347
**Date :** 2025-07-13 22:32:57
**Total événements :** 387

## Résumé par type d'événement

- **anomaly** : 304 événements
- **harmonic_emergence** : 14 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 68 événements

## Anomaly

### 1. t=600-649
- **Métrique :** A_mean(t)
- **Valeur :** 21.1652
- **Sévérité :** high

### 2. t=601-650
- **Métrique :** A_mean(t)
- **Valeur :** 20.4575
- **Sévérité :** high

### 3. t=201-250
- **Métrique :** A_mean(t)
- **Valeur :** 20.0254
- **Sévérité :** high

### 4. t=200-249
- **Métrique :** A_mean(t)
- **Valeur :** 19.9400
- **Sévérité :** high

### 5. t=1700-1749
- **Métrique :** A_mean(t)
- **Valeur :** 19.8855
- **Sévérité :** high

## Harmonic Emergence

### 1. t=160-253
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=170-263
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=200-293
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=220-313
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1113-1118
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=1200-1300
- **Métrique :** S(t)
- **Valeur :** 0.8211
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=450-550
- **Métrique :** entropy_S
- **Valeur :** 0.7901
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=600-700
- **Métrique :** mean_abs_error
- **Valeur :** 0.7803
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=850-950
- **Métrique :** mean_abs_error
- **Valeur :** 0.7581
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=950-1050
- **Métrique :** mean_abs_error
- **Valeur :** 0.7549
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 68

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.749
- Corrélation max : 0.821

### A_mean(t)
- Patterns détectés : 17
- Corrélation moyenne : 0.668
- Corrélation max : 0.701

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.725
- Corrélation max : 0.790

### effort(t)
- Patterns détectés : 7
- Corrélation moyenne : 0.666
- Corrélation max : 0.701

### mean_high_effort
- Patterns détectés : 35
- Corrélation moyenne : 0.663
- Corrélation max : 0.686

### mean_abs_error
- Patterns détectés : 3
- Corrélation moyenne : 0.764
- Corrélation max : 0.780

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
