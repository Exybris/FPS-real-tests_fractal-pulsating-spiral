# Rapport d'exploration FPS

**Run ID :** run_20250623-025904_seed12348
**Date :** 2025-06-23 02:59:04
**Total événements :** 181

## Résumé par type d'événement

- **anomaly** : 151 événements
- **harmonic_emergence** : 26 événements
- **fractal_pattern** : 4 événements

## Anomaly

### 1. t=75-124
- **Métrique :** mean_abs_error
- **Valeur :** 220229.2591
- **Sévérité :** high

### 2. t=76-125
- **Métrique :** mean_abs_error
- **Valeur :** 201750.3608
- **Sévérité :** high

### 3. t=78-127
- **Métrique :** mean_abs_error
- **Valeur :** 178280.2410
- **Sévérité :** high

### 4. t=80-129
- **Métrique :** mean_abs_error
- **Valeur :** 129408.2565
- **Sévérité :** high

### 5. t=81-130
- **Métrique :** mean_abs_error
- **Valeur :** 84005.3596
- **Sévérité :** high

## Harmonic Emergence

### 1. t=40-133
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=140-233
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=160-253
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=230-323
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.9009
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.8213
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.7572
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6573
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 4

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.829
- Corrélation max : 0.901

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.821
- Corrélation max : 0.821

### mean_high_effort
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
