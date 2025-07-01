# Rapport d'exploration FPS

**Run ID :** run_20250623-025635_seed12348
**Date :** 2025-06-23 02:56:35
**Total événements :** 181

## Résumé par type d'événement

- **anomaly** : 150 événements
- **harmonic_emergence** : 25 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=75-124
- **Métrique :** mean_abs_error
- **Valeur :** 221111.2344
- **Sévérité :** high

### 2. t=76-125
- **Métrique :** mean_abs_error
- **Valeur :** 202031.8050
- **Sévérité :** high

### 3. t=78-127
- **Métrique :** mean_abs_error
- **Valeur :** 177932.4730
- **Sévérité :** high

### 4. t=80-129
- **Métrique :** mean_abs_error
- **Valeur :** 128936.5420
- **Sévérité :** high

### 5. t=81-130
- **Métrique :** mean_abs_error
- **Valeur :** 83775.2007
- **Sévérité :** high

## Harmonic Emergence

### 1. t=40-133
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=120-213
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=140-233
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=160-253
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=230-323
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

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

### 3. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.8182
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.6998
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.6936
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.806
- Corrélation max : 0.900

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.759
- Corrélation max : 0.818

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.670
- Corrélation max : 0.670

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
