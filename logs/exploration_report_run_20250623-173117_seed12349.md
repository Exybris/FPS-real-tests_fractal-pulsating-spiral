# Rapport d'exploration FPS

**Run ID :** run_20250623-173117_seed12349
**Date :** 2025-06-23 17:31:17
**Total événements :** 187

## Résumé par type d'événement

- **anomaly** : 152 événements
- **harmonic_emergence** : 28 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=76-125
- **Métrique :** mean_abs_error
- **Valeur :** 1820507.0000
- **Sévérité :** high

### 2. t=77-126
- **Métrique :** mean_abs_error
- **Valeur :** 1300632.9214
- **Sévérité :** high

### 3. t=78-127
- **Métrique :** mean_abs_error
- **Valeur :** 1073200.2587
- **Sévérité :** high

### 4. t=79-128
- **Métrique :** mean_abs_error
- **Valeur :** 939466.6753
- **Sévérité :** high

### 5. t=80-129
- **Métrique :** mean_abs_error
- **Valeur :** 849569.6667
- **Sévérité :** high

## Harmonic Emergence

### 1. t=230-323
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=130-223
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=330-423
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=340-433
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

### 3. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.6936
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** entropy_S
- **Valeur :** 0.6922
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.6918
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
- Corrélation moyenne : 0.692
- Corrélation max : 0.692

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.676
- Corrélation max : 0.676

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.650
- Corrélation max : 0.650

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
