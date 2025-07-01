# Rapport d'exploration FPS

**Run ID :** run_20250623-023826_seed12346
**Date :** 2025-06-23 02:38:27
**Total événements :** 184

## Résumé par type d'événement

- **anomaly** : 152 événements
- **harmonic_emergence** : 27 événements
- **fractal_pattern** : 5 événements

## Anomaly

### 1. t=179-228
- **Métrique :** mean_abs_error
- **Valeur :** 2721292.7143
- **Sévérité :** high

### 2. t=180-229
- **Métrique :** mean_abs_error
- **Valeur :** 1944185.3202
- **Sévérité :** high

### 3. t=182-231
- **Métrique :** mean_abs_error
- **Valeur :** 1056650.8999
- **Sévérité :** high

### 4. t=184-233
- **Métrique :** mean_abs_error
- **Valeur :** 543371.8425
- **Sévérité :** high

### 5. t=185-234
- **Métrique :** mean_abs_error
- **Valeur :** 212695.6164
- **Sévérité :** high

## Harmonic Emergence

### 1. t=30-123
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
- **Valeur :** 0.9005
- **Sévérité :** high
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8759
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.8180
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8122
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7524
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 5

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.863
- Corrélation max : 0.901

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.785
- Corrélation max : 0.818

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
