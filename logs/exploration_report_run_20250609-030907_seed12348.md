# Rapport d'exploration FPS

**Run ID :** run_20250609-030907_seed12348
**Date :** 2025-06-09 03:09:07
**Total événements :** 159

## Résumé par type d'événement

- **anomaly** : 125 événements
- **harmonic_emergence** : 27 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=104-152
- **Métrique :** mean_abs_error
- **Valeur :** 33.5154
- **Sévérité :** high

### 2. t=301-348
- **Métrique :** mean_abs_error
- **Valeur :** 24.0075
- **Sévérité :** high

### 3. t=105-152
- **Métrique :** mean_abs_error
- **Valeur :** 19.2587
- **Sévérité :** high

### 4. t=204-246
- **Métrique :** mean_abs_error
- **Valeur :** 18.2671
- **Sévérité :** high

### 5. t=103-107
- **Métrique :** f_mean(t)
- **Valeur :** 18.2005
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-120
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=140-240
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=240-340
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=330-430
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 5. t=340-440
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=350-450
- **Métrique :** C(t)
- **Valeur :** 0.8758
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** C(t)
- **Valeur :** 0.8725
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** C(t)
- **Valeur :** 0.8689
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** C(t)
- **Valeur :** 0.8647
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** C(t)
- **Valeur :** 0.8591
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### C(t)
- Patterns détectés : 6
- Corrélation moyenne : 0.865
- Corrélation max : 0.876

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.839
- Corrélation max : 0.839

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
