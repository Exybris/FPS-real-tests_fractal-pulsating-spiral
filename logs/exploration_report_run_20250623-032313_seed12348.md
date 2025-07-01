# Rapport d'exploration FPS

**Run ID :** run_20250623-032313_seed12348
**Date :** 2025-06-23 03:23:13
**Total événements :** 202

## Résumé par type d'événement

- **anomaly** : 171 événements
- **harmonic_emergence** : 25 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=65-114
- **Métrique :** mean_abs_error
- **Valeur :** 1596764.1429
- **Sévérité :** high

### 2. t=66-115
- **Métrique :** mean_abs_error
- **Valeur :** 1140783.3032
- **Sévérité :** high

### 3. t=67-116
- **Métrique :** mean_abs_error
- **Valeur :** 941302.4248
- **Sévérité :** high

### 4. t=70-119
- **Métrique :** mean_abs_error
- **Valeur :** 475861.2071
- **Sévérité :** high

### 5. t=68-117
- **Métrique :** mean_abs_error
- **Valeur :** 470651.0861
- **Sévérité :** high

## Harmonic Emergence

### 1. t=60-153
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=20-113
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=40-133
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=50-143
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=110-203
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

### 3. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8122
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** entropy_S
- **Valeur :** 0.8015
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6742
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.863
- Corrélation max : 0.901

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.802
- Corrélation max : 0.802

### mean_high_effort
- Patterns détectés : 2
- Corrélation moyenne : 0.674
- Corrélation max : 0.674

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
