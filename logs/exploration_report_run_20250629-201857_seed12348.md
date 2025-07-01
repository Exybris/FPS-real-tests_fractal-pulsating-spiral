# Rapport d'exploration FPS

**Run ID :** run_20250629-201857_seed12348
**Date :** 2025-06-29 20:18:57
**Total événements :** 158

## Résumé par type d'événement

- **anomaly** : 114 événements
- **harmonic_emergence** : 35 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=68-117
- **Métrique :** mean_abs_error
- **Valeur :** 255416.0622
- **Sévérité :** high

### 2. t=67-116
- **Métrique :** mean_abs_error
- **Valeur :** 229300.5998
- **Sévérité :** high

### 3. t=70-119
- **Métrique :** mean_abs_error
- **Valeur :** 201317.2455
- **Sévérité :** high

### 4. t=71-120
- **Métrique :** mean_abs_error
- **Valeur :** 137136.6673
- **Sévérité :** high

### 5. t=72-121
- **Métrique :** mean_abs_error
- **Valeur :** 86539.1999
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=290-383
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=70-163
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=100-193
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=140-233
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9306
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8905
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** effort(t)
- **Valeur :** 0.8674
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.7687
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7097
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.700
- Corrélation max : 0.710

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.654
- Corrélation max : 0.654

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.867
- Corrélation max : 0.867

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.820
- Corrélation max : 0.931

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.658
- Corrélation max : 0.658

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
