# Rapport d'exploration FPS

**Run ID :** run_20250625-225450_seed12345
**Date :** 2025-06-25 22:54:50
**Total événements :** 117

## Résumé par type d'événement

- **anomaly** : 77 événements
- **harmonic_emergence** : 33 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=443-475
- **Métrique :** d_effort_dt
- **Valeur :** 1361064.1500
- **Sévérité :** high

### 2. t=442-491
- **Métrique :** effort(t)
- **Valeur :** 890642.8362
- **Sévérité :** high

### 3. t=306-310
- **Métrique :** d_effort_dt
- **Valeur :** 83696.7879
- **Sévérité :** high

### 4. t=307-309
- **Métrique :** d_effort_dt
- **Valeur :** 20260.7627
- **Sévérité :** high

### 5. t=444-449
- **Métrique :** d_effort_dt
- **Valeur :** 9260.5720
- **Sévérité :** high

## Harmonic Emergence

### 1. t=290-383
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=350-443
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=380-473
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=40-133
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=130-223
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8752
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.7731
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** d_effort_dt
- **Valeur :** 0.6983
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6811
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=200-300
- **Métrique :** mean_abs_error
- **Valeur :** 0.6698
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.875
- Corrélation max : 0.875

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.773
- Corrélation max : 0.773

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.666
- Corrélation max : 0.681

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.698
- Corrélation max : 0.698

### mean_abs_error
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
