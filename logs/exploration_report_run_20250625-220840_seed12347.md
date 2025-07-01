# Rapport d'exploration FPS

**Run ID :** run_20250625-220840_seed12347
**Date :** 2025-06-25 22:08:41
**Total événements :** 139

## Résumé par type d'événement

- **anomaly** : 101 événements
- **harmonic_emergence** : 33 événements
- **fractal_pattern** : 5 événements

## Anomaly

### 1. t=443-492
- **Métrique :** d_effort_dt
- **Valeur :** 3151511.2002
- **Sévérité :** high

### 2. t=442-491
- **Métrique :** effort(t)
- **Valeur :** 2062015.5824
- **Sévérité :** high

### 3. t=306-310
- **Métrique :** d_effort_dt
- **Valeur :** 84667.7669
- **Sévérité :** high

### 4. t=307-310
- **Métrique :** d_effort_dt
- **Valeur :** 22260.3747
- **Sévérité :** high

### 5. t=444-447
- **Métrique :** d_effort_dt
- **Valeur :** 21345.7836
- **Sévérité :** high

## Harmonic Emergence

### 1. t=290-383
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=370-463
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=40-133
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=120-213
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
- **Valeur :** 0.8807
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.8161
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7149
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** mean_abs_error
- **Valeur :** 0.6684
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6505
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 5

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.881
- Corrélation max : 0.881

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.816
- Corrélation max : 0.816

### mean_high_effort
- Patterns détectés : 2
- Corrélation moyenne : 0.683
- Corrélation max : 0.715

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.668
- Corrélation max : 0.668

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
