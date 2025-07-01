# Rapport d'exploration FPS

**Run ID :** run_20250623-024408_seed12345
**Date :** 2025-06-23 02:44:08
**Total événements :** 191

## Résumé par type d'événement

- **anomaly** : 156 événements
- **harmonic_emergence** : 28 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=88-137
- **Métrique :** mean_abs_error
- **Valeur :** 1772399.8571
- **Sévérité :** high

### 2. t=89-138
- **Métrique :** mean_abs_error
- **Valeur :** 1266263.5184
- **Sévérité :** high

### 3. t=90-139
- **Métrique :** mean_abs_error
- **Valeur :** 736207.8965
- **Sévérité :** high

### 4. t=91-140
- **Métrique :** mean_abs_error
- **Valeur :** 468598.1043
- **Sévérité :** high

### 5. t=92-141
- **Métrique :** mean_abs_error
- **Valeur :** 420871.5585
- **Sévérité :** high

## Harmonic Emergence

### 1. t=120-213
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

### 3. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.8296
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.8180
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8122
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.863
- Corrélation max : 0.901

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.824
- Corrélation max : 0.830

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.668
- Corrélation max : 0.668

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.764
- Corrélation max : 0.764

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
