# Rapport d'exploration FPS

**Run ID :** run_20250623-021334_seed12345
**Date :** 2025-06-23 02:13:34
**Total événements :** 204

## Résumé par type d'événement

- **anomaly** : 171 événements
- **harmonic_emergence** : 27 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=111-160
- **Métrique :** mean_abs_error
- **Valeur :** 483185.5714
- **Sévérité :** high

### 2. t=112-161
- **Métrique :** mean_abs_error
- **Valeur :** 345204.3441
- **Sévérité :** high

### 3. t=113-162
- **Métrique :** mean_abs_error
- **Valeur :** 284840.7916
- **Sévérité :** high

### 4. t=114-163
- **Métrique :** mean_abs_error
- **Valeur :** 187615.9700
- **Sévérité :** high

### 5. t=115-164
- **Métrique :** mean_abs_error
- **Valeur :** 151109.7071
- **Sévérité :** high

## Harmonic Emergence

### 1. t=30-123
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
- **Valeur :** 0.9061
- **Sévérité :** high
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8917
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8501
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.8208
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7435
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.883
- Corrélation max : 0.906

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.782
- Corrélation max : 0.821

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.647
- Corrélation max : 0.647

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
