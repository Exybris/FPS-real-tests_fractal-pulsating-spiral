# Rapport d'exploration FPS

**Run ID :** run_20250725-173217_FPS_seed12348
**Date :** 2025-07-25 17:32:18
**Total événements :** 137

## Résumé par type d'événement

- **anomaly** : 120 événements
- **harmonic_emergence** : 5 événements
- **fractal_pattern** : 12 événements

## Anomaly

### 1. t=60-109
- **Métrique :** C(t)
- **Valeur :** 91.4341
- **Sévérité :** high

### 2. t=61-110
- **Métrique :** C(t)
- **Valeur :** 83.2655
- **Sévérité :** high

### 3. t=62-111
- **Métrique :** C(t)
- **Valeur :** 73.2652
- **Sévérité :** high

### 4. t=63-112
- **Métrique :** C(t)
- **Valeur :** 63.0479
- **Sévérité :** high

### 5. t=64-113
- **Métrique :** C(t)
- **Valeur :** 53.4897
- **Sévérité :** high

## Harmonic Emergence

### 1. t=50-143
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=180-273
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=260-353
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** f_mean(t)
- **Valeur :** 0.9257
- **Sévérité :** high
- **scale :** 10/100

### 2. t=250-350
- **Métrique :** f_mean(t)
- **Valeur :** 0.9233
- **Sévérité :** high
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** f_mean(t)
- **Valeur :** 0.9113
- **Sévérité :** high
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** mean_abs_error
- **Valeur :** 0.7255
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.6996
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 12

### C(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.687
- Corrélation max : 0.687

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.692
- Corrélation max : 0.700

### f_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.920
- Corrélation max : 0.926

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.674
- Corrélation max : 0.688

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.726
- Corrélation max : 0.726

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
