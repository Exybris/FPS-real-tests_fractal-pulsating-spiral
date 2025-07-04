# Rapport d'exploration FPS

**Run ID :** run_20250623-170523_seed12345
**Date :** 2025-06-23 17:05:24
**Total événements :** 115

## Résumé par type d'événement

- **anomaly** : 5 événements
- **phase_cycle** : 103 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=474-499
- **Métrique :** C(t)
- **Valeur :** 7.0000
- **Sévérité :** medium

### 2. t=475-499
- **Métrique :** C(t)
- **Valeur :** 4.8990
- **Sévérité :** medium

### 3. t=476-499
- **Métrique :** C(t)
- **Valeur :** 3.9581
- **Sévérité :** low

### 4. t=477-499
- **Métrique :** C(t)
- **Valeur :** 3.3912
- **Sévérité :** low

### 5. t=478-499
- **Métrique :** C(t)
- **Valeur :** 3.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=5-30
- **Métrique :** S(t)
- **Valeur :** 25.0000
- **Sévérité :** medium

### 2. t=6-24
- **Métrique :** S(t)
- **Valeur :** 18.0000
- **Sévérité :** medium

### 3. t=84-100
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 4. t=85-101
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 5. t=86-102
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=200-300
- **Métrique :** C(t)
- **Valeur :** 0.9243
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** C(t)
- **Valeur :** 0.9237
- **Sévérité :** high
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** C(t)
- **Valeur :** 0.9231
- **Sévérité :** high
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** C(t)
- **Valeur :** 0.9169
- **Sévérité :** high
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** C(t)
- **Valeur :** 0.8817
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### S(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.732
- Corrélation max : 0.819

### C(t)
- Patterns détectés : 5
- Corrélation moyenne : 0.914
- Corrélation max : 0.924

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
