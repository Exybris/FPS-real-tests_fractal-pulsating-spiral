# Rapport d'exploration FPS

**Run ID :** run_20260322-212104_KURAMOTO_seed12345
**Date :** 2026-03-22 21:21:04
**Total événements :** 111

## Résumé par type d'événement

- **anomaly** : 24 événements
- **phase_cycle** : 81 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=431-480
- **Métrique :** C(t)
- **Valeur :** 10.8405
- **Sévérité :** high

### 2. t=432-481
- **Métrique :** C(t)
- **Valeur :** 10.3466
- **Sévérité :** high

### 3. t=433-482
- **Métrique :** C(t)
- **Valeur :** 9.7323
- **Sévérité :** high

### 4. t=434-483
- **Métrique :** C(t)
- **Valeur :** 9.0428
- **Sévérité :** high

### 5. t=235-284
- **Métrique :** C(t)
- **Valeur :** 8.4550
- **Sévérité :** high

## Phase Cycle

### 1. t=82-114
- **Métrique :** S(t)
- **Valeur :** 32.0000
- **Sévérité :** medium

### 2. t=83-115
- **Métrique :** S(t)
- **Valeur :** 32.0000
- **Sévérité :** medium

### 3. t=410-441
- **Métrique :** S(t)
- **Valeur :** 31.0000
- **Sévérité :** medium

### 4. t=412-443
- **Métrique :** S(t)
- **Valeur :** 31.0000
- **Sévérité :** medium

### 5. t=408-438
- **Métrique :** S(t)
- **Valeur :** 30.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=200-300
- **Métrique :** C(t)
- **Valeur :** 0.8987
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** C(t)
- **Valeur :** 0.8510
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** C(t)
- **Valeur :** 0.8217
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** S(t)
- **Valeur :** 0.6541
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** S(t)
- **Valeur :** 0.6519
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### S(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.653
- Corrélation max : 0.654

### C(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.805
- Corrélation max : 0.899

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "C(t)",
    "An_mean(t)",
    "fn_mean(t)",
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
  "detect_fractal_patterns": true,
  "detect_anomalies": true,
  "detect_harmonics": true,
  "anomaly_threshold": 3.0,
  "fractal_threshold": 0.8,
  "min_duration": 3,
  "recurrence_window": [
    1,
    10,
    100
  ]
}
```
