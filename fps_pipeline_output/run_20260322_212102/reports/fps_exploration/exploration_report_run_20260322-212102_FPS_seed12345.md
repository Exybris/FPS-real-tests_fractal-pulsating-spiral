# Rapport d'exploration FPS

**Run ID :** run_20260322-212102_FPS_seed12345
**Date :** 2026-03-22 21:21:04
**Total événements :** 166

## Résumé par type d'événement

- **anomaly** : 138 événements
- **harmonic_emergence** : 4 événements
- **fractal_pattern** : 24 événements

## Anomaly

### 1. t=60-109
- **Métrique :** C(t)
- **Valeur :** 100.2686
- **Sévérité :** high

### 2. t=61-110
- **Métrique :** C(t)
- **Valeur :** 92.5380
- **Sévérité :** high

### 3. t=62-111
- **Métrique :** C(t)
- **Valeur :** 82.3249
- **Sévérité :** high

### 4. t=63-112
- **Métrique :** C(t)
- **Valeur :** 71.2634
- **Sévérité :** high

### 5. t=64-113
- **Métrique :** C(t)
- **Valeur :** 60.7247
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=20-113
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=150-250
- **Métrique :** fn_mean(t)
- **Valeur :** 0.9263
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** fn_mean(t)
- **Valeur :** 0.9263
- **Sévérité :** high
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** An_mean(t)
- **Valeur :** 0.9257
- **Sévérité :** high
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** fn_mean(t)
- **Valeur :** 0.9238
- **Sévérité :** high
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** S(t)
- **Valeur :** 0.9099
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 24

### S(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.847
- Corrélation max : 0.910

### C(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.848
- Corrélation max : 0.855

### An_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.903
- Corrélation max : 0.926

### fn_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.925
- Corrélation max : 0.926

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.703
- Corrélation max : 0.703

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.737
- Corrélation max : 0.737

### mean_high_effort
- Patterns détectés : 6
- Corrélation moyenne : 0.828
- Corrélation max : 0.836

### mean_abs_error
- Patterns détectés : 4
- Corrélation moyenne : 0.788
- Corrélation max : 0.864

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
