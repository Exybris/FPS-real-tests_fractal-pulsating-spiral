# Rapport d'exploration FPS

**Run ID :** run_20250910-202325_FPS_seed12347
**Date :** 2025-09-10 20:25:21
**Total événements :** 1190

## Résumé par type d'événement

- **anomaly** : 962 événements
- **harmonic_emergence** : 50 événements
- **fractal_pattern** : 178 événements

## Anomaly

### 1. t=985-995
- **Métrique :** d_effort_dt
- **Valeur :** 130.6257
- **Sévérité :** high

### 2. t=60-109
- **Métrique :** C(t)
- **Valeur :** 91.4341
- **Sévérité :** high

### 3. t=61-110
- **Métrique :** C(t)
- **Valeur :** 83.2655
- **Sévérité :** high

### 4. t=62-111
- **Métrique :** C(t)
- **Valeur :** 73.2652
- **Sévérité :** high

### 5. t=854-903
- **Métrique :** C(t)
- **Valeur :** 71.3573
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=550-643
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=3360-3453
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=3910-4003
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=2050-2150
- **Métrique :** f_mean(t)
- **Valeur :** 0.9498
- **Sévérité :** high
- **scale :** 10/100

### 2. t=2750-2850
- **Métrique :** f_mean(t)
- **Valeur :** 0.9442
- **Sévérité :** high
- **scale :** 10/100

### 3. t=1650-1750
- **Métrique :** f_mean(t)
- **Valeur :** 0.9407
- **Sévérité :** high
- **scale :** 10/100

### 4. t=4850-4950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9383
- **Sévérité :** high
- **scale :** 10/100

### 5. t=1850-1950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9381
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 178

### S(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.711
- Corrélation max : 0.816

### C(t)
- Patterns détectés : 15
- Corrélation moyenne : 0.749
- Corrélation max : 0.914

### A_mean(t)
- Patterns détectés : 31
- Corrélation moyenne : 0.695
- Corrélation max : 0.814

### f_mean(t)
- Patterns détectés : 48
- Corrélation moyenne : 0.923
- Corrélation max : 0.950

### entropy_S
- Patterns détectés : 6
- Corrélation moyenne : 0.727
- Corrélation max : 0.842

### effort(t)
- Patterns détectés : 10
- Corrélation moyenne : 0.755
- Corrélation max : 0.859

### mean_high_effort
- Patterns détectés : 53
- Corrélation moyenne : 0.684
- Corrélation max : 0.743

### d_effort_dt
- Patterns détectés : 4
- Corrélation moyenne : 0.715
- Corrélation max : 0.762

### mean_abs_error
- Patterns détectés : 7
- Corrélation moyenne : 0.730
- Corrélation max : 0.880

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
  "min_duration": 3,
  "spacing_effect": {
    "enabled": true,
    "start_interval": 2.0,
    "growth": 1.5,
    "num_blocks": 0,
    "order": [
      "gamma",
      "G",
      "gamma",
      "G"
    ]
  }
}
```
