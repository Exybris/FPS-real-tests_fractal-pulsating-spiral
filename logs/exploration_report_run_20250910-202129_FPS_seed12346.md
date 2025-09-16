# Rapport d'exploration FPS

**Run ID :** run_20250910-202129_FPS_seed12346
**Date :** 2025-09-10 20:23:25
**Total événements :** 1374

## Résumé par type d'événement

- **anomaly** : 1137 événements
- **harmonic_emergence** : 33 événements
- **fractal_pattern** : 204 événements

## Anomaly

### 1. t=1497-1501
- **Métrique :** d_effort_dt
- **Valeur :** 156.0282
- **Sévérité :** high

### 2. t=1498-1501
- **Métrique :** d_effort_dt
- **Valeur :** 137.0263
- **Sévérité :** high

### 3. t=985-988
- **Métrique :** d_effort_dt
- **Valeur :** 130.6257
- **Sévérité :** high

### 4. t=2267-2270
- **Métrique :** d_effort_dt
- **Valeur :** 124.8706
- **Sévérité :** high

### 5. t=60-109
- **Métrique :** C(t)
- **Valeur :** 91.4341
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

### 3. t=2640-2733
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=4150-4250
- **Métrique :** mean_high_effort
- **Valeur :** 0.9544
- **Sévérité :** high
- **scale :** 10/100

### 2. t=3450-3550
- **Métrique :** f_mean(t)
- **Valeur :** 0.9339
- **Sévérité :** high
- **scale :** 10/100

### 3. t=550-650
- **Métrique :** f_mean(t)
- **Valeur :** 0.9326
- **Sévérité :** high
- **scale :** 10/100

### 4. t=1850-1950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9325
- **Sévérité :** high
- **scale :** 10/100

### 5. t=2150-2250
- **Métrique :** f_mean(t)
- **Valeur :** 0.9307
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 204

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.748
- Corrélation max : 0.816

### C(t)
- Patterns détectés : 15
- Corrélation moyenne : 0.749
- Corrélation max : 0.914

### A_mean(t)
- Patterns détectés : 46
- Corrélation moyenne : 0.672
- Corrélation max : 0.814

### f_mean(t)
- Patterns détectés : 48
- Corrélation moyenne : 0.925
- Corrélation max : 0.934

### entropy_S
- Patterns détectés : 5
- Corrélation moyenne : 0.796
- Corrélation max : 0.899

### effort(t)
- Patterns détectés : 7
- Corrélation moyenne : 0.729
- Corrélation max : 0.814

### mean_high_effort
- Patterns détectés : 70
- Corrélation moyenne : 0.692
- Corrélation max : 0.954

### d_effort_dt
- Patterns détectés : 5
- Corrélation moyenne : 0.803
- Corrélation max : 0.915

### mean_abs_error
- Patterns détectés : 5
- Corrélation moyenne : 0.750
- Corrélation max : 0.836

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
