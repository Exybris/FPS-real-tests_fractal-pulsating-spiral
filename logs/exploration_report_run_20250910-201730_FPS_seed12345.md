# Rapport d'exploration FPS

**Run ID :** run_20250910-201730_FPS_seed12345
**Date :** 2025-09-10 20:19:25
**Total événements :** 1390

## Résumé par type d'événement

- **anomaly** : 1139 événements
- **harmonic_emergence** : 29 événements
- **fractal_pattern** : 222 événements

## Anomaly

### 1. t=2266-2287
- **Métrique :** effort(t)
- **Valeur :** 613.6583
- **Sévérité :** high

### 2. t=1497-1501
- **Métrique :** d_effort_dt
- **Valeur :** 156.0282
- **Sévérité :** high

### 3. t=1498-1501
- **Métrique :** d_effort_dt
- **Valeur :** 137.0263
- **Sévérité :** high

### 4. t=985-988
- **Métrique :** d_effort_dt
- **Valeur :** 130.6257
- **Sévérité :** high

### 5. t=2267-2271
- **Métrique :** d_effort_dt
- **Valeur :** 124.8706
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=2640-2733
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=180-273
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=3250-3350
- **Métrique :** mean_high_effort
- **Valeur :** 0.9421
- **Sévérité :** high
- **scale :** 10/100

### 2. t=3100-3200
- **Métrique :** mean_high_effort
- **Valeur :** 0.9398
- **Sévérité :** high
- **scale :** 10/100

### 3. t=1850-1950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9325
- **Sévérité :** high
- **scale :** 10/100

### 4. t=2750-2850
- **Métrique :** f_mean(t)
- **Valeur :** 0.9314
- **Sévérité :** high
- **scale :** 10/100

### 5. t=2950-3050
- **Métrique :** mean_high_effort
- **Valeur :** 0.9309
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 222

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.750
- Corrélation max : 0.823

### C(t)
- Patterns détectés : 15
- Corrélation moyenne : 0.749
- Corrélation max : 0.914

### A_mean(t)
- Patterns détectés : 45
- Corrélation moyenne : 0.667
- Corrélation max : 0.710

### f_mean(t)
- Patterns détectés : 48
- Corrélation moyenne : 0.925
- Corrélation max : 0.933

### entropy_S
- Patterns détectés : 9
- Corrélation moyenne : 0.742
- Corrélation max : 0.906

### effort(t)
- Patterns détectés : 9
- Corrélation moyenne : 0.724
- Corrélation max : 0.814

### mean_high_effort
- Patterns détectés : 87
- Corrélation moyenne : 0.708
- Corrélation max : 0.942

### d_effort_dt
- Patterns détectés : 4
- Corrélation moyenne : 0.737
- Corrélation max : 0.894

### mean_abs_error
- Patterns détectés : 2
- Corrélation moyenne : 0.758
- Corrélation max : 0.816

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
