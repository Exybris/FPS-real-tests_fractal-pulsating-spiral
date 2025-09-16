# Rapport d'exploration FPS

**Run ID :** run_20250910-201925_KURAMOTO_seed12345
**Date :** 2025-09-10 20:19:33
**Total événements :** 937

## Résumé par type d'événement

- **anomaly** : 5 événements
- **phase_cycle** : 915 événements
- **fractal_pattern** : 17 événements

## Anomaly

### 1. t=474-523
- **Métrique :** C(t)
- **Valeur :** 7.0000
- **Sévérité :** medium

### 2. t=475-524
- **Métrique :** C(t)
- **Valeur :** 4.8990
- **Sévérité :** medium

### 3. t=476-525
- **Métrique :** C(t)
- **Valeur :** 3.9581
- **Sévérité :** low

### 4. t=477-526
- **Métrique :** C(t)
- **Valeur :** 3.3912
- **Sévérité :** low

### 5. t=478-527
- **Métrique :** C(t)
- **Valeur :** 3.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=5-27
- **Métrique :** S(t)
- **Valeur :** 22.0000
- **Sévérité :** medium

### 2. t=6-23
- **Métrique :** S(t)
- **Valeur :** 17.0000
- **Sévérité :** medium

### 3. t=84-98
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 4. t=85-99
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 5. t=171-185
- **Métrique :** S(t)
- **Valeur :** 14.0000
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

### 5. t=1950-2050
- **Métrique :** S(t)
- **Valeur :** 0.9136
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 17

### S(t)
- Patterns détectés : 12
- Corrélation moyenne : 0.740
- Corrélation max : 0.914

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
