# Rapport d'exploration FPS

**Run ID :** run_20250910-204255_NEUTRAL_seed12345
**Date :** 2025-09-10 20:43:00
**Total événements :** 1292

## Résumé par type d'événement

- **anomaly** : 300 événements
- **harmonic_emergence** : 693 événements
- **phase_cycle** : 299 événements

## Anomaly

### 1. t=91-94
- **Métrique :** S(t)
- **Valeur :** 5.2498
- **Sévérité :** medium

### 2. t=191-194
- **Métrique :** S(t)
- **Valeur :** 5.2498
- **Sévérité :** medium

### 3. t=291-294
- **Métrique :** S(t)
- **Valeur :** 5.2498
- **Sévérité :** medium

### 4. t=391-394
- **Métrique :** S(t)
- **Valeur :** 5.2498
- **Sévérité :** medium

### 5. t=491-494
- **Métrique :** S(t)
- **Valeur :** 5.2498
- **Sévérité :** medium

## Harmonic Emergence

### 1. t=40-133
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=100-193
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=140-233
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=200-293
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=40-63
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 2. t=140-163
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 3. t=240-263
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 4. t=340-363
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

### 5. t=440-463
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

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
