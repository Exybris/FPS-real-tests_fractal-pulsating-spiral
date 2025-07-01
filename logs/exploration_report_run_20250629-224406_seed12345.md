# Rapport d'exploration FPS

**Run ID :** run_20250629-224406_seed12345
**Date :** 2025-06-29 22:44:06
**Total événements :** 5

## Résumé par type d'événement

- **phase_cycle** : 5 événements

## Phase Cycle

### 1. t=40-52
- **Métrique :** S(t)
- **Valeur :** 12.0000
- **Sévérité :** medium

### 2. t=7-16
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 3. t=53-59
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 4. t=67-73
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 5. t=60-65
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "gamma",
    "An_mean(t)"
  ],
  "window_sizes": [
    1,
    10
  ],
  "fractal_threshold": 0.8,
  "detect_fractal_patterns": false,
  "detect_anomalies": false,
  "detect_harmonics": false,
  "recurrence_window": [
    1,
    10
  ],
  "anomaly_threshold": 3.0,
  "min_duration": 3
}
```
