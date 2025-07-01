# Rapport d'exploration FPS

**Run ID :** run_20250629-233151_seed12345
**Date :** 2025-06-29 23:31:51
**Total événements :** 6

## Résumé par type d'événement

- **phase_cycle** : 6 événements

## Phase Cycle

### 1. t=20-35
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 2. t=20-32
- **Métrique :** S(t)
- **Valeur :** 12.0000
- **Sévérité :** medium

### 3. t=1-11
- **Métrique :** S(t)
- **Valeur :** 10.0000
- **Sévérité :** medium

### 4. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=52-57
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "gamma"
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
