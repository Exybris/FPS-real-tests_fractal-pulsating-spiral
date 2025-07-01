# Rapport d'exploration FPS

**Run ID :** run_20250629-234851_seed12345
**Date :** 2025-06-29 23:48:52
**Total événements :** 7

## Résumé par type d'événement

- **phase_cycle** : 7 événements

## Phase Cycle

### 1. t=15-23
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 2. t=35-43
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=65-73
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 4. t=85-93
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=115-123
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

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
    "variance_d2S"
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
