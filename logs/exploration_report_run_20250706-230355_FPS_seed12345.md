# Rapport d'exploration FPS

**Run ID :** run_20250706-230355_FPS_seed12345
**Date :** 2025-07-06 23:03:58
**Total événements :** 552

## Résumé par type d'événement

- **phase_cycle** : 552 événements

## Phase Cycle

### 1. t=233-282
- **Métrique :** S(t)
- **Valeur :** 49.0000
- **Sévérité :** medium

### 2. t=234-281
- **Métrique :** S(t)
- **Valeur :** 47.0000
- **Sévérité :** medium

### 3. t=235-281
- **Métrique :** S(t)
- **Valeur :** 46.0000
- **Sévérité :** medium

### 4. t=236-280
- **Métrique :** S(t)
- **Valeur :** 44.0000
- **Sévérité :** medium

### 5. t=237-277
- **Métrique :** S(t)
- **Valeur :** 40.0000
- **Sévérité :** medium

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "C(t)",
    "effort(t)"
  ],
  "window_sizes": [
    10,
    50
  ],
  "fractal_threshold": 0.8,
  "detect_fractal_patterns": false,
  "detect_anomalies": false,
  "detect_harmonics": false,
  "anomaly_threshold": 3.0,
  "min_duration": 3
}
```
