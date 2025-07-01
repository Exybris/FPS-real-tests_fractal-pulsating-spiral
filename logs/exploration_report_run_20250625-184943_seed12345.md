# Rapport d'exploration FPS

**Run ID :** run_20250625-184943_seed12345
**Date :** 2025-06-25 18:49:43
**Total événements :** 53

## Résumé par type d'événement

- **anomaly** : 53 événements

## Anomaly

### 1. t=28-62
- **Métrique :** d_effort_dt
- **Valeur :** 28061458259930469232494474332930581650918901154719560379108623735253368832.0000
- **Sévérité :** high

### 2. t=29-62
- **Métrique :** d_effort_dt
- **Valeur :** 25344990448391275086849242825792647324746644496324352124553665653808037888.0000
- **Sévérité :** high

### 3. t=48-62
- **Métrique :** d_effort_dt
- **Valeur :** 14471218400827015542751210770945215567088163192351475552716029968726884352.0000
- **Sévérité :** high

### 4. t=49-62
- **Métrique :** d_effort_dt
- **Valeur :** 12785650608296038836415571939182202632578024372213537726538388174557675520.0000
- **Sévérité :** high

### 5. t=50-62
- **Métrique :** d_effort_dt
- **Valeur :** 10502520061860817083305771027347673441378079818151086188565100268556910592.0000
- **Sévérité :** high

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
  "min_duration": 3
}
```
