# Rapport d'exploration FPS

**Run ID :** run_20250910-204313_FPS_seed12345
**Date :** 2025-09-10 20:53:10
**Total événements :** 2887

## Résumé par type d'événement

- **anomaly** : 2471 événements
- **harmonic_emergence** : 48 événements
- **fractal_pattern** : 368 événements

## Anomaly

### 1. t=3420-3425
- **Métrique :** d_effort_dt
- **Valeur :** 446.2355
- **Sévérité :** high

### 2. t=1497-1511
- **Métrique :** d_effort_dt
- **Valeur :** 147.1373
- **Sévérité :** high

### 3. t=1498-1501
- **Métrique :** d_effort_dt
- **Valeur :** 132.6470
- **Sévérité :** high

### 4. t=985-1000
- **Métrique :** d_effort_dt
- **Valeur :** 122.6341
- **Sévérité :** high

### 5. t=416-427
- **Métrique :** d_effort_dt
- **Valeur :** 117.2827
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

### 3. t=7150-7243
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=9300-9400
- **Métrique :** C(t)
- **Valeur :** 0.9640
- **Sévérité :** high
- **scale :** 10/100

### 2. t=9600-9700
- **Métrique :** C(t)
- **Valeur :** 0.9637
- **Sévérité :** high
- **scale :** 10/100

### 3. t=7700-7800
- **Métrique :** mean_high_effort
- **Valeur :** 0.9400
- **Sévérité :** high
- **scale :** 10/100

### 4. t=3450-3550
- **Métrique :** f_mean(t)
- **Valeur :** 0.9368
- **Sévérité :** high
- **scale :** 10/100

### 5. t=2750-2850
- **Métrique :** f_mean(t)
- **Valeur :** 0.9334
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 368

### S(t)
- Patterns détectés : 12
- Corrélation moyenne : 0.733
- Corrélation max : 0.834

### C(t)
- Patterns détectés : 43
- Corrélation moyenne : 0.766
- Corrélation max : 0.964

### A_mean(t)
- Patterns détectés : 97
- Corrélation moyenne : 0.666
- Corrélation max : 0.707

### f_mean(t)
- Patterns détectés : 98
- Corrélation moyenne : 0.925
- Corrélation max : 0.937

### entropy_S
- Patterns détectés : 14
- Corrélation moyenne : 0.724
- Corrélation max : 0.911

### effort(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.759
- Corrélation max : 0.817

### mean_high_effort
- Patterns détectés : 72
- Corrélation moyenne : 0.683
- Corrélation max : 0.940

### d_effort_dt
- Patterns détectés : 14
- Corrélation moyenne : 0.782
- Corrélation max : 0.924

### mean_abs_error
- Patterns détectés : 14
- Corrélation moyenne : 0.724
- Corrélation max : 0.851

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
