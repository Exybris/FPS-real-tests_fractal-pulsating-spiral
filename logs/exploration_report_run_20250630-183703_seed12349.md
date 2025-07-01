# Rapport d'exploration FPS

**Run ID :** run_20250630-183703_seed12349
**Date :** 2025-06-30 18:37:03
**Total événements :** 121

## Résumé par type d'événement

- **anomaly** : 63 événements
- **harmonic_emergence** : 36 événements
- **phase_cycle** : 14 événements
- **fractal_pattern** : 8 événements

## Anomaly

### 1. t=238-287
- **Métrique :** effort(t)
- **Valeur :** 519.4889
- **Sévérité :** high

### 2. t=239-288
- **Métrique :** effort(t)
- **Valeur :** 478.7522
- **Sévérité :** high

### 3. t=240-289
- **Métrique :** effort(t)
- **Valeur :** 402.8150
- **Sévérité :** high

### 4. t=241-290
- **Métrique :** effort(t)
- **Valeur :** 337.5627
- **Sévérité :** high

### 5. t=242-291
- **Métrique :** effort(t)
- **Valeur :** 311.1782
- **Sévérité :** high

## Harmonic Emergence

### 1. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=320-413
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=330-423
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=350-443
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-15
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 2. t=396-404
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=351-358
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 4. t=395-402
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 5. t=397-404
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9193
- **Sévérité :** high
- **scale :** 10/100

### 2. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8977
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=300-400
- **Métrique :** d_effort_dt
- **Valeur :** 0.8495
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8109
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.8059
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 8

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.706
- Corrélation max : 0.711

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.821
- Corrélation max : 0.919

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.849
- Corrélation max : 0.849

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
