# Rapport d'exploration FPS

**Run ID :** run_20250630-193832_seed12345
**Date :** 2025-06-30 19:38:33
**Total événements :** 142

## Résumé par type d'événement

- **anomaly** : 82 événements
- **harmonic_emergence** : 34 événements
- **phase_cycle** : 16 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=239-288
- **Métrique :** effort(t)
- **Valeur :** 216.7268
- **Sévérité :** high

### 2. t=240-289
- **Métrique :** effort(t)
- **Valeur :** 194.7227
- **Sévérité :** high

### 3. t=250-299
- **Métrique :** mean_high_effort
- **Valeur :** 150.4872
- **Sévérité :** high

### 4. t=254-303
- **Métrique :** mean_high_effort
- **Valeur :** 146.4119
- **Sévérité :** high

### 5. t=253-302
- **Métrique :** mean_high_effort
- **Valeur :** 140.8749
- **Sévérité :** high

## Harmonic Emergence

### 1. t=320-413
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=350-443
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=200-293
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 2. t=385-395
- **Métrique :** S(t)
- **Valeur :** 10.0000
- **Sévérité :** medium

### 3. t=348-356
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 4. t=351-359
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=381-389
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9273
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9096
- **Sévérité :** high
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8927
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.8300
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.8099
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.676
- Corrélation max : 0.694

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.752
- Corrélation max : 0.810

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.843
- Corrélation max : 0.927

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
