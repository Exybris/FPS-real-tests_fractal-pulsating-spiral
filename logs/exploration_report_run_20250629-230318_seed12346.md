# Rapport d'exploration FPS

**Run ID :** run_20250629-230318_seed12346
**Date :** 2025-06-29 23:03:18
**Total événements :** 164

## Résumé par type d'événement

- **anomaly** : 97 événements
- **harmonic_emergence** : 34 événements
- **phase_cycle** : 23 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=242-291
- **Métrique :** effort(t)
- **Valeur :** 528.9495
- **Sévérité :** high

### 2. t=247-296
- **Métrique :** mean_high_effort
- **Valeur :** 171.7269
- **Sévérité :** high

### 3. t=248-297
- **Métrique :** mean_high_effort
- **Valeur :** 166.0127
- **Sévérité :** high

### 4. t=249-298
- **Métrique :** mean_high_effort
- **Valeur :** 159.2094
- **Sévérité :** high

### 5. t=250-299
- **Métrique :** mean_high_effort
- **Valeur :** 149.0012
- **Sévérité :** high

## Harmonic Emergence

### 1. t=200-293
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=270-363
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=280-373
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=385-400
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 2. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 3. t=350-359
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 4. t=386-395
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 5. t=346-354
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9101
- **Sévérité :** high
- **scale :** 10/100

### 2. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8798
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.8463
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.8199
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.7080
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.686
- Corrélation max : 0.708

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.648
- Corrélation max : 0.648

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.820
- Corrélation max : 0.820

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.794
- Corrélation max : 0.910

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.655
- Corrélation max : 0.655

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
