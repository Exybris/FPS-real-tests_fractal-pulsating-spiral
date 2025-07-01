# Rapport d'exploration FPS

**Run ID :** run_20250629-211128_seed12346
**Date :** 2025-06-29 21:11:28
**Total événements :** 110

## Résumé par type d'événement

- **anomaly** : 54 événements
- **harmonic_emergence** : 34 événements
- **phase_cycle** : 17 événements
- **fractal_pattern** : 5 événements

## Anomaly

### 1. t=283-332
- **Métrique :** mean_high_effort
- **Valeur :** 47.4589
- **Sévérité :** high

### 2. t=284-333
- **Métrique :** mean_high_effort
- **Valeur :** 44.6884
- **Sévérité :** high

### 3. t=285-334
- **Métrique :** mean_high_effort
- **Valeur :** 42.1976
- **Sévérité :** high

### 4. t=286-335
- **Métrique :** mean_high_effort
- **Valeur :** 40.6992
- **Sévérité :** high

### 5. t=288-337
- **Métrique :** mean_high_effort
- **Valeur :** 36.6257
- **Sévérité :** high

## Harmonic Emergence

### 1. t=320-413
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=340-433
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=360-453
- **Métrique :** S(t)
- **Valeur :** 4.0000
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

### 1. t=347-355
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 2. t=351-359
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=381-389
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 4. t=386-394
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=1-8
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9152
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9101
- **Sévérité :** high
- **scale :** 10/100

### 3. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.7950
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6840
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** A_mean(t)
- **Valeur :** 0.6561
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 5

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.656
- Corrélation max : 0.656

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.795
- Corrélation max : 0.795

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.836
- Corrélation max : 0.915

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
