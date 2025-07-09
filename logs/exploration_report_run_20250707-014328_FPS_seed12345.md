# Rapport d'exploration FPS

**Run ID :** run_20250707-014328_FPS_seed12345
**Date :** 2025-07-07 01:43:28
**Total événements :** 100

## Résumé par type d'événement

- **anomaly** : 65 événements
- **harmonic_emergence** : 24 événements
- **phase_cycle** : 4 événements
- **fractal_pattern** : 7 événements

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

### 1. t=230-323
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=240-333
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=250-343
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

### 1. t=378-386
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 2. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=376-381
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=398-403
- **Métrique :** S(t)
- **Valeur :** 5.0000
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

### 3. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.8251
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.7950
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** entropy_S
- **Valeur :** 0.7165
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.656
- Corrélation max : 0.656

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.771
- Corrélation max : 0.825

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
