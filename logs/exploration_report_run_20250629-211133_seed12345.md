# Rapport d'exploration FPS

**Run ID :** run_20250629-211133_seed12345
**Date :** 2025-06-29 21:11:33
**Total événements :** 141

## Résumé par type d'événement

- **anomaly** : 84 événements
- **harmonic_emergence** : 36 événements
- **phase_cycle** : 12 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=280-329
- **Métrique :** mean_high_effort
- **Valeur :** 91.6574
- **Sévérité :** high

### 2. t=281-330
- **Métrique :** mean_high_effort
- **Valeur :** 86.3838
- **Sévérité :** high

### 3. t=282-331
- **Métrique :** mean_high_effort
- **Valeur :** 81.4264
- **Sévérité :** high

### 4. t=283-332
- **Métrique :** mean_high_effort
- **Valeur :** 76.3464
- **Sévérité :** high

### 5. t=284-333
- **Métrique :** mean_high_effort
- **Valeur :** 69.7981
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=340-433
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=290-383
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=310-403
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 2. t=349-357
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=350-357
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 4. t=351-358
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 5. t=2-8
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9272
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8468
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.7813
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7348
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.6772
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.686
- Corrélation max : 0.735

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.657
- Corrélation max : 0.657

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.729
- Corrélation max : 0.781

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.816
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
