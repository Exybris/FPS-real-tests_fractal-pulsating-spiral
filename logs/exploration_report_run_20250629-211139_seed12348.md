# Rapport d'exploration FPS

**Run ID :** run_20250629-211139_seed12348
**Date :** 2025-06-29 21:11:40
**Total événements :** 134

## Résumé par type d'événement

- **anomaly** : 81 événements
- **harmonic_emergence** : 38 événements
- **phase_cycle** : 9 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=283-332
- **Métrique :** mean_high_effort
- **Valeur :** 292.8267
- **Sévérité :** high

### 2. t=284-333
- **Métrique :** mean_high_effort
- **Valeur :** 256.0508
- **Sévérité :** high

### 3. t=285-334
- **Métrique :** mean_high_effort
- **Valeur :** 227.6630
- **Sévérité :** high

### 4. t=286-335
- **Métrique :** mean_high_effort
- **Valeur :** 203.0262
- **Sévérité :** high

### 5. t=287-336
- **Métrique :** mean_high_effort
- **Valeur :** 170.0694
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=340-433
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=290-383
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=80-173
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=100-193
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-13
- **Métrique :** S(t)
- **Valeur :** 12.0000
- **Sévérité :** medium

### 2. t=350-356
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 3. t=351-357
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 4. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9327
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8831
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.7820
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.7337
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** A_mean(t)
- **Valeur :** 0.6894
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.689
- Corrélation max : 0.689

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.782
- Corrélation max : 0.782

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.807
- Corrélation max : 0.933

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
