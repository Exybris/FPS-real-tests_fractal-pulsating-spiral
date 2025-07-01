# Rapport d'exploration FPS

**Run ID :** run_20250629-211149_seed12347
**Date :** 2025-06-29 21:11:50
**Total événements :** 141

## Résumé par type d'événement

- **anomaly** : 88 événements
- **harmonic_emergence** : 31 événements
- **phase_cycle** : 11 événements
- **fractal_pattern** : 11 événements

## Anomaly

### 1. t=261-310
- **Métrique :** mean_high_effort
- **Valeur :** 293.0458
- **Sévérité :** high

### 2. t=262-311
- **Métrique :** mean_high_effort
- **Valeur :** 262.4475
- **Sévérité :** high

### 3. t=263-312
- **Métrique :** mean_high_effort
- **Valeur :** 244.8434
- **Sévérité :** high

### 4. t=264-313
- **Métrique :** mean_high_effort
- **Valeur :** 217.6272
- **Sévérité :** high

### 5. t=265-314
- **Métrique :** mean_high_effort
- **Valeur :** 208.6214
- **Sévérité :** high

## Harmonic Emergence

### 1. t=340-433
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=360-453
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=310-403
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

### 1. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 2. t=344-352
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=343-350
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 4. t=345-352
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 5. t=350-357
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9289
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.7790
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7770
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.7362
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** d_effort_dt
- **Valeur :** 0.7273
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 11

### A_mean(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.689
- Corrélation max : 0.777

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.689
- Corrélation max : 0.714

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.784
- Corrélation max : 0.929

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.727
- Corrélation max : 0.727

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
