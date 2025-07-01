# Rapport d'exploration FPS

**Run ID :** run_20250629-211143_seed12345
**Date :** 2025-06-29 21:11:44
**Total événements :** 134

## Résumé par type d'événement

- **anomaly** : 75 événements
- **harmonic_emergence** : 36 événements
- **phase_cycle** : 14 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=281-330
- **Métrique :** mean_high_effort
- **Valeur :** 68.6363
- **Sévérité :** high

### 2. t=282-331
- **Métrique :** mean_high_effort
- **Valeur :** 65.1567
- **Sévérité :** high

### 3. t=283-332
- **Métrique :** mean_high_effort
- **Valeur :** 61.9632
- **Sévérité :** high

### 4. t=284-333
- **Métrique :** mean_high_effort
- **Valeur :** 57.3121
- **Sévérité :** high

### 5. t=285-334
- **Métrique :** mean_high_effort
- **Valeur :** 53.9121
- **Sévérité :** high

## Harmonic Emergence

### 1. t=390-483
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=290-383
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=340-433
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=400-493
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-13
- **Métrique :** S(t)
- **Valeur :** 12.0000
- **Sévérité :** medium

### 2. t=351-359
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=377-384
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 4. t=378-384
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 5. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9322
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** d_effort_dt
- **Valeur :** 0.8597
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.8063
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.7838
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.7836
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.671
- Corrélation max : 0.675

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.651
- Corrélation max : 0.651

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.795
- Corrélation max : 0.806

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.788
- Corrélation max : 0.932

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.860
- Corrélation max : 0.860

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
