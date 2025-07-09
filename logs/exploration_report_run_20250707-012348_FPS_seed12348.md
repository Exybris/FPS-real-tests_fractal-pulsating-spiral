# Rapport d'exploration FPS

**Run ID :** run_20250707-012348_FPS_seed12348
**Date :** 2025-07-07 01:23:48
**Total événements :** 91

## Résumé par type d'événement

- **anomaly** : 61 événements
- **harmonic_emergence** : 18 événements
- **phase_cycle** : 5 événements
- **fractal_pattern** : 7 événements

## Anomaly

### 1. t=130-132
- **Métrique :** effort(t)
- **Valeur :** 210.4057
- **Sévérité :** high

### 2. t=131-134
- **Métrique :** d_effort_dt
- **Valeur :** 189.4004
- **Sévérité :** high

### 3. t=300-349
- **Métrique :** mean_high_effort
- **Valeur :** 42.3774
- **Sévérité :** high

### 4. t=301-350
- **Métrique :** mean_high_effort
- **Valeur :** 38.7294
- **Sévérité :** high

### 5. t=302-351
- **Métrique :** mean_high_effort
- **Valeur :** 35.8234
- **Sévérité :** high

## Harmonic Emergence

### 1. t=350-443
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=280-373
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=310-403
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-12
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

### 2. t=382-389
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 3. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=381-386
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9323
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8314
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.6943
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6862
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.6815
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 7

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.658
- Corrélation max : 0.666

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.694
- Corrélation max : 0.694

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.783
- Corrélation max : 0.932

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
