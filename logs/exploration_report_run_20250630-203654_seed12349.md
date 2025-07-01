# Rapport d'exploration FPS

**Run ID :** run_20250630-203654_seed12349
**Date :** 2025-06-30 20:37:01
**Total événements :** 991

## Résumé par type d'événement

- **anomaly** : 392 événements
- **harmonic_emergence** : 356 événements
- **phase_cycle** : 142 événements
- **fractal_pattern** : 101 événements

## Anomaly

### 1. t=2015-2064
- **Métrique :** mean_high_effort
- **Valeur :** 30.8634
- **Sévérité :** high

### 2. t=2107-2156
- **Métrique :** mean_high_effort
- **Valeur :** 29.1365
- **Sévérité :** high

### 3. t=2016-2065
- **Métrique :** mean_high_effort
- **Valeur :** 28.7729
- **Sévérité :** high

### 4. t=2108-2157
- **Métrique :** mean_high_effort
- **Valeur :** 27.3546
- **Sévérité :** high

### 5. t=2017-2066
- **Métrique :** mean_high_effort
- **Valeur :** 26.4803
- **Sévérité :** high

## Harmonic Emergence

### 1. t=2230-2323
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=470-563
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=670-763
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=1240-1333
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=2944-2959
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 2. t=2946-2961
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 3. t=3068-3083
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 4. t=1-15
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 5. t=2676-2690
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=2000-2100
- **Métrique :** mean_high_effort
- **Valeur :** 0.9460
- **Sévérité :** high
- **scale :** 10/100

### 2. t=2600-2700
- **Métrique :** mean_high_effort
- **Valeur :** 0.9307
- **Sévérité :** high
- **scale :** 10/100

### 3. t=3100-3200
- **Métrique :** mean_high_effort
- **Valeur :** 0.9291
- **Sévérité :** high
- **scale :** 10/100

### 4. t=2700-2800
- **Métrique :** mean_high_effort
- **Valeur :** 0.9263
- **Sévérité :** high
- **scale :** 10/100

### 5. t=2500-2600
- **Métrique :** mean_high_effort
- **Valeur :** 0.9170
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 101

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.690
- Corrélation max : 0.705

### A_mean(t)
- Patterns détectés : 16
- Corrélation moyenne : 0.674
- Corrélation max : 0.769

### f_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.715
- Corrélation max : 0.715

### entropy_S
- Patterns détectés : 6
- Corrélation moyenne : 0.746
- Corrélation max : 0.903

### effort(t)
- Patterns détectés : 8
- Corrélation moyenne : 0.737
- Corrélation max : 0.817

### mean_high_effort
- Patterns détectés : 59
- Corrélation moyenne : 0.736
- Corrélation max : 0.946

### d_effort_dt
- Patterns détectés : 7
- Corrélation moyenne : 0.730
- Corrélation max : 0.816

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.785
- Corrélation max : 0.785

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
