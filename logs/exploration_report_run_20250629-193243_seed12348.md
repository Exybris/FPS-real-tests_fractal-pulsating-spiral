# Rapport d'exploration FPS

**Run ID :** run_20250629-193243_seed12348
**Date :** 2025-06-29 19:32:43
**Total événements :** 265

## Résumé par type d'événement

- **anomaly** : 96 événements
- **harmonic_emergence** : 35 événements
- **phase_cycle** : 125 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=443-492
- **Métrique :** d_effort_dt
- **Valeur :** 13415.5550
- **Sévérité :** high

### 2. t=442-491
- **Métrique :** effort(t)
- **Valeur :** 9273.8990
- **Sévérité :** high

### 3. t=442-491
- **Métrique :** mean_abs_error
- **Valeur :** 5008.7523
- **Sévérité :** high

### 4. t=443-481
- **Métrique :** effort(t)
- **Valeur :** 132.2864
- **Sévérité :** high

### 5. t=444-455
- **Métrique :** d_effort_dt
- **Valeur :** 125.7601
- **Sévérité :** high

## Harmonic Emergence

### 1. t=290-383
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=90-183
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=220-313
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=260-353
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=350-443
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=305-310
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=306-311
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=307-312
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9167
- **Sévérité :** high
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8771
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.8103
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** entropy_S
- **Valeur :** 0.7982
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** entropy_S
- **Valeur :** 0.7541
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### S(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.752
- Corrélation max : 0.752

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.877
- Corrélation max : 0.877

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.739
- Corrélation max : 0.798

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.758
- Corrélation max : 0.810

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.917
- Corrélation max : 0.917

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.667
- Corrélation max : 0.667

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
