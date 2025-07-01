# Rapport d'exploration FPS

**Run ID :** run_20250629-185608_seed12346
**Date :** 2025-06-29 18:56:08
**Total événements :** 248

## Résumé par type d'événement

- **anomaly** : 99 événements
- **harmonic_emergence** : 20 événements
- **phase_cycle** : 124 événements
- **fractal_pattern** : 5 événements

## Anomaly

### 1. t=443-492
- **Métrique :** d_effort_dt
- **Valeur :** 3147382.1060
- **Sévérité :** high

### 2. t=442-491
- **Métrique :** effort(t)
- **Valeur :** 2059314.1763
- **Sévérité :** high

### 3. t=306-310
- **Métrique :** d_effort_dt
- **Valeur :** 82469.6155
- **Sévérité :** high

### 4. t=444-447
- **Métrique :** d_effort_dt
- **Valeur :** 21244.7165
- **Sévérité :** high

### 5. t=307-309
- **Métrique :** d_effort_dt
- **Valeur :** 18779.2354
- **Sévérité :** high

## Harmonic Emergence

### 1. t=250-343
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=280-373
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=350-443
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=240-333
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=270-363
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=305-310
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=306-311
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=307-312
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=308-313
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8771
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.8147
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7385
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** mean_abs_error
- **Valeur :** 0.6672
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6507
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 5

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.877
- Corrélation max : 0.877

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.815
- Corrélation max : 0.815

### mean_high_effort
- Patterns détectés : 2
- Corrélation moyenne : 0.695
- Corrélation max : 0.739

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
