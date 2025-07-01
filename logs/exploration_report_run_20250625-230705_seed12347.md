# Rapport d'exploration FPS

**Run ID :** run_20250625-230705_seed12347
**Date :** 2025-06-25 23:07:09
**Total événements :** 1299

## Résumé par type d'événement

- **anomaly** : 226 événements
- **harmonic_emergence** : 201 événements
- **phase_cycle** : 814 événements
- **fractal_pattern** : 58 événements

## Anomaly

### 1. t=2652-2701
- **Métrique :** effort(t)
- **Valeur :** 1284159604738.2532
- **Sévérité :** high

### 2. t=2653-2702
- **Métrique :** d_effort_dt
- **Valeur :** 27702931.3534
- **Sévérité :** high

### 3. t=2654-2659
- **Métrique :** d_effort_dt
- **Valeur :** 193599.2917
- **Sévérité :** high

### 4. t=1799-1802
- **Métrique :** d_effort_dt
- **Valeur :** 2308.0869
- **Sévérité :** high

### 5. t=1800-1802
- **Métrique :** d_effort_dt
- **Valeur :** 1597.5016
- **Sévérité :** high

## Harmonic Emergence

### 1. t=1340-1433
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=1540-1633
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=120-213
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=150-243
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=320-413
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=1825-1830
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=1826-1831
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=1827-1832
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=1828-1833
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=1800-1900
- **Métrique :** effort(t)
- **Valeur :** 0.9471
- **Sévérité :** high
- **scale :** 10/100

### 2. t=1400-1500
- **Métrique :** mean_high_effort
- **Valeur :** 0.9313
- **Sévérité :** high
- **scale :** 10/100

### 3. t=1900-2000
- **Métrique :** mean_abs_error
- **Valeur :** 0.9148
- **Sévérité :** high
- **scale :** 10/100

### 4. t=1850-1950
- **Métrique :** mean_abs_error
- **Valeur :** 0.9146
- **Sévérité :** high
- **scale :** 10/100

### 5. t=1950-2050
- **Métrique :** mean_abs_error
- **Valeur :** 0.9129
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 58

### S(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.711
- Corrélation max : 0.778

### A_mean(t)
- Patterns détectés : 6
- Corrélation moyenne : 0.713
- Corrélation max : 0.806

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.702
- Corrélation max : 0.702

### effort(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.825
- Corrélation max : 0.947

### mean_high_effort
- Patterns détectés : 39
- Corrélation moyenne : 0.711
- Corrélation max : 0.931

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.766
- Corrélation max : 0.771

### mean_abs_error
- Patterns détectés : 5
- Corrélation moyenne : 0.843
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
