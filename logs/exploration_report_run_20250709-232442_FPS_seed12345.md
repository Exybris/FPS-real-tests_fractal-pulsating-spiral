# Rapport d'exploration FPS

**Run ID :** run_20250709-232442_FPS_seed12345
**Date :** 2025-07-09 23:24:43
**Total événements :** 62

## Résumé par type d'événement

- **anomaly** : 40 événements
- **harmonic_emergence** : 11 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 8 événements

## Anomaly

### 1. t=300-349
- **Métrique :** mean_high_effort
- **Valeur :** 37.3835
- **Sévérité :** high

### 2. t=301-350
- **Métrique :** mean_high_effort
- **Valeur :** 35.1841
- **Sévérité :** high

### 3. t=302-351
- **Métrique :** mean_high_effort
- **Valeur :** 32.1918
- **Sévérité :** high

### 4. t=303-352
- **Métrique :** mean_high_effort
- **Valeur :** 28.5473
- **Sévérité :** high

### 5. t=304-353
- **Métrique :** mean_high_effort
- **Valeur :** 24.5125
- **Sévérité :** high

## Harmonic Emergence

### 1. t=270-363
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=330-423
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1-17
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 2. t=2-8
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9449
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.8939
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.8364
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** d_effort_dt
- **Valeur :** 0.8069
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** d_effort_dt
- **Valeur :** 0.8019
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 8

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.865
- Corrélation max : 0.894

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.769
- Corrélation max : 0.945

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.804
- Corrélation max : 0.807

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
