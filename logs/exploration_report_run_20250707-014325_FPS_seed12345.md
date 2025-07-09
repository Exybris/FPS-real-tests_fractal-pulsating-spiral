# Rapport d'exploration FPS

**Run ID :** run_20250707-014325_FPS_seed12345
**Date :** 2025-07-07 01:43:26
**Total événements :** 139

## Résumé par type d'événement

- **anomaly** : 96 événements
- **harmonic_emergence** : 22 événements
- **phase_cycle** : 9 événements
- **fractal_pattern** : 12 événements

## Anomaly

### 1. t=239-288
- **Métrique :** effort(t)
- **Valeur :** 216.7268
- **Sévérité :** high

### 2. t=240-289
- **Métrique :** effort(t)
- **Valeur :** 194.7227
- **Sévérité :** high

### 3. t=250-299
- **Métrique :** mean_high_effort
- **Valeur :** 150.4872
- **Sévérité :** high

### 4. t=254-303
- **Métrique :** mean_high_effort
- **Valeur :** 146.4119
- **Sévérité :** high

### 5. t=253-302
- **Métrique :** mean_high_effort
- **Valeur :** 140.8749
- **Sévérité :** high

## Harmonic Emergence

### 1. t=240-333
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=340-433
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=250-343
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=330-423
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=378-386
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 2. t=383-391
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 4. t=376-382
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 5. t=377-383
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.9273
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9096
- **Sévérité :** high
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8927
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.8300
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.8099
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 12

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.676
- Corrélation max : 0.694

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.762
- Corrélation max : 0.771

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.752
- Corrélation max : 0.810

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.843
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
