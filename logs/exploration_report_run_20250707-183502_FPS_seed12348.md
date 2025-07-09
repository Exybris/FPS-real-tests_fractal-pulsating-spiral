# Rapport d'exploration FPS

**Run ID :** run_20250707-183502_FPS_seed12348
**Date :** 2025-07-07 18:35:17
**Total événements :** 994

## Résumé par type d'événement

- **anomaly** : 741 événements
- **harmonic_emergence** : 104 événements
- **phase_cycle** : 15 événements
- **fractal_pattern** : 134 événements

## Anomaly

### 1. t=2908-2957
- **Métrique :** mean_high_effort
- **Valeur :** 59.4299
- **Sévérité :** high

### 2. t=2909-2958
- **Métrique :** mean_high_effort
- **Valeur :** 57.2438
- **Sévérité :** high

### 3. t=2910-2959
- **Métrique :** mean_high_effort
- **Valeur :** 54.2650
- **Sévérité :** high

### 4. t=2911-2960
- **Métrique :** mean_high_effort
- **Valeur :** 49.8920
- **Sévérité :** high

### 5. t=2912-2961
- **Métrique :** mean_high_effort
- **Valeur :** 46.2436
- **Sévérité :** high

## Harmonic Emergence

### 1. t=3640-3733
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

### 4. t=2130-2223
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=2850-2943
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-17
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 2. t=2-17
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 3. t=3-16
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 4. t=7-17
- **Métrique :** S(t)
- **Valeur :** 10.0000
- **Sévérité :** medium

### 5. t=8-17
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=3100-3200
- **Métrique :** mean_high_effort
- **Valeur :** 0.9338
- **Sévérité :** high
- **scale :** 10/100

### 2. t=3000-3100
- **Métrique :** mean_high_effort
- **Valeur :** 0.9202
- **Sévérité :** high
- **scale :** 10/100

### 3. t=3700-3800
- **Métrique :** mean_high_effort
- **Valeur :** 0.9195
- **Sévérité :** high
- **scale :** 10/100

### 4. t=2900-3000
- **Métrique :** mean_high_effort
- **Valeur :** 0.9180
- **Sévérité :** high
- **scale :** 10/100

### 5. t=3300-3400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9175
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 134

### S(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.692
- Corrélation max : 0.729

### A_mean(t)
- Patterns détectés : 29
- Corrélation moyenne : 0.665
- Corrélation max : 0.714

### f_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.744
- Corrélation max : 0.744

### entropy_S
- Patterns détectés : 7
- Corrélation moyenne : 0.711
- Corrélation max : 0.752

### effort(t)
- Patterns détectés : 9
- Corrélation moyenne : 0.701
- Corrélation max : 0.840

### mean_high_effort
- Patterns détectés : 80
- Corrélation moyenne : 0.703
- Corrélation max : 0.934

### d_effort_dt
- Patterns détectés : 4
- Corrélation moyenne : 0.740
- Corrélation max : 0.811

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
