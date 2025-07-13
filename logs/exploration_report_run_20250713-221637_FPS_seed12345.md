# Rapport d'exploration FPS

**Run ID :** run_20250713-221637_FPS_seed12345
**Date :** 2025-07-13 22:16:41
**Total événements :** 401

## Résumé par type d'événement

- **anomaly** : 302 événements
- **harmonic_emergence** : 14 événements
- **phase_cycle** : 14 événements
- **fractal_pattern** : 71 événements

## Anomaly

### 1. t=600-649
- **Métrique :** A_mean(t)
- **Valeur :** 22.3879
- **Sévérité :** high

### 2. t=601-650
- **Métrique :** A_mean(t)
- **Valeur :** 21.2084
- **Sévérité :** high

### 3. t=1000-1049
- **Métrique :** A_mean(t)
- **Valeur :** 19.9508
- **Sévérité :** high

### 4. t=1700-1749
- **Métrique :** A_mean(t)
- **Valeur :** 19.9358
- **Sévérité :** high

### 5. t=1001-1050
- **Métrique :** A_mean(t)
- **Valeur :** 19.9177
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=170-263
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

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

### 1. t=1200-1300
- **Métrique :** S(t)
- **Valeur :** 0.8118
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=450-550
- **Métrique :** entropy_S
- **Valeur :** 0.7810
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=600-700
- **Métrique :** mean_abs_error
- **Valeur :** 0.7649
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=950-1050
- **Métrique :** S(t)
- **Valeur :** 0.7367
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=1850-1950
- **Métrique :** mean_abs_error
- **Valeur :** 0.7250
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 71

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.751
- Corrélation max : 0.812

### A_mean(t)
- Patterns détectés : 19
- Corrélation moyenne : 0.665
- Corrélation max : 0.679

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.714
- Corrélation max : 0.781

### effort(t)
- Patterns détectés : 8
- Corrélation moyenne : 0.674
- Corrélation max : 0.706

### mean_high_effort
- Patterns détectés : 35
- Corrélation moyenne : 0.663
- Corrélation max : 0.685

### mean_abs_error
- Patterns détectés : 3
- Corrélation moyenne : 0.717
- Corrélation max : 0.765

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
