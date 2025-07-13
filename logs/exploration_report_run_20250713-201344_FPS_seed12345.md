# Rapport d'exploration FPS

**Run ID :** run_20250713-201344_FPS_seed12345
**Date :** 2025-07-13 20:14:40
**Total événements :** 1894

## Résumé par type d'événement

- **anomaly** : 1580 événements
- **harmonic_emergence** : 39 événements
- **phase_cycle** : 21 événements
- **fractal_pattern** : 254 événements

## Anomaly

### 1. t=501-550
- **Métrique :** A_mean(t)
- **Valeur :** 22.1818
- **Sévérité :** high

### 2. t=400-449
- **Métrique :** A_mean(t)
- **Valeur :** 21.4470
- **Sévérité :** high

### 3. t=600-649
- **Métrique :** A_mean(t)
- **Valeur :** 20.0983
- **Sévérité :** high

### 4. t=3501-3550
- **Métrique :** A_mean(t)
- **Valeur :** 20.0843
- **Sévérité :** high

### 5. t=2601-2650
- **Métrique :** A_mean(t)
- **Valeur :** 20.0696
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=140-233
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=3150-3243
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=160-253
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

### 3. t=3-17
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 4. t=6-17
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

### 5. t=7-17
- **Métrique :** S(t)
- **Valeur :** 10.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=200-300
- **Métrique :** effort(t)
- **Valeur :** 0.8183
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** mean_abs_error
- **Valeur :** 0.8103
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=8650-8750
- **Métrique :** mean_abs_error
- **Valeur :** 0.8100
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=1900-2000
- **Métrique :** entropy_S
- **Valeur :** 0.8082
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=6350-6450
- **Métrique :** mean_abs_error
- **Valeur :** 0.8008
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 254

### S(t)
- Patterns détectés : 15
- Corrélation moyenne : 0.693
- Corrélation max : 0.778

### A_mean(t)
- Patterns détectés : 97
- Corrélation moyenne : 0.667
- Corrélation max : 0.793

### entropy_S
- Patterns détectés : 6
- Corrélation moyenne : 0.734
- Corrélation max : 0.808

### effort(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.738
- Corrélation max : 0.818

### mean_high_effort
- Patterns détectés : 114
- Corrélation moyenne : 0.656
- Corrélation max : 0.688

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.709
- Corrélation max : 0.721

### mean_abs_error
- Patterns détectés : 17
- Corrélation moyenne : 0.729
- Corrélation max : 0.810

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
