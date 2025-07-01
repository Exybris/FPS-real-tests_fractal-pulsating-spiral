# Rapport d'exploration FPS

**Run ID :** run_20250623-030641_seed12345
**Date :** 2025-06-23 03:06:42
**Total événements :** 206

## Résumé par type d'événement

- **anomaly** : 171 événements
- **harmonic_emergence** : 28 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=98-147
- **Métrique :** mean_abs_error
- **Valeur :** 288285.5714
- **Sévérité :** high

### 2. t=99-148
- **Métrique :** mean_abs_error
- **Valeur :** 260763.2883
- **Sévérité :** high

### 3. t=100-149
- **Métrique :** mean_abs_error
- **Valeur :** 151608.1525
- **Sévérité :** high

### 4. t=101-150
- **Métrique :** mean_abs_error
- **Valeur :** 143712.6022
- **Sévérité :** high

### 5. t=103-152
- **Métrique :** mean_abs_error
- **Valeur :** 72637.4821
- **Sévérité :** high

## Harmonic Emergence

### 1. t=120-213
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=140-233
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=160-253
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=230-323
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=100-122
- **Métrique :** S(t)
- **Valeur :** 22.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.8996
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8245
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.8182
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.7419
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.6936
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.806
- Corrélation max : 0.900

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.747
- Corrélation max : 0.818

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.742
- Corrélation max : 0.742

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
