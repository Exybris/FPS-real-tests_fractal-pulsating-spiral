# Rapport d'exploration FPS

**Run ID :** run_20250629-201650_seed12345
**Date :** 2025-06-29 20:16:51
**Total événements :** 129

## Résumé par type d'événement

- **anomaly** : 91 événements
- **harmonic_emergence** : 32 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 5 événements

## Anomaly

### 1. t=92-141
- **Métrique :** mean_abs_error
- **Valeur :** 116.0214
- **Sévérité :** high

### 2. t=93-142
- **Métrique :** mean_abs_error
- **Valeur :** 104.9838
- **Sévérité :** high

### 3. t=94-143
- **Métrique :** mean_abs_error
- **Valeur :** 98.8619
- **Sévérité :** high

### 4. t=95-144
- **Métrique :** mean_abs_error
- **Valeur :** 81.4482
- **Sévérité :** high

### 5. t=96-145
- **Métrique :** mean_abs_error
- **Valeur :** 62.6790
- **Sévérité :** high

## Harmonic Emergence

### 1. t=280-373
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=70-163
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=80-173
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=100-193
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=120-213
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=341-346
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9267
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8191
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** mean_abs_error
- **Valeur :** 0.7291
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6805
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.6448
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 5

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.645
- Corrélation max : 0.645

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.809
- Corrélation max : 0.927

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.729
- Corrélation max : 0.729

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
