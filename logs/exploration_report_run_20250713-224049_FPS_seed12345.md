# Rapport d'exploration FPS

**Run ID :** run_20250713-224049_FPS_seed12345
**Date :** 2025-07-13 22:40:53
**Total événements :** 390

## Résumé par type d'événement

- **anomaly** : 304 événements
- **harmonic_emergence** : 15 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 70 événements

## Anomaly

### 1. t=600-649
- **Métrique :** A_mean(t)
- **Valeur :** 20.7983
- **Sévérité :** high

### 2. t=201-250
- **Métrique :** A_mean(t)
- **Valeur :** 20.3156
- **Sévérité :** high

### 3. t=200-249
- **Métrique :** A_mean(t)
- **Valeur :** 20.2785
- **Sévérité :** high

### 4. t=601-650
- **Métrique :** A_mean(t)
- **Valeur :** 20.1839
- **Sévérité :** high

### 5. t=1000-1049
- **Métrique :** A_mean(t)
- **Valeur :** 19.9715
- **Sévérité :** high

## Harmonic Emergence

### 1. t=160-253
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=170-263
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=200-293
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=220-313
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=240-333
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1202-1207
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=1200-1300
- **Métrique :** S(t)
- **Valeur :** 0.8171
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=450-550
- **Métrique :** entropy_S
- **Valeur :** 0.7876
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=600-700
- **Métrique :** mean_abs_error
- **Valeur :** 0.7544
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=950-1050
- **Métrique :** S(t)
- **Valeur :** 0.7303
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.7105
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 70

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.749
- Corrélation max : 0.817

### A_mean(t)
- Patterns détectés : 17
- Corrélation moyenne : 0.669
- Corrélation max : 0.711

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.714
- Corrélation max : 0.788

### effort(t)
- Patterns détectés : 11
- Corrélation moyenne : 0.671
- Corrélation max : 0.704

### mean_high_effort
- Patterns détectés : 35
- Corrélation moyenne : 0.664
- Corrélation max : 0.688

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.754
- Corrélation max : 0.754

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
