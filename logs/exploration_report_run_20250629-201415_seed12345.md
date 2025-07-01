# Rapport d'exploration FPS

**Run ID :** run_20250629-201415_seed12345
**Date :** 2025-06-29 20:14:15
**Total événements :** 201

## Résumé par type d'événement

- **anomaly** : 59 événements
- **harmonic_emergence** : 35 événements
- **phase_cycle** : 101 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=84-133
- **Métrique :** mean_high_effort
- **Valeur :** 70.2896
- **Sévérité :** high

### 2. t=85-134
- **Métrique :** mean_high_effort
- **Valeur :** 65.7871
- **Sévérité :** high

### 3. t=220-269
- **Métrique :** mean_high_effort
- **Valeur :** 58.8758
- **Sévérité :** high

### 4. t=86-135
- **Métrique :** mean_high_effort
- **Valeur :** 50.3700
- **Sévérité :** high

### 5. t=221-270
- **Métrique :** mean_high_effort
- **Valeur :** 47.9462
- **Sévérité :** high

## Harmonic Emergence

### 1. t=240-333
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=250-343
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=280-373
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=340-433
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-41
- **Métrique :** S(t)
- **Valeur :** 40.0000
- **Sévérité :** medium

### 2. t=2-31
- **Métrique :** S(t)
- **Valeur :** 29.0000
- **Sévérité :** medium

### 3. t=16-41
- **Métrique :** S(t)
- **Valeur :** 25.0000
- **Sévérité :** medium

### 4. t=17-41
- **Métrique :** S(t)
- **Valeur :** 24.0000
- **Sévérité :** medium

### 5. t=11-34
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.9000
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8784
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8545
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8242
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** A_mean(t)
- **Valeur :** 0.7074
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.707
- Corrélation max : 0.707

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.864
- Corrélation max : 0.900

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.671
- Corrélation max : 0.671

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
