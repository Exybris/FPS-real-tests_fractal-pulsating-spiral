# Rapport d'exploration FPS

**Run ID :** run_20250707-015726_FPS_seed12346
**Date :** 2025-07-07 01:57:27
**Total événements :** 112

## Résumé par type d'événement

- **anomaly** : 83 événements
- **harmonic_emergence** : 16 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=130-134
- **Métrique :** effort(t)
- **Valeur :** 93.2514
- **Sévérité :** high

### 2. t=131-134
- **Métrique :** d_effort_dt
- **Valeur :** 49.6902
- **Sévérité :** high

### 3. t=412-461
- **Métrique :** mean_high_effort
- **Valeur :** 31.2971
- **Sévérité :** high

### 4. t=98-147
- **Métrique :** A_mean(t)
- **Valeur :** 28.2162
- **Sévérité :** high

### 5. t=413-462
- **Métrique :** mean_high_effort
- **Valeur :** 26.8925
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=270-363
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=330-423
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=340-433
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-12
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

### 2. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9318
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7479
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.7301
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** effort(t)
- **Valeur :** 0.7103
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** d_effort_dt
- **Valeur :** 0.7094
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.666
- Corrélation max : 0.669

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.701
- Corrélation max : 0.748

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.710
- Corrélation max : 0.710

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.782
- Corrélation max : 0.932

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.707
- Corrélation max : 0.709

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
