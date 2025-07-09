# Rapport d'exploration FPS

**Run ID :** run_20250707-004621_FPS_seed12348
**Date :** 2025-07-07 00:46:22
**Total événements :** 106

## Résumé par type d'événement

- **anomaly** : 71 événements
- **harmonic_emergence** : 19 événements
- **phase_cycle** : 5 événements
- **fractal_pattern** : 11 événements

## Anomaly

### 1. t=130-135
- **Métrique :** effort(t)
- **Valeur :** 210.4057
- **Sévérité :** high

### 2. t=131-137
- **Métrique :** d_effort_dt
- **Valeur :** 189.4004
- **Sévérité :** high

### 3. t=303-352
- **Métrique :** mean_high_effort
- **Valeur :** 36.4176
- **Sévérité :** high

### 4. t=304-353
- **Métrique :** mean_high_effort
- **Valeur :** 33.2858
- **Sévérité :** high

### 5. t=305-354
- **Métrique :** mean_high_effort
- **Valeur :** 29.5327
- **Sévérité :** high

## Harmonic Emergence

### 1. t=280-373
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

### 2. t=382-389
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 3. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 4. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 5. t=381-386
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9372
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8147
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7611
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.7508
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** d_effort_dt
- **Valeur :** 0.7313
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 11

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.662
- Corrélation max : 0.666

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.761
- Corrélation max : 0.761

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.658
- Corrélation max : 0.666

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.773
- Corrélation max : 0.937

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.731
- Corrélation max : 0.731

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
