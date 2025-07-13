# Rapport d'exploration FPS

**Run ID :** run_20250713-202421_FPS_seed12346
**Date :** 2025-07-13 20:25:20
**Total événements :** 2015

## Résumé par type d'événement

- **anomaly** : 1593 événements
- **harmonic_emergence** : 52 événements
- **phase_cycle** : 30 événements
- **fractal_pattern** : 340 événements

## Anomaly

### 1. t=600-649
- **Métrique :** A_mean(t)
- **Valeur :** 21.2470
- **Sévérité :** high

### 2. t=1000-1049
- **Métrique :** A_mean(t)
- **Valeur :** 20.1267
- **Sévérité :** high

### 3. t=1700-1749
- **Métrique :** A_mean(t)
- **Valeur :** 20.1146
- **Sévérité :** high

### 4. t=2301-2350
- **Métrique :** A_mean(t)
- **Valeur :** 20.0736
- **Sévérité :** high

### 5. t=3500-3549
- **Métrique :** A_mean(t)
- **Valeur :** 20.0639
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

### 3. t=4430-4523
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=100-193
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

### 4. t=7-17
- **Métrique :** S(t)
- **Valeur :** 10.0000
- **Sévérité :** medium

### 5. t=8-17
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=4850-4950
- **Métrique :** entropy_S
- **Valeur :** 0.8270
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=5650-5750
- **Métrique :** mean_abs_error
- **Valeur :** 0.8061
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=3700-3800
- **Métrique :** effort(t)
- **Valeur :** 0.8059
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=7850-7950
- **Métrique :** mean_abs_error
- **Valeur :** 0.7977
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=1200-1300
- **Métrique :** S(t)
- **Valeur :** 0.7965
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 340

### S(t)
- Patterns détectés : 13
- Corrélation moyenne : 0.716
- Corrélation max : 0.796

### A_mean(t)
- Patterns détectés : 97
- Corrélation moyenne : 0.665
- Corrélation max : 0.689

### entropy_S
- Patterns détectés : 15
- Corrélation moyenne : 0.710
- Corrélation max : 0.827

### effort(t)
- Patterns détectés : 34
- Corrélation moyenne : 0.699
- Corrélation max : 0.806

### mean_high_effort
- Patterns détectés : 160
- Corrélation moyenne : 0.657
- Corrélation max : 0.693

### d_effort_dt
- Patterns détectés : 7
- Corrélation moyenne : 0.704
- Corrélation max : 0.778

### mean_abs_error
- Patterns détectés : 14
- Corrélation moyenne : 0.723
- Corrélation max : 0.806

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
