# Rapport d'exploration FPS

**Run ID :** run_20250707-025513_FPS_seed12345
**Date :** 2025-07-07 02:55:14
**Total événements :** 83

## Résumé par type d'événement

- **anomaly** : 60 événements
- **harmonic_emergence** : 10 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=98-147
- **Métrique :** A_mean(t)
- **Valeur :** 26.0995
- **Sévérité :** high

### 2. t=99-148
- **Métrique :** A_mean(t)
- **Valeur :** 25.1316
- **Sévérité :** high

### 3. t=100-149
- **Métrique :** A_mean(t)
- **Valeur :** 22.5744
- **Sévérité :** high

### 4. t=201-249
- **Métrique :** A_mean(t)
- **Valeur :** 21.9446
- **Sévérité :** high

### 5. t=202-248
- **Métrique :** A_mean(t)
- **Valeur :** 21.0677
- **Sévérité :** high

## Harmonic Emergence

### 1. t=270-363
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

### 4. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=70-163
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

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
- **Métrique :** mean_abs_error
- **Valeur :** 0.8030
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7656
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.7486
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.7160
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** d_effort_dt
- **Valeur :** 0.7074
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.664
- Corrélation max : 0.666

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.766
- Corrélation max : 0.766

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.667
- Corrélation max : 0.667

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.708
- Corrélation max : 0.749

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.707
- Corrélation max : 0.707

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.803
- Corrélation max : 0.803

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
