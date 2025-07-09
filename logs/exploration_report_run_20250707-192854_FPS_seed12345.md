# Rapport d'exploration FPS

**Run ID :** run_20250707-192854_FPS_seed12345
**Date :** 2025-07-07 19:29:53
**Total événements :** 1929

## Résumé par type d'événement

- **anomaly** : 1466 événements
- **harmonic_emergence** : 200 événements
- **phase_cycle** : 17 événements
- **fractal_pattern** : 246 événements

## Anomaly

### 1. t=8615-8664
- **Métrique :** mean_high_effort
- **Valeur :** 72.8680
- **Sévérité :** high

### 2. t=8617-8666
- **Métrique :** mean_high_effort
- **Valeur :** 63.6633
- **Sévérité :** high

### 3. t=8714-8763
- **Métrique :** mean_high_effort
- **Valeur :** 62.5751
- **Sévérité :** high

### 4. t=8408-8457
- **Métrique :** mean_high_effort
- **Valeur :** 59.6491
- **Sévérité :** high

### 5. t=8012-8061
- **Métrique :** mean_high_effort
- **Valeur :** 59.1987
- **Sévérité :** high

## Harmonic Emergence

### 1. t=7650-7743
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=20-113
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=7210-7303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=7440-7533
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=7820-7913
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

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

### 1. t=7800-7900
- **Métrique :** mean_high_effort
- **Valeur :** 0.9563
- **Sévérité :** high
- **scale :** 10/100

### 2. t=7100-7200
- **Métrique :** mean_high_effort
- **Valeur :** 0.9378
- **Sévérité :** high
- **scale :** 10/100

### 3. t=7200-7300
- **Métrique :** mean_high_effort
- **Valeur :** 0.9340
- **Sévérité :** high
- **scale :** 10/100

### 4. t=6300-6400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9337
- **Sévérité :** high
- **scale :** 10/100

### 5. t=6800-6900
- **Métrique :** mean_high_effort
- **Valeur :** 0.9328
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 246

### S(t)
- Patterns détectés : 8
- Corrélation moyenne : 0.680
- Corrélation max : 0.706

### A_mean(t)
- Patterns détectés : 62
- Corrélation moyenne : 0.668
- Corrélation max : 0.717

### entropy_S
- Patterns détectés : 17
- Corrélation moyenne : 0.744
- Corrélation max : 0.913

### effort(t)
- Patterns détectés : 10
- Corrélation moyenne : 0.697
- Corrélation max : 0.800

### mean_high_effort
- Patterns détectés : 137
- Corrélation moyenne : 0.712
- Corrélation max : 0.956

### d_effort_dt
- Patterns détectés : 7
- Corrélation moyenne : 0.703
- Corrélation max : 0.740

### mean_abs_error
- Patterns détectés : 5
- Corrélation moyenne : 0.670
- Corrélation max : 0.712

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
