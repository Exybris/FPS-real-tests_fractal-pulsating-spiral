# Rapport d'exploration FPS

**Run ID :** run_20250713-225652_FPS_seed12349
**Date :** 2025-07-13 22:56:57
**Total événements :** 363

## Résumé par type d'événement

- **anomaly** : 272 événements
- **harmonic_emergence** : 27 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 61 événements

## Anomaly

### 1. t=1000-1049
- **Métrique :** A_mean(t)
- **Valeur :** 21.8501
- **Sévérité :** high

### 2. t=1001-1050
- **Métrique :** A_mean(t)
- **Valeur :** 21.6950
- **Sévérité :** high

### 3. t=900-949
- **Métrique :** A_mean(t)
- **Valeur :** 21.6930
- **Sévérité :** high

### 4. t=901-950
- **Métrique :** A_mean(t)
- **Valeur :** 21.2366
- **Sévérité :** high

### 5. t=1500-1549
- **Métrique :** A_mean(t)
- **Valeur :** 21.0631
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=260-353
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=320-413
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1041-1048
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

### 2. t=413-418
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=842-847
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=1200-1300
- **Métrique :** S(t)
- **Valeur :** 0.8416
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=550-650
- **Métrique :** A_mean(t)
- **Valeur :** 0.8211
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=1850-1950
- **Métrique :** effort(t)
- **Valeur :** 0.8185
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=500-600
- **Métrique :** A_mean(t)
- **Valeur :** 0.7894
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=600-700
- **Métrique :** mean_abs_error
- **Valeur :** 0.7560
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 61

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.751
- Corrélation max : 0.842

### A_mean(t)
- Patterns détectés : 16
- Corrélation moyenne : 0.689
- Corrélation max : 0.821

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.689
- Corrélation max : 0.689

### effort(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.734
- Corrélation max : 0.818

### mean_high_effort
- Patterns détectés : 35
- Corrélation moyenne : 0.664
- Corrélation max : 0.690

### d_effort_dt
- Patterns détectés : 2
- Corrélation moyenne : 0.742
- Corrélation max : 0.745

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.756
- Corrélation max : 0.756

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
