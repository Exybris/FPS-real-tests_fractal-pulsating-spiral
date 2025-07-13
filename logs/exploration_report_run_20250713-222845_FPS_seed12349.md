# Rapport d'exploration FPS

**Run ID :** run_20250713-222845_FPS_seed12349
**Date :** 2025-07-13 22:28:49
**Total événements :** 396

## Résumé par type d'événement

- **anomaly** : 306 événements
- **harmonic_emergence** : 16 événements
- **phase_cycle** : 2 événements
- **fractal_pattern** : 72 événements

## Anomaly

### 1. t=200-249
- **Métrique :** A_mean(t)
- **Valeur :** 23.3923
- **Sévérité :** high

### 2. t=201-250
- **Métrique :** A_mean(t)
- **Valeur :** 22.2828
- **Sévérité :** high

### 3. t=300-349
- **Métrique :** A_mean(t)
- **Valeur :** 22.2253
- **Sévérité :** high

### 4. t=301-350
- **Métrique :** A_mean(t)
- **Valeur :** 21.2457
- **Sévérité :** high

### 5. t=500-549
- **Métrique :** A_mean(t)
- **Valeur :** 20.5387
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

### 1. t=1113-1118
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 2. t=1202-1207
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=1200-1300
- **Métrique :** S(t)
- **Valeur :** 0.8268
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=500-600
- **Métrique :** mean_abs_error
- **Valeur :** 0.7816
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=450-550
- **Métrique :** entropy_S
- **Valeur :** 0.7619
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=1700-1800
- **Métrique :** entropy_S
- **Valeur :** 0.7326
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.7233
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 72

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.746
- Corrélation max : 0.827

### A_mean(t)
- Patterns détectés : 18
- Corrélation moyenne : 0.671
- Corrélation max : 0.723

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.747
- Corrélation max : 0.762

### effort(t)
- Patterns détectés : 13
- Corrélation moyenne : 0.680
- Corrélation max : 0.716

### mean_high_effort
- Patterns détectés : 35
- Corrélation moyenne : 0.663
- Corrélation max : 0.687

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.782
- Corrélation max : 0.782

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
