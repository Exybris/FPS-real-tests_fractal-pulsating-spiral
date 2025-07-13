# Rapport d'exploration FPS

**Run ID :** run_20250713-204216_FPS_seed12349
**Date :** 2025-07-13 20:43:12
**Total événements :** 1944

## Résumé par type d'événement

- **anomaly** : 1572 événements
- **harmonic_emergence** : 45 événements
- **phase_cycle** : 20 événements
- **fractal_pattern** : 307 événements

## Anomaly

### 1. t=401-446
- **Métrique :** A_mean(t)
- **Valeur :** 24.5513
- **Sévérité :** high

### 2. t=402-446
- **Métrique :** A_mean(t)
- **Valeur :** 23.9873
- **Sévérité :** high

### 3. t=600-649
- **Métrique :** A_mean(t)
- **Valeur :** 22.2143
- **Sévérité :** high

### 4. t=601-650
- **Métrique :** A_mean(t)
- **Valeur :** 20.4069
- **Sévérité :** high

### 5. t=1801-1850
- **Métrique :** A_mean(t)
- **Valeur :** 20.1315
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=4430-4523
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=30-123
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=40-133
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=8-23
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 2. t=7-21
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 3. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 4. t=2-13
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

### 5. t=12-23
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=600-700
- **Métrique :** mean_abs_error
- **Valeur :** 0.9224
- **Sévérité :** high
- **scale :** 10/100

### 2. t=9300-9400
- **Métrique :** entropy_S
- **Valeur :** 0.8649
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=2450-2550
- **Métrique :** mean_abs_error
- **Valeur :** 0.8346
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=6700-6800
- **Métrique :** S(t)
- **Valeur :** 0.8198
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=2450-2550
- **Métrique :** S(t)
- **Valeur :** 0.8093
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 307

### S(t)
- Patterns détectés : 10
- Corrélation moyenne : 0.710
- Corrélation max : 0.820

### A_mean(t)
- Patterns détectés : 95
- Corrélation moyenne : 0.666
- Corrélation max : 0.704

### entropy_S
- Patterns détectés : 18
- Corrélation moyenne : 0.714
- Corrélation max : 0.865

### mean_high_effort
- Patterns détectés : 164
- Corrélation moyenne : 0.657
- Corrélation max : 0.687

### mean_abs_error
- Patterns détectés : 20
- Corrélation moyenne : 0.723
- Corrélation max : 0.922

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
