# Rapport d'exploration FPS

**Run ID :** run_20250707-010658_FPS_seed12345
**Date :** 2025-07-07 01:06:59
**Total événements :** 44

## Résumé par type d'événement

- **anomaly** : 29 événements
- **harmonic_emergence** : 5 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=98-147
- **Métrique :** A_mean(t)
- **Valeur :** 242.4630
- **Sévérité :** high

### 2. t=99-148
- **Métrique :** A_mean(t)
- **Valeur :** 233.5583
- **Sévérité :** high

### 3. t=100-149
- **Métrique :** A_mean(t)
- **Valeur :** 210.8123
- **Sévérité :** high

### 4. t=101-104
- **Métrique :** S(t)
- **Valeur :** 202.1422
- **Sévérité :** high

### 5. t=101-109
- **Métrique :** mean_abs_error
- **Valeur :** 61.3600
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=40-133
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=50-143
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=30-123
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=100-200
- **Métrique :** effort(t)
- **Valeur :** 0.9623
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.8344
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** d_effort_dt
- **Valeur :** 0.8135
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.8068
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** entropy_S
- **Valeur :** 0.7712
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.803
- Corrélation max : 0.834

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.962
- Corrélation max : 0.962

### mean_high_effort
- Patterns détectés : 6
- Corrélation moyenne : 0.696
- Corrélation max : 0.807

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.814
- Corrélation max : 0.814

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
