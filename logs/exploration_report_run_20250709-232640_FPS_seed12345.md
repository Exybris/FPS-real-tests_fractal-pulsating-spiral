# Rapport d'exploration FPS

**Run ID :** run_20250709-232640_FPS_seed12345
**Date :** 2025-07-09 23:26:41
**Total événements :** 85

## Résumé par type d'événement

- **anomaly** : 71 événements
- **harmonic_emergence** : 5 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=101-150
- **Métrique :** A_mean(t)
- **Valeur :** 19.8487
- **Sévérité :** high

### 2. t=201-250
- **Métrique :** A_mean(t)
- **Valeur :** 19.8487
- **Sévérité :** high

### 3. t=301-350
- **Métrique :** A_mean(t)
- **Valeur :** 19.8487
- **Sévérité :** high

### 4. t=401-450
- **Métrique :** A_mean(t)
- **Valeur :** 19.8487
- **Sévérité :** high

### 5. t=100-149
- **Métrique :** A_mean(t)
- **Valeur :** 19.8487
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=70-163
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=270-363
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7226
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6895
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.6822
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** mean_high_effort
- **Valeur :** 0.6780
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.6725
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.664
- Corrélation max : 0.664

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.723
- Corrélation max : 0.723

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.677
- Corrélation max : 0.689

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
