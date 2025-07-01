# Rapport d'exploration FPS

**Run ID :** run_20250623-040056_seed12345
**Date :** 2025-06-23 04:00:57
**Total événements :** 197

## Résumé par type d'événement

- **anomaly** : 164 événements
- **harmonic_emergence** : 28 événements
- **fractal_pattern** : 5 événements

## Anomaly

### 1. t=76-125
- **Métrique :** mean_abs_error
- **Valeur :** 2882164.1429
- **Sévérité :** high

### 2. t=77-126
- **Métrique :** mean_abs_error
- **Valeur :** 2059117.4202
- **Sévérité :** high

### 3. t=78-127
- **Métrique :** mean_abs_error
- **Valeur :** 1699053.8804
- **Sévérité :** high

### 4. t=79-128
- **Métrique :** mean_abs_error
- **Valeur :** 1487331.4960
- **Sévérité :** high

### 5. t=80-129
- **Métrique :** mean_abs_error
- **Valeur :** 1345009.6667
- **Sévérité :** high

## Harmonic Emergence

### 1. t=60-153
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=110-203
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=350-443
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=40-133
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=50-143
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.8996
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8245
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7570
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.6936
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6761
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 5

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.806
- Corrélation max : 0.900

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.757
- Corrélation max : 0.757

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.676
- Corrélation max : 0.676

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
