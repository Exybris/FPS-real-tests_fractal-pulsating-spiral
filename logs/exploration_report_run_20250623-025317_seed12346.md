# Rapport d'exploration FPS

**Run ID :** run_20250623-025317_seed12346
**Date :** 2025-06-23 02:53:17
**Total événements :** 165

## Résumé par type d'événement

- **anomaly** : 128 événements
- **harmonic_emergence** : 31 événements
- **fractal_pattern** : 6 événements

## Anomaly

### 1. t=85-134
- **Métrique :** mean_abs_error
- **Valeur :** 1709049.8571
- **Sévérité :** high

### 2. t=86-135
- **Métrique :** mean_abs_error
- **Valeur :** 1221004.0923
- **Sévérité :** high

### 3. t=87-136
- **Métrique :** mean_abs_error
- **Valeur :** 1007495.5658
- **Sévérité :** high

### 4. t=88-137
- **Métrique :** mean_abs_error
- **Valeur :** 881949.5682
- **Sévérité :** high

### 5. t=89-138
- **Métrique :** mean_abs_error
- **Valeur :** 627049.6056
- **Sévérité :** high

## Harmonic Emergence

### 1. t=120-213
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 2. t=140-233
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=160-253
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=260-353
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.9043
- **Sévérité :** high
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8231
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.7892
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7687
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=350-450
- **Métrique :** entropy_S
- **Valeur :** 0.7643
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 6

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.809
- Corrélation max : 0.904

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.767
- Corrélation max : 0.769

### mean_high_effort
- Patterns détectés : 1
- Corrélation moyenne : 0.789
- Corrélation max : 0.789

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
