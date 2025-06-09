# Rapport d'exploration FPS

**Run ID :** run_20250609-181622_seed12346
**Date :** 2025-06-09 18:16:23
**Total événements :** 202

## Résumé par type d'événement

- **anomaly** : 170 événements
- **harmonic_emergence** : 23 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=199-248
- **Métrique :** mean_abs_error
- **Valeur :** 157321.2857
- **Sévérité :** high

### 2. t=200-249
- **Métrique :** mean_abs_error
- **Valeur :** 135038.1242
- **Sévérité :** high

### 3. t=201-250
- **Métrique :** mean_abs_error
- **Valeur :** 128112.8426
- **Sévérité :** high

### 4. t=202-251
- **Métrique :** mean_abs_error
- **Valeur :** 81544.0402
- **Sévérité :** high

### 5. t=203-252
- **Métrique :** mean_abs_error
- **Valeur :** 57112.9135
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-120
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=220-320
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=330-430
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=340-440
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 5. t=120-220
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** A_mean(t)
- **Valeur :** 0.9199
- **Sévérité :** high
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.9104
- **Sévérité :** high
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8935
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=350-450
- **Métrique :** C(t)
- **Valeur :** 0.8758
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** C(t)
- **Valeur :** 0.8725
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### C(t)
- Patterns détectés : 6
- Corrélation moyenne : 0.865
- Corrélation max : 0.876

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.908
- Corrélation max : 0.920

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
