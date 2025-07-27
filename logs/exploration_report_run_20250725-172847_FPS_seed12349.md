# Rapport d'exploration FPS

**Run ID :** run_20250725-172847_FPS_seed12349
**Date :** 2025-07-25 17:28:50
**Total événements :** 325

## Résumé par type d'événement

- **anomaly** : 274 événements
- **harmonic_emergence** : 9 événements
- **fractal_pattern** : 42 événements

## Anomaly

### 1. t=60-109
- **Métrique :** C(t)
- **Valeur :** 91.4341
- **Sévérité :** high

### 2. t=61-110
- **Métrique :** C(t)
- **Valeur :** 83.2655
- **Sévérité :** high

### 3. t=62-111
- **Métrique :** C(t)
- **Valeur :** 73.2652
- **Sévérité :** high

### 4. t=854-903
- **Métrique :** C(t)
- **Valeur :** 71.3573
- **Sévérité :** high

### 5. t=855-904
- **Métrique :** C(t)
- **Valeur :** 69.0219
- **Sévérité :** high

## Harmonic Emergence

### 1. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=180-273
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=550-650
- **Métrique :** f_mean(t)
- **Valeur :** 0.9283
- **Sévérité :** high
- **scale :** 10/100

### 2. t=450-550
- **Métrique :** f_mean(t)
- **Valeur :** 0.9265
- **Sévérité :** high
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** f_mean(t)
- **Valeur :** 0.9247
- **Sévérité :** high
- **scale :** 10/100

### 4. t=750-850
- **Métrique :** f_mean(t)
- **Valeur :** 0.9246
- **Sévérité :** high
- **scale :** 10/100

### 5. t=650-750
- **Métrique :** f_mean(t)
- **Valeur :** 0.9242
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 42

### S(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.836
- Corrélation max : 0.836

### C(t)
- Patterns détectés : 6
- Corrélation moyenne : 0.759
- Corrélation max : 0.847

### A_mean(t)
- Patterns détectés : 7
- Corrélation moyenne : 0.675
- Corrélation max : 0.716

### f_mean(t)
- Patterns détectés : 8
- Corrélation moyenne : 0.923
- Corrélation max : 0.928

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.789
- Corrélation max : 0.912

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.715
- Corrélation max : 0.715

### mean_high_effort
- Patterns détectés : 15
- Corrélation moyenne : 0.668
- Corrélation max : 0.689

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.732
- Corrélation max : 0.732

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
