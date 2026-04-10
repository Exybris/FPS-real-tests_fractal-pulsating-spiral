# Rapport d'exploration FPS

**Run ID :** run_20260407-173917_FPS_seed12345
**Date :** 2026-04-07 17:49:56
**Total événements :** 4735

## Résumé par type d'événement

- **anomaly** : 4202 événements
- **harmonic_emergence** : 5 événements
- **fractal_pattern** : 528 événements

## Anomaly

### 1. t=61-110
- **Métrique :** C(t)
- **Valeur :** 100.6625
- **Sévérité :** high

### 2. t=62-111
- **Métrique :** C(t)
- **Valeur :** 91.8973
- **Sévérité :** high

### 3. t=63-112
- **Métrique :** C(t)
- **Valeur :** 81.1054
- **Sévérité :** high

### 4. t=64-113
- **Métrique :** C(t)
- **Valeur :** 69.9123
- **Sévérité :** high

### 5. t=65-114
- **Métrique :** C(t)
- **Valeur :** 59.5109
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=30-123
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=90-183
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=100-193
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=120-213
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=200-300
- **Métrique :** An_mean(t)
- **Valeur :** 0.9383
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** An_mean(t)
- **Valeur :** 0.9256
- **Sévérité :** high
- **scale :** 10/100

### 3. t=1950-2050
- **Métrique :** fn_mean(t)
- **Valeur :** 0.9225
- **Sévérité :** high
- **scale :** 10/100

### 4. t=2150-2250
- **Métrique :** fn_mean(t)
- **Valeur :** 0.9225
- **Sévérité :** high
- **scale :** 10/100

### 5. t=3350-3450
- **Métrique :** fn_mean(t)
- **Valeur :** 0.9225
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 528

### S(t)
- Patterns détectés : 98
- Corrélation moyenne : 0.858
- Corrélation max : 0.877

### C(t)
- Patterns détectés : 70
- Corrélation moyenne : 0.808
- Corrélation max : 0.916

### An_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.908
- Corrélation max : 0.938

### fn_mean(t)
- Patterns détectés : 98
- Corrélation moyenne : 0.919
- Corrélation max : 0.922

### entropy_S
- Patterns détectés : 61
- Corrélation moyenne : 0.652
- Corrélation max : 0.693

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.662
- Corrélation max : 0.675

### mean_high_effort
- Patterns détectés : 196
- Corrélation moyenne : 0.802
- Corrélation max : 0.832

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "C(t)",
    "An_mean(t)",
    "fn_mean(t)",
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
  "detect_fractal_patterns": true,
  "detect_anomalies": true,
  "detect_harmonics": true,
  "anomaly_threshold": 3.0,
  "fractal_threshold": 0.8,
  "min_duration": 3,
  "recurrence_window": [
    1,
    10,
    100
  ]
}
```
