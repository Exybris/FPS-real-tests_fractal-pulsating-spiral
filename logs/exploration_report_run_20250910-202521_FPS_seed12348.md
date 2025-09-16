# Rapport d'exploration FPS

**Run ID :** run_20250910-202521_FPS_seed12348
**Date :** 2025-09-10 20:27:17
**Total événements :** 1371

## Résumé par type d'événement

- **anomaly** : 1144 événements
- **harmonic_emergence** : 31 événements
- **fractal_pattern** : 196 événements

## Anomaly

### 1. t=4156-4159
- **Métrique :** d_effort_dt
- **Valeur :** 625.9768
- **Sévérité :** high

### 2. t=4828-4831
- **Métrique :** d_effort_dt
- **Valeur :** 542.1797
- **Sévérité :** high

### 3. t=4492-4495
- **Métrique :** d_effort_dt
- **Valeur :** 534.1297
- **Sévérité :** high

### 4. t=4548-4550
- **Métrique :** d_effort_dt
- **Valeur :** 518.4026
- **Sévérité :** high

### 5. t=4324-4327
- **Métrique :** d_effort_dt
- **Valeur :** 507.5060
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=2640-2733
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=10-103
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

### 1. t=1850-1950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9325
- **Sévérité :** high
- **scale :** 10/100

### 2. t=2150-2250
- **Métrique :** f_mean(t)
- **Valeur :** 0.9307
- **Sévérité :** high
- **scale :** 10/100

### 3. t=450-550
- **Métrique :** f_mean(t)
- **Valeur :** 0.9304
- **Sévérité :** high
- **scale :** 10/100

### 4. t=2750-2850
- **Métrique :** f_mean(t)
- **Valeur :** 0.9304
- **Sévérité :** high
- **scale :** 10/100

### 5. t=2050-2150
- **Métrique :** f_mean(t)
- **Valeur :** 0.9301
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 196

### S(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.745
- Corrélation max : 0.806

### C(t)
- Patterns détectés : 15
- Corrélation moyenne : 0.749
- Corrélation max : 0.914

### A_mean(t)
- Patterns détectés : 47
- Corrélation moyenne : 0.669
- Corrélation max : 0.814

### f_mean(t)
- Patterns détectés : 48
- Corrélation moyenne : 0.925
- Corrélation max : 0.933

### entropy_S
- Patterns détectés : 7
- Corrélation moyenne : 0.735
- Corrélation max : 0.859

### effort(t)
- Patterns détectés : 5
- Corrélation moyenne : 0.729
- Corrélation max : 0.814

### mean_high_effort
- Patterns détectés : 64
- Corrélation moyenne : 0.681
- Corrélation max : 0.909

### d_effort_dt
- Patterns détectés : 3
- Corrélation moyenne : 0.753
- Corrélation max : 0.881

### mean_abs_error
- Patterns détectés : 4
- Corrélation moyenne : 0.766
- Corrélation max : 0.836

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
  "min_duration": 3,
  "spacing_effect": {
    "enabled": true,
    "start_interval": 2.0,
    "growth": 1.5,
    "num_blocks": 0,
    "order": [
      "gamma",
      "G",
      "gamma",
      "G"
    ]
  }
}
```
