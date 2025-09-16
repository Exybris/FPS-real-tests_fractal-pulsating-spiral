# Rapport d'exploration FPS

**Run ID :** run_20250910-202717_FPS_seed12349
**Date :** 2025-09-10 20:29:13
**Total événements :** 1319

## Résumé par type d'événement

- **anomaly** : 1071 événements
- **harmonic_emergence** : 30 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 217 événements

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

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=550-643
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=3440-3533
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1101-1106
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=2050-2150
- **Métrique :** f_mean(t)
- **Valeur :** 0.9436
- **Sévérité :** high
- **scale :** 10/100

### 2. t=1850-1950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9403
- **Sévérité :** high
- **scale :** 10/100

### 3. t=2750-2850
- **Métrique :** f_mean(t)
- **Valeur :** 0.9401
- **Sévérité :** high
- **scale :** 10/100

### 4. t=1650-1750
- **Métrique :** f_mean(t)
- **Valeur :** 0.9363
- **Sévérité :** high
- **scale :** 10/100

### 5. t=2150-2250
- **Métrique :** f_mean(t)
- **Valeur :** 0.9361
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 217

### S(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.764
- Corrélation max : 0.816

### C(t)
- Patterns détectés : 14
- Corrélation moyenne : 0.756
- Corrélation max : 0.914

### A_mean(t)
- Patterns détectés : 42
- Corrélation moyenne : 0.680
- Corrélation max : 0.814

### f_mean(t)
- Patterns détectés : 48
- Corrélation moyenne : 0.923
- Corrélation max : 0.944

### entropy_S
- Patterns détectés : 5
- Corrélation moyenne : 0.734
- Corrélation max : 0.854

### effort(t)
- Patterns détectés : 13
- Corrélation moyenne : 0.730
- Corrélation max : 0.845

### mean_high_effort
- Patterns détectés : 84
- Corrélation moyenne : 0.663
- Corrélation max : 0.730

### d_effort_dt
- Patterns détectés : 4
- Corrélation moyenne : 0.719
- Corrélation max : 0.774

### mean_abs_error
- Patterns détectés : 5
- Corrélation moyenne : 0.748
- Corrélation max : 0.853

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
