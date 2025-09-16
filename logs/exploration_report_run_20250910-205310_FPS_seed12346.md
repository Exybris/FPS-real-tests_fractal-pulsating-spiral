# Rapport d'exploration FPS

**Run ID :** run_20250910-205310_FPS_seed12346
**Date :** 2025-09-10 21:03:07
**Total événements :** 2677

## Résumé par type d'événement

- **anomaly** : 2181 événements
- **harmonic_emergence** : 52 événements
- **phase_cycle** : 1 événements
- **fractal_pattern** : 443 événements

## Anomaly

### 1. t=3420-3425
- **Métrique :** d_effort_dt
- **Valeur :** 446.2355
- **Sévérité :** high

### 2. t=2267-2271
- **Métrique :** d_effort_dt
- **Valeur :** 111.0943
- **Sévérité :** high

### 3. t=60-109
- **Métrique :** C(t)
- **Valeur :** 91.4341
- **Sévérité :** high

### 4. t=61-110
- **Métrique :** C(t)
- **Valeur :** 83.2655
- **Sévérité :** high

### 5. t=62-111
- **Métrique :** C(t)
- **Valeur :** 73.2652
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

### 3. t=750-843
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=2640-2733
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1101-1106
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=9300-9400
- **Métrique :** C(t)
- **Valeur :** 0.9640
- **Sévérité :** high
- **scale :** 10/100

### 2. t=9600-9700
- **Métrique :** C(t)
- **Valeur :** 0.9637
- **Sévérité :** high
- **scale :** 10/100

### 3. t=2050-2150
- **Métrique :** f_mean(t)
- **Valeur :** 0.9457
- **Sévérité :** high
- **scale :** 10/100

### 4. t=6800-6900
- **Métrique :** mean_high_effort
- **Valeur :** 0.9435
- **Sévérité :** high
- **scale :** 10/100

### 5. t=1850-1950
- **Métrique :** f_mean(t)
- **Valeur :** 0.9402
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 443

### S(t)
- Patterns détectés : 12
- Corrélation moyenne : 0.739
- Corrélation max : 0.838

### C(t)
- Patterns détectés : 43
- Corrélation moyenne : 0.766
- Corrélation max : 0.964

### A_mean(t)
- Patterns détectés : 90
- Corrélation moyenne : 0.669
- Corrélation max : 0.826

### f_mean(t)
- Patterns détectés : 98
- Corrélation moyenne : 0.924
- Corrélation max : 0.946

### entropy_S
- Patterns détectés : 15
- Corrélation moyenne : 0.726
- Corrélation max : 0.858

### effort(t)
- Patterns détectés : 9
- Corrélation moyenne : 0.705
- Corrélation max : 0.760

### mean_high_effort
- Patterns détectés : 163
- Corrélation moyenne : 0.731
- Corrélation max : 0.943

### d_effort_dt
- Patterns détectés : 6
- Corrélation moyenne : 0.739
- Corrélation max : 0.917

### mean_abs_error
- Patterns détectés : 7
- Corrélation moyenne : 0.732
- Corrélation max : 0.851

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
