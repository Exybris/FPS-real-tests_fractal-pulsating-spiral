# Rapport d'exploration FPS

**Run ID :** run_20260407-174904_KURAMOTO_seed12345
**Date :** 2026-04-07 17:49:59
**Total événements :** 2886

## Résumé par type d'événement

- **anomaly** : 868 événements
- **harmonic_emergence** : 46 événements
- **phase_cycle** : 1800 événements
- **fractal_pattern** : 172 événements

## Anomaly

### 1. t=3299-3348
- **Métrique :** C(t)
- **Valeur :** 46.2112
- **Sévérité :** high

### 2. t=3300-3349
- **Métrique :** C(t)
- **Valeur :** 44.8177
- **Sévérité :** high

### 3. t=3301-3350
- **Métrique :** C(t)
- **Valeur :** 41.9332
- **Sévérité :** high

### 4. t=3302-3351
- **Métrique :** C(t)
- **Valeur :** 38.5537
- **Sévérité :** high

### 5. t=3303-3352
- **Métrique :** C(t)
- **Valeur :** 35.1619
- **Sévérité :** high

## Harmonic Emergence

### 1. t=140-233
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 2. t=160-253
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 3. t=350-443
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 4. t=360-453
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=490-583
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=6689-6737
- **Métrique :** S(t)
- **Valeur :** 48.0000
- **Sévérité :** medium

### 2. t=6690-6738
- **Métrique :** S(t)
- **Valeur :** 48.0000
- **Sévérité :** medium

### 3. t=6691-6739
- **Métrique :** S(t)
- **Valeur :** 48.0000
- **Sévérité :** medium

### 4. t=2474-2521
- **Métrique :** S(t)
- **Valeur :** 47.0000
- **Sévérité :** medium

### 5. t=2475-2522
- **Métrique :** S(t)
- **Valeur :** 47.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=1650-1750
- **Métrique :** C(t)
- **Valeur :** 0.9788
- **Sévérité :** high
- **scale :** 10/100

### 2. t=7250-7350
- **Métrique :** C(t)
- **Valeur :** 0.9717
- **Sévérité :** high
- **scale :** 10/100

### 3. t=5500-5600
- **Métrique :** S(t)
- **Valeur :** 0.9713
- **Sévérité :** high
- **scale :** 10/100

### 4. t=3150-3250
- **Métrique :** C(t)
- **Valeur :** 0.9455
- **Sévérité :** high
- **scale :** 10/100

### 5. t=3450-3550
- **Métrique :** C(t)
- **Valeur :** 0.9438
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 172

### S(t)
- Patterns détectés : 80
- Corrélation moyenne : 0.781
- Corrélation max : 0.971

### C(t)
- Patterns détectés : 92
- Corrélation moyenne : 0.790
- Corrélation max : 0.979

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
