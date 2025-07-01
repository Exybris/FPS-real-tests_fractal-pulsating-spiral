# Rapport d'exploration FPS

**Run ID :** run_20250630-175422_seed12349
**Date :** 2025-06-30 17:54:22
**Total événements :** 144

## Résumé par type d'événement

- **anomaly** : 83 événements
- **harmonic_emergence** : 35 événements
- **phase_cycle** : 16 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=238-287
- **Métrique :** effort(t)
- **Valeur :** 220.2409
- **Sévérité :** high

### 2. t=239-288
- **Métrique :** effort(t)
- **Valeur :** 182.3976
- **Sévérité :** high

### 3. t=240-289
- **Métrique :** effort(t)
- **Valeur :** 160.1804
- **Sévérité :** high

### 4. t=236-285
- **Métrique :** effort(t)
- **Valeur :** 154.3116
- **Sévérité :** high

### 5. t=241-290
- **Métrique :** effort(t)
- **Valeur :** 144.2836
- **Sévérité :** high

## Harmonic Emergence

### 1. t=270-363
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=200-293
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=220-313
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 2. t=348-357
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 3. t=343-351
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 4. t=347-355
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=342-349
- **Métrique :** S(t)
- **Valeur :** 7.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9176
- **Sévérité :** high
- **scale :** 10/100

### 2. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8924
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.8791
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** mean_high_effort
- **Valeur :** 0.8447
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** effort(t)
- **Valeur :** 0.8223
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.685
- Corrélation max : 0.705

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.649
- Corrélation max : 0.649

### effort(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.758
- Corrélation max : 0.822

### mean_high_effort
- Patterns détectés : 5
- Corrélation moyenne : 0.842
- Corrélation max : 0.918

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
