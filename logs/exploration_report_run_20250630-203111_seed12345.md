# Rapport d'exploration FPS

**Run ID :** run_20250630-203111_seed12345
**Date :** 2025-06-30 20:31:12
**Total événements :** 235

## Résumé par type d'événement

- **anomaly** : 97 événements
- **harmonic_emergence** : 79 événements
- **phase_cycle** : 38 événements
- **fractal_pattern** : 21 événements

## Anomaly

### 1. t=519-568
- **Métrique :** mean_high_effort
- **Valeur :** 156.6143
- **Sévérité :** high

### 2. t=520-569
- **Métrique :** mean_high_effort
- **Valeur :** 148.0799
- **Sévérité :** high

### 3. t=521-570
- **Métrique :** mean_high_effort
- **Valeur :** 109.7326
- **Sévérité :** high

### 4. t=522-571
- **Métrique :** mean_high_effort
- **Valeur :** 73.9363
- **Sévérité :** high

### 5. t=523-572
- **Métrique :** mean_high_effort
- **Valeur :** 50.6612
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=470-563
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=420-513
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=540-633
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=560-653
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=671-686
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 2. t=1-14
- **Métrique :** S(t)
- **Valeur :** 13.0000
- **Sévérité :** medium

### 3. t=662-673
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

### 4. t=672-681
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 5. t=559-567
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=700-800
- **Métrique :** mean_high_effort
- **Valeur :** 0.9515
- **Sévérité :** high
- **scale :** 10/100

### 2. t=600-700
- **Métrique :** mean_high_effort
- **Valeur :** 0.9232
- **Sévérité :** high
- **scale :** 10/100

### 3. t=850-950
- **Métrique :** entropy_S
- **Valeur :** 0.8949
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=650-750
- **Métrique :** mean_high_effort
- **Valeur :** 0.8675
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=550-650
- **Métrique :** mean_high_effort
- **Valeur :** 0.8207
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 21

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.690
- Corrélation max : 0.721

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.747
- Corrélation max : 0.895

### mean_high_effort
- Patterns détectés : 14
- Corrélation moyenne : 0.748
- Corrélation max : 0.952

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.658
- Corrélation max : 0.658

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
