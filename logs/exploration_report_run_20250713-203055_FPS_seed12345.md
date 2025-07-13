# Rapport d'exploration FPS

**Run ID :** run_20250713-203055_FPS_seed12345
**Date :** 2025-07-13 20:31:50
**Total événements :** 1977

## Résumé par type d'événement

- **anomaly** : 1561 événements
- **harmonic_emergence** : 52 événements
- **phase_cycle** : 30 événements
- **fractal_pattern** : 334 événements

## Anomaly

### 1. t=1300-1349
- **Métrique :** A_mean(t)
- **Valeur :** 20.1340
- **Sévérité :** high

### 2. t=1800-1849
- **Métrique :** A_mean(t)
- **Valeur :** 20.0989
- **Sévérité :** high

### 3. t=2300-2349
- **Métrique :** A_mean(t)
- **Valeur :** 20.0225
- **Sévérité :** high

### 4. t=1100-1149
- **Métrique :** A_mean(t)
- **Valeur :** 20.0208
- **Sévérité :** high

### 5. t=1801-1850
- **Métrique :** A_mean(t)
- **Valeur :** 20.0126
- **Sévérité :** high

## Harmonic Emergence

### 1. t=1140-1233
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 2. t=1510-1603
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=2210-2303
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=10-103
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

### 5. t=20-113
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

## Phase Cycle

### 1. t=1-17
- **Métrique :** S(t)
- **Valeur :** 16.0000
- **Sévérité :** medium

### 2. t=2-17
- **Métrique :** S(t)
- **Valeur :** 15.0000
- **Sévérité :** medium

### 3. t=3-17
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 4. t=6-17
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

### 5. t=5625-5636
- **Métrique :** S(t)
- **Valeur :** 11.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=4300-4400
- **Métrique :** entropy_S
- **Valeur :** 0.8875
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=7700-7800
- **Métrique :** mean_abs_error
- **Valeur :** 0.8228
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=2850-2950
- **Métrique :** entropy_S
- **Valeur :** 0.8004
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=4100-4200
- **Métrique :** effort(t)
- **Valeur :** 0.7969
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=2500-2600
- **Métrique :** entropy_S
- **Valeur :** 0.7951
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 334

### S(t)
- Patterns détectés : 11
- Corrélation moyenne : 0.709
- Corrélation max : 0.791

### A_mean(t)
- Patterns détectés : 96
- Corrélation moyenne : 0.665
- Corrélation max : 0.693

### entropy_S
- Patterns détectés : 12
- Corrélation moyenne : 0.740
- Corrélation max : 0.888

### effort(t)
- Patterns détectés : 49
- Corrélation moyenne : 0.717
- Corrélation max : 0.797

### mean_high_effort
- Patterns détectés : 154
- Corrélation moyenne : 0.657
- Corrélation max : 0.685

### d_effort_dt
- Patterns détectés : 3
- Corrélation moyenne : 0.696
- Corrélation max : 0.757

### mean_abs_error
- Patterns détectés : 9
- Corrélation moyenne : 0.721
- Corrélation max : 0.823

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
