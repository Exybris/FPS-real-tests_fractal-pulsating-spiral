# Rapport d'exploration FPS

**Run ID :** run_20250706-232717_FPS_seed12345
**Date :** 2025-07-06 23:27:18
**Total événements :** 64

## Résumé par type d'événement

- **anomaly** : 43 événements
- **harmonic_emergence** : 8 événements
- **phase_cycle** : 3 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=98-147
- **Métrique :** A_mean(t)
- **Valeur :** 242.4630
- **Sévérité :** high

### 2. t=99-148
- **Métrique :** A_mean(t)
- **Valeur :** 233.5583
- **Sévérité :** high

### 3. t=100-149
- **Métrique :** A_mean(t)
- **Valeur :** 210.8123
- **Sévérité :** high

### 4. t=101-150
- **Métrique :** A_mean(t)
- **Valeur :** 176.3304
- **Sévérité :** high

### 5. t=102-151
- **Métrique :** A_mean(t)
- **Valeur :** 148.7826
- **Sévérité :** high

## Harmonic Emergence

### 1. t=50-143
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=40-133
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=70-163
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=20-113
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-7
- **Métrique :** S(t)
- **Valeur :** 6.0000
- **Sévérité :** low

### 2. t=2-7
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

### 3. t=3-8
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=200-300
- **Métrique :** entropy_S
- **Valeur :** 0.8617
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=100-200
- **Métrique :** A_mean(t)
- **Valeur :** 0.8242
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=100-200
- **Métrique :** C(t)
- **Valeur :** 0.7709
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=150-250
- **Métrique :** entropy_S
- **Valeur :** 0.7496
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** mean_high_effort
- **Valeur :** 0.6882
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### C(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.771
- Corrélation max : 0.771

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.824
- Corrélation max : 0.824

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.806
- Corrélation max : 0.862

### mean_high_effort
- Patterns détectés : 6
- Corrélation moyenne : 0.676
- Corrélation max : 0.688

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
