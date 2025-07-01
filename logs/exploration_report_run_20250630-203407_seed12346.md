# Rapport d'exploration FPS

**Run ID :** run_20250630-203407_seed12346
**Date :** 2025-06-30 20:34:09
**Total événements :** 482

## Résumé par type d'événement

- **anomaly** : 198 événements
- **harmonic_emergence** : 178 événements
- **phase_cycle** : 58 événements
- **fractal_pattern** : 48 événements

## Anomaly

### 1. t=1011-1060
- **Métrique :** mean_high_effort
- **Valeur :** 46.5881
- **Sévérité :** high

### 2. t=1012-1061
- **Métrique :** mean_high_effort
- **Valeur :** 44.0662
- **Sévérité :** high

### 3. t=1013-1062
- **Métrique :** mean_high_effort
- **Valeur :** 39.7034
- **Sévérité :** high

### 4. t=1014-1063
- **Métrique :** mean_high_effort
- **Valeur :** 36.0299
- **Sévérité :** high

### 5. t=1015-1064
- **Métrique :** mean_high_effort
- **Valeur :** 32.8354
- **Sévérité :** high

## Harmonic Emergence

### 1. t=330-423
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=470-563
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=520-613
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=680-773
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=920-1013
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Phase Cycle

### 1. t=1-15
- **Métrique :** S(t)
- **Valeur :** 14.0000
- **Sévérité :** medium

### 2. t=2-11
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 3. t=1150-1159
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 4. t=1151-1160
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

### 5. t=1187-1196
- **Métrique :** S(t)
- **Valeur :** 9.0000
- **Sévérité :** low

## Fractal Pattern

### 1. t=1550-1650
- **Métrique :** mean_high_effort
- **Valeur :** 0.9579
- **Sévérité :** high
- **scale :** 10/100

### 2. t=1300-1400
- **Métrique :** mean_high_effort
- **Valeur :** 0.9449
- **Sévérité :** high
- **scale :** 10/100

### 3. t=1400-1500
- **Métrique :** mean_high_effort
- **Valeur :** 0.9449
- **Sévérité :** high
- **scale :** 10/100

### 4. t=1450-1550
- **Métrique :** mean_high_effort
- **Valeur :** 0.9437
- **Sévérité :** high
- **scale :** 10/100

### 5. t=1200-1300
- **Métrique :** mean_high_effort
- **Valeur :** 0.9142
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 48

### S(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.708
- Corrélation max : 0.708

### A_mean(t)
- Patterns détectés : 7
- Corrélation moyenne : 0.687
- Corrélation max : 0.767

### entropy_S
- Patterns détectés : 4
- Corrélation moyenne : 0.750
- Corrélation max : 0.820

### effort(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.706
- Corrélation max : 0.730

### mean_high_effort
- Patterns détectés : 30
- Corrélation moyenne : 0.740
- Corrélation max : 0.958

### d_effort_dt
- Patterns détectés : 3
- Corrélation moyenne : 0.676
- Corrélation max : 0.692

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
