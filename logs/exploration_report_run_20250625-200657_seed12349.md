# Rapport d'exploration FPS

**Run ID :** run_20250625-200657_seed12349
**Date :** 2025-06-25 20:06:57
**Total événements :** 249

## Résumé par type d'événement

- **anomaly** : 218 événements
- **harmonic_emergence** : 20 événements
- **fractal_pattern** : 11 événements

## Anomaly

### 1. t=258-305
- **Métrique :** d_effort_dt
- **Valeur :** 1147929568724230883415397113976563441395303793700053667077638506332307231458600726798668922880.0000
- **Sévérité :** high

### 2. t=259-305
- **Métrique :** d_effort_dt
- **Valeur :** 1018226691553117557414079601144894101122592096405708659457205711778076993991257310699846631424.0000
- **Sévérité :** high

### 3. t=260-305
- **Métrique :** d_effort_dt
- **Valeur :** 918216418566452978907252660385333471060435109315359919695537095176108300854598709196294717440.0000
- **Sévérité :** high

### 4. t=261-305
- **Métrique :** d_effort_dt
- **Valeur :** 840780250028504662997955419362250768796053590118042938702806321484452720240516921389337280512.0000
- **Sévérité :** high

### 5. t=267-305
- **Métrique :** d_effort_dt
- **Valeur :** 601719854265355152482474711906843964814568693061307725110343723130393483218305420818269601792.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=200-293
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=210-303
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=230-323
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=190-283
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=30-123
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_abs_error
- **Valeur :** 0.9222
- **Sévérité :** high
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** A_mean(t)
- **Valeur :** 0.9006
- **Sévérité :** high
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8956
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** entropy_S
- **Valeur :** 0.7743
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7686
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 11

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.855
- Corrélation max : 0.901

### entropy_S
- Patterns détectés : 1
- Corrélation moyenne : 0.774
- Corrélation max : 0.774

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.676
- Corrélation max : 0.676

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.686
- Corrélation max : 0.726

### d_effort_dt
- Patterns détectés : 1
- Corrélation moyenne : 0.646
- Corrélation max : 0.646

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.922
- Corrélation max : 0.922

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
