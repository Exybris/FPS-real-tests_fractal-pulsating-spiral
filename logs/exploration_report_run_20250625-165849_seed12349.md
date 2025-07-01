# Rapport d'exploration FPS

**Run ID :** run_20250625-165849_seed12349
**Date :** 2025-06-25 16:58:49
**Total événements :** 270

## Résumé par type d'événement

- **anomaly** : 246 événements
- **harmonic_emergence** : 15 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=275-305
- **Métrique :** d_effort_dt
- **Valeur :** 118091317353219582311776033135325731537147980908350605279946113062681834558541281457499799552.0000
- **Sévérité :** high

### 2. t=276-305
- **Métrique :** d_effort_dt
- **Valeur :** 109740516849206962704593948693559964086866944742975682080403994306676504178300917582263222272.0000
- **Sévérité :** high

### 3. t=277-305
- **Métrique :** d_effort_dt
- **Valeur :** 100038650241223567359357932513957149446497196080722351597723429167081890736866021607305379840.0000
- **Sévérité :** high

### 4. t=278-305
- **Métrique :** d_effort_dt
- **Valeur :** 88278411725521327376973712446394428143074623881900823296261959446693524663907396834513911808.0000
- **Sévérité :** high

### 5. t=279-305
- **Métrique :** d_effort_dt
- **Valeur :** 75500637839643403711080562221706061105306098362090185634297082750451757975311288051290865664.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=50-143
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=40-133
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=30-123
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=60-153
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** entropy_S
- **Valeur :** 0.8161
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.7913
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7548
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7273
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** mean_abs_error
- **Valeur :** 0.7205
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### A_mean(t)
- Patterns détectés : 2
- Corrélation moyenne : 0.773
- Corrélation max : 0.791

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.744
- Corrélation max : 0.816

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.688
- Corrélation max : 0.727

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.721
- Corrélation max : 0.721

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
