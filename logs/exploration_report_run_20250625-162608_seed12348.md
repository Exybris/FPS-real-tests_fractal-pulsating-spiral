# Rapport d'exploration FPS

**Run ID :** run_20250625-162608_seed12348
**Date :** 2025-06-25 16:26:08
**Total événements :** 262

## Résumé par type d'événement

- **anomaly** : 235 événements
- **harmonic_emergence** : 15 événements
- **fractal_pattern** : 12 événements

## Anomaly

### 1. t=271-305
- **Métrique :** d_effort_dt
- **Valeur :** 388466915868060125843137807464358934359602142642915153074137546141775536859111209959801487360.0000
- **Sévérité :** high

### 2. t=272-305
- **Métrique :** d_effort_dt
- **Valeur :** 360035946588544326981217001397767683151109969734363059681297903331059983455853723988820754432.0000
- **Sévérité :** high

### 3. t=273-305
- **Métrique :** d_effort_dt
- **Valeur :** 329142394809829388493612945669097036129168893089622885306098650785078016543654782457675251712.0000
- **Sévérité :** high

### 4. t=274-305
- **Métrique :** d_effort_dt
- **Valeur :** 295630933445426627298861139745515628319047419149330525768993254911766603338506972986534264832.0000
- **Sévérité :** high

### 5. t=275-305
- **Métrique :** d_effort_dt
- **Valeur :** 259779243063105111744284962246364988031527806816111382725353854591504077615584823538299699200.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=30-123
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=150-243
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=210-303
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 5. t=40-133
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=250-350
- **Métrique :** mean_abs_error
- **Valeur :** 0.8937
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8050
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.8034
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7463
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** entropy_S
- **Valeur :** 0.7284
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 12

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.767
- Corrélation max : 0.805

### f_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.720
- Corrélation max : 0.720

### entropy_S
- Patterns détectés : 4
- Corrélation moyenne : 0.686
- Corrélation max : 0.728

### mean_high_effort
- Patterns détectés : 3
- Corrélation moyenne : 0.689
- Corrélation max : 0.746

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.894
- Corrélation max : 0.894

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
