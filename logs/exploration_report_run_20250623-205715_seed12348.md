# Rapport d'exploration FPS

**Run ID :** run_20250623-205715_seed12348
**Date :** 2025-06-23 20:57:16
**Total événements :** 252

## Résumé par type d'événement

- **anomaly** : 229 événements
- **harmonic_emergence** : 13 événements
- **fractal_pattern** : 10 événements

## Anomaly

### 1. t=271-305
- **Métrique :** d_effort_dt
- **Valeur :** 388470028829395351426577436430941899345380188527277692808454518343009973433809714326099984384.0000
- **Sévérité :** high

### 2. t=272-305
- **Métrique :** d_effort_dt
- **Valeur :** 360039556457685152135857733281492753127015744088510148043881362354706135588547064328842379264.0000
- **Sévérité :** high

### 3. t=273-305
- **Métrique :** d_effort_dt
- **Valeur :** 329143810160962535330126078059946808394541802114246683330922114879532942341099650049512570880.0000
- **Sévérité :** high

### 4. t=274-305
- **Métrique :** d_effort_dt
- **Valeur :** 295632337050136362776815691839553717046970900275357571898993768835312727210691585420212305920.0000
- **Sévérité :** high

### 5. t=275-305
- **Métrique :** d_effort_dt
- **Valeur :** 259778998342583040265625411164137965032751631988651510666689735609854796330439364259149250560.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=30-123
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 3. t=40-133
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=130-223
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=150-243
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
- **Valeur :** 0.7464
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=300-400
- **Métrique :** entropy_S
- **Valeur :** 0.7282
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 3
- Corrélation moyenne : 0.767
- Corrélation max : 0.805

### entropy_S
- Patterns détectés : 3
- Corrélation moyenne : 0.683
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
