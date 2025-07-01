# Rapport d'exploration FPS

**Run ID :** run_20250625-200351_seed12349
**Date :** 2025-06-25 20:03:51
**Total événements :** 226

## Résumé par type d'événement

- **anomaly** : 194 événements
- **harmonic_emergence** : 20 événements
- **fractal_pattern** : 12 événements

## Anomaly

### 1. t=257-305
- **Métrique :** d_effort_dt
- **Valeur :** 1248319903300454155433104747334274740303262490353479307093227556146004722660019799329991557120.0000
- **Sévérité :** high

### 2. t=258-305
- **Métrique :** d_effort_dt
- **Valeur :** 1022409654471412051219750427978061500159575821790279528730690289341435821265795832584559656960.0000
- **Sévérité :** high

### 3. t=259-305
- **Métrique :** d_effort_dt
- **Valeur :** 927133685065597148529821827987230527342207874773774402203785456711247839640865666765586169856.0000
- **Sévérité :** high

### 4. t=260-305
- **Métrique :** d_effort_dt
- **Valeur :** 850651950688260332962326925682357166163012581822856951556528200264005648031828438090403282944.0000
- **Sévérité :** high

### 5. t=267-305
- **Métrique :** d_effort_dt
- **Valeur :** 572807737148781759487854206343059957387744252085054521444698656811665863712990919782477856768.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=140-233
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=190-283
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=210-303
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 4. t=10-103
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=130-223
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=300-400
- **Métrique :** mean_abs_error
- **Valeur :** 0.9010
- **Sévérité :** high
- **scale :** 10/100

### 2. t=250-350
- **Métrique :** entropy_S
- **Valeur :** 0.7933
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7918
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=200-300
- **Métrique :** entropy_S
- **Valeur :** 0.7693
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=150-250
- **Métrique :** A_mean(t)
- **Valeur :** 0.7591
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 12

### A_mean(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.732
- Corrélation max : 0.792

### entropy_S
- Patterns détectés : 2
- Corrélation moyenne : 0.781
- Corrélation max : 0.793

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.731
- Corrélation max : 0.731

### mean_high_effort
- Patterns détectés : 4
- Corrélation moyenne : 0.689
- Corrélation max : 0.730

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.901
- Corrélation max : 0.901

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
