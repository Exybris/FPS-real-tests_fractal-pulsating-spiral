# Rapport d'exploration FPS

**Run ID :** run_20250625-202513_seed12345
**Date :** 2025-06-25 20:25:14
**Total événements :** 230

## Résumé par type d'événement

- **anomaly** : 200 événements
- **harmonic_emergence** : 20 événements
- **fractal_pattern** : 10 événements

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

### 1. t=220-313
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=10-103
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 3. t=210-303
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 4. t=240-333
- **Métrique :** S(t)
- **Valeur :** 3.0000
- **Sévérité :** medium

### 5. t=30-123
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=150-250
- **Métrique :** mean_abs_error
- **Valeur :** 0.8199
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.7918
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=150-250
- **Métrique :** A_mean(t)
- **Valeur :** 0.7591
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=100-200
- **Métrique :** effort(t)
- **Valeur :** 0.7315
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=250-350
- **Métrique :** mean_high_effort
- **Valeur :** 0.7302
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 10

### A_mean(t)
- Patterns détectés : 4
- Corrélation moyenne : 0.732
- Corrélation max : 0.792

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
- Corrélation moyenne : 0.820
- Corrélation max : 0.820

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
