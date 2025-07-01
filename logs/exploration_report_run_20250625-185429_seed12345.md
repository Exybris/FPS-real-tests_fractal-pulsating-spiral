# Rapport d'exploration FPS

**Run ID :** run_20250625-185429_seed12345
**Date :** 2025-06-25 18:54:29
**Total événements :** 78

## Résumé par type d'événement

- **anomaly** : 73 événements
- **harmonic_emergence** : 5 événements

## Anomaly

### 1. t=105-123
- **Métrique :** d_effort_dt
- **Valeur :** 107377662926868711412535759002094433737063438605388759185896867237797763755865640089697634462679375907692865871421590723335368474624.0000
- **Sévérité :** high

### 2. t=106-123
- **Métrique :** d_effort_dt
- **Valeur :** 94425724520653068614026257615401650077682401932963751048994288467148564720517910503037404331088319330155177568264242099387174486016.0000
- **Sévérité :** high

### 3. t=107-123
- **Métrique :** d_effort_dt
- **Valeur :** 71535602434153563959401330910187003713264528455490670607272625961553437041238758723718731902750106649091082537331046050855338901504.0000
- **Sévérité :** high

### 4. t=108-123
- **Métrique :** d_effort_dt
- **Valeur :** 48427829021657005524381015078489112692954824898486682354043723804911760273478888444276340776767089442486195991759370018524986605568.0000
- **Sévérité :** high

### 5. t=109-123
- **Métrique :** d_effort_dt
- **Valeur :** 31360522598229341494716615881925556843118597664628417787957069173479526325207894238696726055977489934445619330093444465378415607808.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=20-113
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 2. t=10-103
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 3. t=30-123
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 4. t=40-133
- **Métrique :** S(t)
- **Valeur :** 2.0000
- **Sévérité :** medium

### 5. t=50-143
- **Métrique :** S(t)
- **Valeur :** 1.0000
- **Sévérité :** low

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
