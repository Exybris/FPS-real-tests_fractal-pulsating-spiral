# Rapport d'exploration FPS

**Run ID :** run_20250625-184705_seed12345
**Date :** 2025-06-25 18:47:05
**Total événements :** 29

## Résumé par type d'événement

- **anomaly** : 29 événements

## Anomaly

### 1. t=16-32
- **Métrique :** d_effort_dt
- **Valeur :** 3018174469698203064178001331263427251162578189492138271636325069834175401040570421177894305792.0000
- **Sévérité :** high

### 2. t=17-32
- **Métrique :** d_effort_dt
- **Valeur :** 1652688557761947300052651965065372754291001580724473762206567700293668978200634576977367400448.0000
- **Sévérité :** high

### 3. t=25-32
- **Métrique :** d_effort_dt
- **Valeur :** 841024298829474707393113289901519902188645239713083994888140487163619442157210323638831546368.0000
- **Sévérité :** high

### 4. t=26-32
- **Métrique :** d_effort_dt
- **Valeur :** 609985486118770310322226104784182219142512854613103991389512446728524146207109787738783612928.0000
- **Sévérité :** high

### 5. t=25-49
- **Métrique :** mean_high_effort
- **Valeur :** 502518340621870521559105474731227092158650137526516314389847525469931875020406545619248742400.0000
- **Sévérité :** high

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
