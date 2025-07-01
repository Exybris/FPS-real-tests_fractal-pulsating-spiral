# Rapport d'exploration FPS

**Run ID :** run_20250625-210400_seed12346
**Date :** 2025-06-25 21:04:00
**Total événements :** 124

## Résumé par type d'événement

- **anomaly** : 88 événements
- **harmonic_emergence** : 33 événements
- **fractal_pattern** : 3 événements

## Anomaly

### 1. t=306-310
- **Métrique :** d_effort_dt
- **Valeur :** 79526593693760797716670682868040541219166191068746165667033125513593721188741290526184719432305352653702322496897300958958073195481573746581811422750784683840683999847663810123401082219435946475440818308940392867241682965440233472.0000
- **Sévérité :** high

### 2. t=307-310
- **Métrique :** d_effort_dt
- **Valeur :** 20908555262891450794353700821968533177496014933582686399820323475187565958730047669219756494476865345606778574216207282255944236180105734736339439378085050674003845656104076804766415432587315251501218053203664221096137344770113536.0000
- **Sévérité :** high

### 3. t=280-329
- **Métrique :** mean_high_effort
- **Valeur :** 17150829362169174460650373864474085806421126101218772194285489116141280891675018729427511508750064012907150553242876787147657489448435785412314293323861380902248142385528810521647323684300660958446420617232027195664236444678881280.0000
- **Sévérité :** high

### 4. t=308-310
- **Métrique :** d_effort_dt
- **Valeur :** 14743896573714931271952116036953774618983374373263260735955551420040796026837784964316214045377659393728811216966283981735222997199634932349093370561276443839554775321686263503823797991179519544736816558063267839773429116021243904.0000
- **Sévérité :** high

### 5. t=288-337
- **Métrique :** mean_high_effort
- **Valeur :** 11544928118423228402283071811041832426599528006155061215522591543736928043850643349829984903695954621399063459949681041836070562110821870192639267509030202929253330908278095142040521691882272463604102105165260080267726563749920768.0000
- **Sévérité :** high

## Harmonic Emergence

### 1. t=290-383
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 2. t=370-463
- **Métrique :** S(t)
- **Valeur :** 5.0000
- **Sévérité :** medium

### 3. t=40-133
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 4. t=120-213
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

### 5. t=130-223
- **Métrique :** S(t)
- **Valeur :** 4.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=200-300
- **Métrique :** A_mean(t)
- **Valeur :** 0.8807
- **Sévérité :** medium
- **scale :** 10/100

### 2. t=150-250
- **Métrique :** effort(t)
- **Valeur :** 0.8161
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=200-300
- **Métrique :** mean_abs_error
- **Valeur :** 0.6684
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 3

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.881
- Corrélation max : 0.881

### effort(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.816
- Corrélation max : 0.816

### mean_abs_error
- Patterns détectés : 1
- Corrélation moyenne : 0.668
- Corrélation max : 0.668

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
