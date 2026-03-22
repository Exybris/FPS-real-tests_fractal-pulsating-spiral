# FPS — Fractal Pulsating Spiral

Système cybernétique oscillatoire fondé sur un réseau d'oscillateurs adaptatifs métastables dotés d'une régulation endogène.

Le notebook de référence complet se trouve dans `/notebooks`. Le pipeline reproduit fidèlement la même dynamique en ajoutant l'orchestration batch, la comparaison multi-modes et la génération automatique de rapports.

---

## Présentation

La FPS est un système cybernétique oscillatoire fondé sur un réseau d'oscillateurs adaptatifs métastables dotés d'une régulation endogène. Elle se situe entre les modèles descriptifs (Kuramoto) et prescriptifs (contrôleur PID) en simulant un système qui s'auto-régule autour de sept métriques de performance sur un signal non-stationnaire. L'hypothèse centrale est que *les systèmes les plus efficaces sont structurellement considérés et considérants*, une régulation parcimonieuse et contextuelle peut améliorer la performance en réduisant les oscillations inutiles.

Le système repose sur une séparation **perception spécialiste / action généraliste** : le signal global O(t) (somme des oscillateurs) sert d'observable unique pour l'évaluation multi-métriques et la construction de l'état cible émergent E(t). Un prior perceptif S(t) (sélectionné selon le déficit dominant parmi les scores calculés sur O(t) ) fournit une vue filtrée sur laquelle γ(t) et G(x) ajustent la régulation. Cette indirection (O → scores → S → métriques → γ, G → feedback) préserve l'émergence tout en rendant la régulation pertinente.

---

## Démarrage rapide

### Installation

```bash
git clone https://github.com/Exybris/FPS-real-tests_fractal-pulsating-spiral
cd FPS_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Exécution

```bash
# Pipeline complet : FPS + Kuramoto + Neutral + batch + exploration + visualisations
python3 main.py complete --config config.json

# Run simple FPS uniquement
python3 main.py run --config config.json --mode FPS

# Mode strict (arrêt si NaN/Inf détecté)
python3 simulate.py --config config.json --mode FPS --strict

# Validation de la configuration seule
python3 main.py validate --config config.json

# Batch de simulations (5 runs par défaut, configurable)
python3 main.py batch --config config.json

# Comparaison FPS vs Kuramoto
python3 main.py compare --config config.json
```

---

## Architecture

```
FPS_Project/
├── main.py                # Orchestrateur : pipeline complet, batch, comparaison
├── validate_config.py     # Validation exhaustive du config.json (1er appel)
├── init.py                # Initialisation strates, seeds, poids spiralés, logging
├── perturbations.py       # Input contextuel In(t), perturbations composites
├── dynamics.py            # Cœur FPS : An(t), fn(t), φn(t), On(t), En(t), γ(t), G(x)
├── regulation.py          # Fonctions G(x), enveloppes, réponse locale Gn
├── metrics.py             # 7 scores + 38 métriques loguées, scoring multi-fenêtres
├── simulate.py            # Boucle de simulation FPS/Kuramoto/Neutral
├── explore.py             # Détection d'émergences, fractals, anomalies
├── analyze.py             # Analyse batch, raffinement automatique des seuils
├── visualize.py           # Visualisations et rapport HTML
├── utils.py               # Utilitaires, batch_runner, gestion fichiers
├── config.json            # Configuration complète du système
└── notebooks/
    └── NOTEBOOK_FPS.ipynb # Notebook de référence (implémentation alignée)
```

### Flux d'exécution (`main.py complete`)

1. Validation complète du `config.json` via `validate_config.py`
2. Initialisation de l'environnement, strates et dossiers de sortie
3. Simulation FPS (run principal avec `simulate.py`) :
   - Input In(t) → amplitudes An(t), fréquences fn(t), phases φn(t)
   - Sorties individuelles On(t) = An · sin(2π · ∫fn·dt + φn)
   - Signal global O(t) = Σn On(t)
   - Prior prospectif E(t) par trace lissée et différée de O(t)
   - Scores multi-métriques calculés sur O(t) → sélection du prior perceptif S(t)
   - Latence γ(t) et régulation G(x) ajustées sur métriques calculées via S(t)
   - Feedback Fn(t) réinjecté dans les oscillateurs
4. Simulation Kuramoto (contrôle oscillateurs classiques)
5. Simulation Neutral (contrôle sans rétroaction)
6. Exploration des émergences via `explore.py`
7. Analyse batch (5 runs FPS pour validation statistique) via `analyze.py`
8. Génération des visualisations via `visualize.py`
9. Rapport final HTML avec comparaison des trois modes

---

## Dynamique FPS — Équations implémentées

### Boucle de simulation (un pas de temps)

À chaque pas `t`, dans cet ordre :

1. **Input contextuel** : `In(t) = Offset + Gain · tanh(Σ wᵢ·Pertᵢ(t) / scale)`
2. **Amplitudes** : `An(t) = A₀ · σ(In(t)) · env(x,t) · Fn_A(t)` — produit énergie × contexte adouci × focus × feedback
3. **Fréquences** : `fn(t) = f₀ · Δfn(t) · βn(t) · Fn_f(t)` — produit socle × couplage × plasticité × latence
4. **Phases** : `φn(t) = φsignature,n + personal_spiral + global_influence + inter_strata_influence`
5. **Sorties individuelles** : `On(t) = An(t) · sin(2π · ∫fn(t)dt + φn(t))`
6. **Signal global** : `O(t) = Σn On(t)` — observable unique
7. **Prior prospectif** : `En(t) = (1-λ) · En(t-dt) + λ · κ · On(t-T)` — trace lissée et différée
8. **Scores sur O(t)** : 7 scores normalisés (stabilité, régulation, fluidité, résilience, innovation, effort, CPU)
9. **Sélection du prior perceptif S(t)** : prior correspondant au score le plus faible, ou mode neutre S(t)=O(t) si tous satisfaisants
10. **Latence** : `γ(t)` via `adaptive_aware` — optimise les scores calculés sur S(t)
11. **Régulation** : `G(x)` adaptatif — transforme l'erreur (E-O) en correction bornée, archétype choisi selon contexte
12. **Feedback** : `Fn_A(t) = βn · G_value` (→ amplitude) et `Fn_f(t) = βn · γ` (→ fréquence)
13. **Mise à jour A₀** : `A₀ ← A₀(1-ρ) + An(t)·ρ`, plancher min_amplitude (mémoire lente d'énergie)
14. **38 métriques** calculées et loguées

### Signal global O(t) et prior perceptif S(t)

```
O(t) = Σₙ Aₙ(t) · sin(2π · ∫fₙ(t)dt + φₙ(t))
```

O(t) est l'observable unique : on y calcule les métriques, on y sélectionne le prior perceptif, on y construit E(t).

**Prior perceptif S(t)** — deux modes :

- **Neutre** (tous scores satisfaisants) : `S(t) = O(t)`
- **Régulation** (score de régulation le plus faible) : `S(t) = Σₙ [Aₙ · sin(2π·∫fₙdt + φₙ) · γₙ(t)] · G(Eₙ(t) - Oₙ(t))`

Le prior met en exergue ce qui peine, sans spécifier la nature du signal — γ et G travaillent sur les métriques évaluées à travers S(t), pas directement sur O(t).

### Enveloppe adaptative

```
Aₙ(t) = A₀ · σ(Iₙ(t)) · env(x, t) · Fn_A(t)
```

Où `env(x,t) = exp(−½((x−μₙ(t))/σₙ(t))²)` (gaussienne) ou sigmoïde douce, et `σₙ(t) = offset + amp·sin(2π·freq·t/T)` en mode dynamique. L'enveloppe localise la régulation autour de μ — la correction est située (comme un projecteur), pas uniforme.

### Prior prospectif En(t)

```
Eₙ(t) = (1 - λ) · Eₙ(t-dt) + λ · κ · Oₙ(t-T)
```

Trace lissée et différée du signal : E(t) est informé par O(t) sans en être une copie instantanée, et n'agit pas directement sur O(t) en retour. Le délai T évite la boucle « miroir » instantanée, et (1-λ) donne de l'inertie à l'anticipation. κ (gain de couplage) est adaptatif selon l'effort, oscillant entre -1 et 1.618.

### Effort d'adaptation

```
effort(t) = Σₙ [|ΔAₙ|/(|Aₙ|+ε) + |Δfₙ|/(|fₙ|+ε) + |Δγₙ|/(|γₙ|+ε)]
```

Saturé à `MAX_EFFORT = 100.0` avec epsilon adaptatif à l'échelle des références.

### Latence γ(t) adaptive_aware

γ(t) est un gain d'intégration qui module les fréquences via Fn_f(t). En mode `adaptive_aware`, γ apprend quelles valeurs maximisent la moyenne des 7 scores calculés sur le prior perceptif S(t) courant, en tenant compte des synergies avec G :

```
γ(t) = Π[0.1,1.0] (γ(t-Δt) + η_γ · ∇_γ Score(S(t)))
```

- Phase d'exploration systématique des combinaisons (γ, G)
- Calcul de `synergy_score = mean_perf · stability · (1 + growth)` par couple
- Détection de synergies exceptionnelles (score > 4.5)
- Mode transcendant synergique avec micro-oscillations autour de l'optimum
- Signaux de communication γ→G pour suggérer des changements d'archétype

### Régulation G(x) adaptive_aware

G(x) transforme l'erreur (E(t)-O(t)) en signal de correction bornée, transmis à l'amplitude via Fn_A(t). γ et G ne cherchent pas à optimiser directement O(t), mais les scores multi-métriques évalués via S(t).

Archétypes disponibles :
- `tanh(λx)` pour la saturation douce (γ faible)
- `sinc(x) = sin(x)/x` pour les oscillations amorties
- `sin(βx)·exp(-αx²)` pour la résonance localisée (γ élevé)
- `sign(x)·log(1+α|x|)·sin(βx)` pour la spirale logarithmique (γ élevé)
- `α·tanh + (1-α)·spiral_log` pour le mode adaptatif (zone intermédiaire)
- Rotation temporelle et transitions douces entre archétypes
- Mémoire d'efficacité contextuelle par couple (G, γ_bucket, error_bucket)

---

## Configuration

Le fichier `config.json` contrôle l'intégralité du comportement. Voici les blocs principaux :

### Système

```json
{
  "system": {
    "N": 10,
    "T": 50,
    "dt": 0.1,
    "seed": 12345,
    "mode": "FPS",
    "signal_mode": "extended"
  }
}
```

`signal_mode: "extended"` active la modulation par γn(t) et G(x) dans le prior perceptif S(t) (mode régulation).

### Input contextuel

```json
{
  "input": {
    "mode": "classic",
    "baseline": {
      "offset_mode": "static",
      "offset": 0.1,
      "gain_mode": "static",
      "gain": 1.0
    },
    "scale": 1.2,
    "perturbations": [
      {"type": "none", "amplitude": 2.0, "t0": 0.0, "freq": 100.0, "weight": 1.0}
    ]
  }
}
```

Types de perturbations disponibles : `none`, `choc`, `rampe`, `sinus`, `bruit`. L'offset et le gain peuvent être `adaptive` (s'auto-ajustent selon σ(In) et la saturation).

### Couplage

```json
{
  "coupling": {
    "type": "spiral",
    "c": 0.1,
    "closed": false,
    "mirror": false
  }
}
```

Types : `spiral` (poids générés automatiquement avec décroissance en distance), `ring` (spirale fermée). Les poids sont normalisés pour que Σw=0 par strate (conservation du signal).

### Paramètres dynamiques

```json
{
  "dynamic_parameters": {
    "dynamic_phi": true,
    "dynamic_beta": true,
    "dynamic_alpha": false,
    "dynamic_gamma": true,
    "dynamic_G": true
  }
}
```

Chaque paramètre peut être figé (`false`) ou adaptatif (`true`). `dynamic_phi` active la contrainte spiralée r(t) = φ + ε·sin(2π·ω·t + θ).

### Latence et régulation

```json
{
  "latence": {
    "gamma_mode": "adaptive_aware"
  },
  "regulation": {
    "G_arch": "adaptive_aware",
    "phi_mode": "adaptive",
    "lambda_E": 0.1,
    "phi_adaptive": {
      "effort_low": 0.5,
      "effort_high": 5,
      "phi_min": 0.9,
      "phi_max": 1.618
    }
  }
}
```

Modes gamma disponibles : `static`, `dynamic`, `adaptive_aware`. Archétypes G : `tanh`, `sinc`, `resonance`, `spiral_log`, `adaptive`, `adaptive_aware`.

### Enveloppe

```json
{
  "enveloppe": {
    "env_type": "gaussienne",
    "env_mode": "dynamic",
    "sigma_n_dynamic": {
      "amp": 0.1,
      "freq": 0.3,
      "offset": 0.1
    }
  }
}
```

L'enveloppe module l'amplitude An(t) selon l'erreur En-On. En mode `dynamic`, σn(t) oscille dans le temps.

### Tests chimériques

```json
{
  "chimera_tests": {
    "uniform_frequencies": {"enabled": false, "value": 1.0},
    "reset_frequencies_midrun": {"enabled": false, "t_reset": 0.5},
    "reset_phases_midrun": {"enabled": false, "t_reset": 0.5}
  }
}
```

Ces tests vérifient si les comportements émergents (états chimériques) sont intrinsèques à l'architecture FPS ou artefacts des conditions initiales.

### Exploration

```json
{
  "exploration": {
    "detect_fractal_patterns": true,
    "detect_anomalies": true,
    "detect_harmonics": true,
    "anomaly_threshold": 3.0,
    "fractal_threshold": 0.8,
    "window_sizes": [1, 10, 100]
  }
}
```

### Validation batch

```json
{
  "validation": {
    "batch_size": 5,
    "criteria": ["fluidity", "stability", "resilience", "innovation",
                 "regulation", "cpu_cost", "effort_internal", "effort_transient"]
  }
}
```

---

## Métriques (38 loguées par pas de temps)

### Signaux globaux

| Métrique | Description |
|----------|-------------|
| `O(t)` | Signal global observable (somme des contributions de chaque strate) |
| `S(t)` | Prior perceptif (vue filtrée de O(t) selon déficit dominant) |
| `C(t)` | Coefficient d'accord spiralé (cohérence des phases adjacentes) |
| `A_spiral(t)` | Amplitude spirale (modulation fréquentielle globale) |
| `E(t)` | Prior prospectif (trace lissée et différée de O(t)) |
| `L(t)` | Indice de la strate leader (max dAn/dt) |

### Adaptation

| Métrique | Description |
|----------|-------------|
| `An_mean(t)` | Amplitude moyenne des strates |
| `fn_mean(t)` | Fréquence moyenne des strates |
| `En_mean(t)` | Sortie attendue moyenne |
| `On_mean(t)` | Sortie observée moyenne |
| `In_mean(t)` | Input contextuel moyen |
| `gamma` | Latence globale γ(t) |
| `gamma_mean(t)` | Latence moyenne par strate γn(t) |
| `G_arch_used` | Archétype de régulation utilisé |

### Performance

| Métrique | Description |
|----------|-------------|
| `effort(t)` | Effort d'adaptation interne (changement relatif) |
| `effort_status` | Statut : `stable`, `transitoire`, `chronique` |
| `mean_high_effort` | Percentile 80 de l'effort (effort chronique) |
| `d_effort_dt` | Dérivée temporelle de l'effort (pics transitoires) |
| `mean_abs_error` | Erreur moyenne |En-On| (qualité de la régulation) |
| `cpu_step(t)` | Temps CPU par strate par pas |

### Stabilité et résilience

| Métrique | Description |
|----------|-------------|
| `variance_d2S` | Variance de la dérivée seconde de S (accélération) |
| `fluidity` | Fluidité du signal (sigmoïde inversée sur variance_d2S) |
| `entropy_S` | Entropie spectrale normalisée (Shannon sur spectre de puissance = innovation) |
| `max_median_ratio` | Ratio max/médiane de S (détection d'outliers) |
| `continuous_resilience` | Résilience sous perturbation continue |
| `adaptive_resilience` | Résilience adaptative multi-critères |
| `t_retour` | Temps de retour après choc |
| `temporal_coherence` | Cohérence temporelle (mémoire douce du signal) |

### Échelles temporelles

| Métrique | Description |
|----------|-------------|
| `tau_S` | Temps caractéristique du signal S(t) |
| `tau_gamma` | Temps caractéristique de γ(t) |
| `tau_C` | Temps caractéristique de C(t) |
| `tau_A_mean` | Temps caractéristique des amplitudes |
| `tau_f_mean` | Temps caractéristique des fréquences |
| `autocorr_tau` | Tau d'autocorrélation effectif |
| `decorrelation_time` | Temps de décorrélation |

### Découverte (γ, G)

| Métrique | Description |
|----------|-------------|
| `best_pair_gamma` | γ du meilleur couple découvert |
| `best_pair_G` | Archétype G du meilleur couple |
| `best_pair_score` | Score de synergie du meilleur couple |

### Scoring adaptatif (1-5)

Le système calcule 7 scores normalisés avec fenêtres multi-échelles (immediate, recent, medium, global) et pondération selon la maturité de la simulation. Les scores ont un double rôle : calculés sur O(t) ils sélectionnent le prior perceptif S(t) ; calculés sur S(t) ils pilotent γ et G.

- **stability** : basé sur std(S)
- **regulation** : basé sur mean_abs_error
- **fluidity** : basé sur la fluidité moyenne
- **resilience** : adaptive_resilience → continuous_resilience → C(t) proxy
- **innovation** : basé sur entropy_S
- **cpu_cost** : basé sur cpu_step
- **effort** : basé sur mean_effort

---

## Sorties

### Structure des fichiers générés

```
fps_pipeline_output/run_YYYYMMDD_HHMMSS/
├── logs/
│   ├── run_*_FPS_seed*.csv           # Métriques principales (38 colonnes, 1 ligne/step)
│   ├── batch_run_*_FPS_seed*.csv     # Runs batch pour validation statistique
│   ├── stratum_details_*.csv         # Signaux individuels par strate
│   ├── log_plus_delta_fn_Si_*.csv    # delta_fn, S_i, f0n, An par strate
│   ├── debug_detailed_*.csv          # Log détaillé tous les 10 steps
│   ├── emergence_events_*.csv        # Événements d'émergence détectés
│   ├── fractal_events_*.csv          # Patterns fractals détectés
│   └── seeds.txt                     # Traçabilité des seeds
├── figures/
│   ├── signal_evolution_fps.png
│   ├── metrics_dashboard.png
│   ├── discovery_timeline.png
│   ├── scores_evolution.png
│   └── ...
├── reports/
│   ├── comparison_fps_vs_controls.json
│   └── rapport_complet_fps.html
├── configs/
│   └── config_*_main.json            # Config utilisée (snapshot)
└── checkpoints/
    └── *_backup_*.pkl                # Sauvegardes d'état (tous les 100 steps)
```

### Indicateurs d'un run sain

- `effort(t)` oscille sans saturer constamment à 100
- `entropy_S` > 0.3 (le système innove)
- `fluidity` > 0.9 (pas de sauts brusques)
- `mean_abs_error` < 0.05 (régulation effective)
- `C(t)` > 0.99 (cohérence des phases maintenue)
- `best_pair_score` converge vers > 4.5 (synergie trouvée)

---

## Outils complémentaires

### Agrégation des logs

```bash
pip install pyarrow
python3 aggregate_all.py -o aggregated/fps_dataset.h5
python3 aggregate_all.py --metrics "S(t),A_mean(t),effort(t)"
```

### Visualisation des dynamiques individuelles par strate

```bash
python3 visualize_individual.py
```

### Corrélations tau-performances

```bash
python3 analyze_temporal_correlations.py
```

---

## Alignement Pipeline ↔ Notebook

Le pipeline et le notebook implémentent la même dynamique FPS et produisent des résultats identiques (377/500 steps parfaits, divergences résiduelles < 0.01% sur les métriques principales). Les deux partagent les mêmes modules `init.py`, `utils.py`, `visualize.py` et `explore.py`.

Différences structurelles intentionnelles :
- Le pipeline orchestre 3 modes (FPS, Kuramoto, Neutral) + batch + exploration automatique
- Le notebook offre un environnement interactif avec visualisations inline
- Le pipeline sauvegarde des checkpoints et des logs détaillés par strate

---

## Reproductibilité

Avec la même seed et le même `config.json` :
- Run principal et batch run 0 sont bit-pour-bit identiques
- Pipeline et notebook produisent les mêmes valeurs sur toutes les métriques principales
- Les strates sont auto-générées par `generate_strates(N, seed)` avec `RandomState` isolé

---

*FPS v3 — Système oscillatoire métastable à émergence fractale*
*© 2025 Andréa Gadal — Recherche indépendante (Exybris)*
*Avec les contributions de Gepetto, Claude & Gemini*
