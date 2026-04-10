*Read this in other languages: [🇫🇷 Français](README.fr.md), [🇬🇧 English](README.md).*

-----

# FPS — Fractal Pulsating Spiral

Oscillatory cybernetic system based on a network of metastable adaptive oscillators with endogenous regulation.

The complete reference notebook is located in `/notebooks`. The pipeline faithfully reproduces the same dynamics while adding batch orchestration, multi-mode comparison, and automatic report generation.

-----

## Overview

The FPS is an oscillatory cybernetic system based on a network of metastable adaptive oscillators with endogenous regulation. It sits between descriptive models (Kuramoto) and prescriptive models (PID controller) by simulating a system that self-regulates around seven performance metrics on a non-stationary signal. The central hypothesis is that *the most efficient systems are structurally considerate*, a parsimonious and contextual regulation can improve performance by reducing unnecessary oscillations.

The system relies on a **specialist perception / generalist action** separation: the global signal O(t) (sum of the oscillators) serves as the single observable for multi-metric evaluation and the construction of the emergent target state E(t). A perceptual prior S(t), selected according to the dominant deficit among scores calculated on O(t), provides a filtered view on which γ(t) and G(x) adjust the regulation. This indirection (O → scores → S → metrics → γ, G → feedback) preserves emergence while making the regulation relevant.

-----

## Quick Start

### Installation

```bash
git clone https://github.com/Exybris/FPS-real-tests_fractal-pulsating-spiral
cd FPS_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Execution

```bash
# Complete pipeline: FPS + Kuramoto + Neutral + batch + exploration + visualizations
python3 main.py complete --config config.json

# Simple FPS run only
python3 main.py run --config config.json --mode FPS

# Strict mode (stops if NaN/Inf detected)
python3 simulate.py --config config.json --mode FPS --strict

# Configuration validation only
python3 main.py validate --config config.json

# Batch simulations (5 runs by default, configurable)
python3 main.py batch --config config.json

# FPS vs Kuramoto comparison
python3 main.py compare --config config.json
```

-----

## Architecture

```
FPS_Project/
├── main.py                # Orchestrator: complete pipeline, batch, comparison
├── validate_config.py     # Exhaustive config.json validation (1st call)
├── init.py                # Strata initialization, seeds, spiral weights, logging
├── perturbations.py       # Contextual input In(t), composite perturbations
├── dynamics.py            # FPS core: An(t), fn(t), φn(t), On(t), En(t), γ(t), G(x)
├── regulation.py          # G(x) functions, envelopes, local response Gn
├── metrics.py             # 7 scores + 38 logged metrics, multi-window scoring
├── simulate.py            # FPS/Kuramoto/Neutral simulation loop
├── explore.py             # Detection of emergences, fractals, anomalies
├── analyze.py             # Batch analysis, automatic threshold refinement
├── visualize.py           # Visualizations and HTML report
├── utils.py               # Utilities, batch_runner, file management
├── config.json            # Complete system configuration
└── notebooks/
    └── NOTEBOOK_FPS.ipynb # Reference notebook (aligned implementation)
```

### Execution Flow (`main.py complete`)

1. Complete validation of `config.json` via `validate_config.py`
1. Environment, strata, and output folders initialization
1. FPS simulation (main run with `simulate.py`):
- Input In(t) → amplitudes An(t), frequencies fn(t), phases φn(t) (`dynamics.py`)
- Individual outputs On(t) = An · sin(2π · ∫fn·dt + φn) (`dynamics.py`)
- Global signal O(t) = Σn On(t) (`dynamics.py`)
- Prospective prior E(t) via smoothed and delayed trace of O(t) (`dynamics.py`)
- Multi-metric scores calculated on O(t) → perceptual prior S(t) selection
- Latency γ(t) and regulation G(x) adjusted on metrics calculated via S(t)
- Feedback Fn(t) fed back into the oscillators
1. Kuramoto simulation (classic oscillator control)
1. Neutral simulation (control without feedback)
1. Emergence exploration via `explore.py`
1. Batch analysis (5 FPS runs for statistical validation) via `analyze.py`
1. Visualization generation via `visualize.py`
1. Final HTML report with comparison of the three modes

-----

## FPS Dynamics: Implemented Equations

### Simulation Loop (one time step)

At each step `t`, in this order:

1. **Contextual input**: `In(t) = Offset + Gain · tanh(Σ wᵢ·Pertᵢ(t) / scale)`
1. **Amplitudes**: `An(t) = A₀ · σ(In(t)) · env(x,t) · (1 + Fn_A(t))` — product of energy × softened context × focus × feedback
1. **Frequencies**: `fn(t) = f₀ · Δfn(t) · βn(t) · (1 + Fn_f(t))` — product of base × coupling × plasticity × latency
1. **Phases**: `φn(t) = φsignature,n + personal_spiral + global_influence + inter_strata_influence`
1. **Individual outputs**: `On(t) = An(t) · sin(2π · ∫fn(t)dt + φn(t))`
1. **Global signal**: `O(t) = Σn On(t)` — single observable
1. **Prospective prior**: `En(t) = (1-λ) · En(t-dt) + λ · κ · On(t-T)` — smoothed and delayed trace
1. **Scores on O(t)**: 7 normalized scores (stability, regulation, fluidity, resilience, innovation, effort, CPU)
1. **Perceptual prior S(t) selection**: prior corresponding to the lowest score, or neutral mode S(t)=O(t) if all are satisfactory
1. **Latency**: `γ(t)` via `adaptive_aware` — optimizes scores calculated on S(t)
1. **Regulation**: `G(x)` adaptive — transforms error (E-O) into bounded correction, archetype chosen according to context
1. **Feedback**: `Fn_A(t) = βn · G_value` (→ amplitude) and `Fn_f(t) = βn · γ` (→ frequency)
1. **A₀ update**: `A₀ ← A₀(1-ρ) + An(t)·ρ`, floor min_amplitude (slow energy memory)
1. **38 metrics** calculated and logged

### Global Signal O(t) and Perceptual Prior S(t)

```
O(t) = Σₙ Aₙ(t) · sin(2π · ∫fₙ(t)dt + φₙ(t))
```

O(t) is the single observable: metrics are calculated on it, the perceptual prior is selected from it, E(t) is built from it.

**Perceptual prior S(t)** — two modes:

- **Neutral** (all scores satisfactory): `S(t) = O(t)`
- **Regulation** (lowest regulation score): `S(t) = Σₙ [Aₙ · sin(2π·∫fₙdt + φₙ) · γₙ(t)] · G(Eₙ(t) - Oₙ(t))`

The prior highlights what is struggling, without specifying the nature of the signal — γ and G work on metrics evaluated through S(t), not directly on O(t).

### Adaptive Envelope

```
Aₙ(t) = A₀ · σ(Iₙ(t)) · env(x, t) · Fn_A(t)
```

Where `env(x,t) = exp(−½((x−μₙ(t))/σₙ(t))²)` (Gaussian) or soft sigmoid, and `σₙ(t) = offset + amp·sin(2π·freq·t/T)` in dynamic mode. The envelope localizes regulation around μ — the correction is situated (like a spotlight), not uniform.

### Prospective Prior En(t)

```
Eₙ(t) = (1 - λ) · Eₙ(t-dt) + λ · κ · Oₙ(t-T)
```

Smoothed and delayed trace of the signal: E(t) is informed by O(t) without being an instantaneous copy of it, and does not act directly on O(t) in return. The delay T avoids the instantaneous “mirror” loop, and (1-λ) gives inertia to the anticipation. κ (coupling gain) is adaptive according to effort, oscillating between -1 and 1.618.

### Adaptation Effort

```
effort(t) = Σₙ [|ΔAₙ|/(|Aₙ|+ε) + |Δfₙ|/(|fₙ|+ε) + |Δγₙ|/(|γₙ|+ε)]
```

Saturated at `MAX_EFFORT = 100.0` with adaptive epsilon at the scale of references.

### Latency γ(t) adaptive_aware

γ(t) is an integration gain that modulates frequencies via Fn_f(t). In `adaptive_aware` mode, γ learns which values maximize the average of the 7 scores calculated on the current perceptual prior S(t), taking into account synergies with G:

```
γ(t) = Π[0.1,1.0] (γ(t-Δt) + η_γ · ∇_γ Score(S(t)))
```

- Systematic exploration phase of (γ, G) combinations
- `synergy_score = mean_perf · stability · (1 + growth)` calculated per pair
- Detection of exceptional synergies (score > 4.5)
- Synergistic transcendent mode with micro-oscillations around the optimum
- Communication signals γ→G to suggest archetype changes

### Regulation G(x) adaptive_aware

G(x) transforms the error (E(t)-O(t)) into a bounded correction signal, transmitted to amplitude via Fn_A(t). γ and G do not seek to optimize O(t) directly, but rather the multi-metric scores evaluated via S(t).

Available archetypes:

- `tanh(λx)` for soft saturation (low γ)
- `sinc(x) = sin(x)/x` for damped oscillations
- `sin(βx)·exp(-αx²)` for localized resonance (high γ)
- `sign(x)·log(1+α|x|)·sin(βx)` for logarithmic spiral (high γ)
- `α·tanh + (1-α)·spiral_log` for adaptive mode (intermediate zone)
- Temporal rotation and smooth transitions between archetypes
- Contextual efficiency memory per pair (G, γ_bucket, error_bucket)

-----

## Configuration

The `config.json` file controls the entire behavior. Here are the main blocks:

### System

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

`signal_mode: "extended"` activates modulation by γn(t) and G(x) in the perceptual prior S(t) (regulation mode).

### Contextual Input

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

Available perturbation types: `none`, `choc`, `rampe`, `sinus`, `bruit`. Offset and gain can be `adaptive` (self-adjusting according to σ(In) and saturation).

### Coupling

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

Types: `spiral` (automatically generated weights with distance decay), `ring` (closed spiral). Weights are normalized so that Σw=0 per stratum (signal conservation).

### Dynamic Parameters

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

Each parameter can be fixed (`false`) or adaptive (`true`). `dynamic_phi` activates the spiral constraint `r(t) = φ + ε·sin(2π·ω·t + θ)`.

### Latency and Regulation

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

Available gamma modes: `static`, `dynamic`, `adaptive_aware`. G archetypes: `tanh`, `sinc`, `resonance`, `spiral_log`, `adaptive`, `adaptive_aware`.

### Envelope

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

The envelope modulates amplitude An(t) according to the error En-On. In `dynamic` mode, σn(t) oscillates over time.

### Chimera Tests

```json
{
  "chimera_tests": {
    "uniform_frequencies": {"enabled": false, "value": 1.0},
    "reset_frequencies_midrun": {"enabled": false, "t_reset": 0.5},
    "reset_phases_midrun": {"enabled": false, "t_reset": 0.5}
  }
}
```

These tests verify whether emergent behaviors (chimera states) are intrinsic to the FPS architecture or artifacts of initial conditions.

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

### Batch Validation

```json
{
  "validation": {
    "batch_size": 5,
    "criteria": ["fluidity", "stability", "resilience", "innovation",
                 "regulation", "cpu_cost", "effort_internal", "effort_transient"]
  }
}
```

-----

## Metrics (38 logged per time step)

### Global Signals

|Metric       |Description                                                           |
|-------------|----------------------------------------------------------------------|
|`O(t)`       |Observable global signal (sum of each stratum’s contributions)        |
|`S(t)`       |Perceptual prior (filtered view of O(t) according to dominant deficit)|
|`C(t)`       |Spiral agreement coefficient (adjacent phase coherence)               |
|`A_spiral(t)`|Spiral amplitude (global frequency modulation)                        |
|`E(t)`       |Prospective prior (smoothed and delayed trace of O(t))                |
|`L(t)`       |Leading stratum index (max dAn/dt)                                    |

### Adaptation

|Metric         |Description                      |
|---------------|---------------------------------|
|`An_mean(t)`   |Average stratum amplitude        |
|`fn_mean(t)`   |Average stratum frequency        |
|`En_mean(t)`   |Average expected output          |
|`On_mean(t)`   |Average observed output          |
|`In_mean(t)`   |Average contextual input         |
|`gamma`        |Global latency γ(t)              |
|`gamma_mean(t)`|Average per-stratum latency γn(t)|
|`G_arch_used`  |Regulation archetype used        |

### Performance

|Metric            |Description                                                       |
|------------------|------------------------------------------------------------------|
|`effort(t)`       |Internal adaptation effort (relative change)                      |
|`effort_status`   |Status: `stable`, `transitoire` (transient), `chronique` (chronic)|
|`mean_high_effort`|80th percentile of effort (chronic effort)                        |
|`d_effort_dt`     |Time derivative of effort (transient peaks)                       |
|`mean_abs_error`  |Mean error                                                        |
|`cpu_step(t)`     |CPU time per stratum per step                                     |

### Stability and Resilience

|Metric                 |Description                                                         |
|-----------------------|--------------------------------------------------------------------|
|`variance_d2S`         |Variance of the second derivative of S (acceleration)               |
|`fluidity`             |Signal fluidity (inverted sigmoid on variance_d2S)                  |
|`entropy_S`            |Normalized spectral entropy (Shannon on power spectrum = innovation)|
|`max_median_ratio`     |Max/median ratio of S (outlier detection)                           |
|`continuous_resilience`|Resilience under continuous perturbation                            |
|`adaptive_resilience`  |Multi-criteria adaptive resilience                                  |
|`t_retour`             |Recovery time after shock                                           |
|`temporal_coherence`   |Temporal coherence (soft signal memory)                             |

### Time Scales

|Metric              |Description                       |
|--------------------|----------------------------------|
|`tau_S`             |Characteristic time of signal S(t)|
|`tau_gamma`         |Characteristic time of γ(t)       |
|`tau_C`             |Characteristic time of C(t)       |
|`tau_A_mean`        |Characteristic time of amplitudes |
|`tau_f_mean`        |Characteristic time of frequencies|
|`autocorr_tau`      |Effective autocorrelation tau     |
|`decorrelation_time`|Decorrelation time                |

### Discovery (γ, G)

|Metric           |Description                   |
|-----------------|------------------------------|
|`best_pair_gamma`|γ of the best discovered pair |
|`best_pair_G`    |G archetype of the best pair  |
|`best_pair_score`|Synergy score of the best pair|

### Adaptive Scoring (1-5)

The system calculates 7 normalized scores with multi-scale windows (immediate, recent, medium, global) and weighting according to the simulation’s maturity. Scores have a dual role: calculated on O(t) they select the perceptual prior S(t); calculated on S(t) they drive γ and G.

- **stability**: based on std(S)
- **regulation**: based on mean_abs_error
- **fluidity**: based on average fluidity
- **resilience**: adaptive_resilience → continuous_resilience → C(t) proxy
- **innovation**: based on entropy_S
- **cpu_cost**: based on cpu_step
- **effort**: based on mean_effort

-----

## Outputs

### Generated Files Structure

```
fps_pipeline_output/run_YYYYMMDD_HHMMSS/
├── logs/
│   ├── run_*_FPS_seed*.csv           # Main metrics (38 columns, 1 row/step)
│   ├── batch_run_*_FPS_seed*.csv     # Batch runs for statistical validation
│   ├── stratum_details_*.csv         # Individual signals per stratum
│   ├── log_plus_delta_fn_Si_*.csv    # delta_fn, S_i, f0n, An per stratum
│   ├── debug_detailed_*.csv          # Detailed log every 10 steps
│   ├── emergence_events_*.csv        # Detected emergence events
│   ├── fractal_events_*.csv          # Detected fractal patterns
│   └── seeds.txt                     # Seed traceability
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
│   └── config_*_main.json            # Used config (snapshot)
└── checkpoints/
    └── *_backup_*.pkl                # State backups (every 100 steps)
```

### Healthy Run Indicators

- `effort(t)` oscillates without constantly saturating at 100
- `entropy_S` > 0.3 (the system innovates)
- `fluidity` > 0.9 (no sudden jumps)
- `mean_abs_error` < 0.05 (effective regulation)
- `C(t)` > 0.99 (phase coherence maintained)
- `best_pair_score` converges to > 4.5 (synergy found)

-----

## Complementary Tools

### Log Aggregation

```bash
pip install pyarrow
python3 aggregate_all.py -o aggregated/fps_dataset.h5
python3 aggregate_all.py --metrics "S(t),A_mean(t),effort(t)"
```

### Individual Per-Stratum Dynamics Visualization

```bash
python3 visualize_individual.py
```

### Tau-Performance Correlations

```bash
python3 analyze_temporal_correlations.py
```

-----

## Pipeline ↔ Notebook Alignment

The pipeline and the notebook implement the same FPS dynamics and produce identical results (377/500 perfect steps, residual divergences < 0.01% on main metrics). Both share the same modules `init.py`, `utils.py`, `visualize.py`, and `explore.py`.

Intentional structural differences:

- The pipeline orchestrates 3 modes (FPS, Kuramoto, Neutral) + batch + automatic exploration
- The notebook offers an interactive environment with inline visualizations
- The pipeline saves checkpoints and detailed per-stratum logs

-----

## Reproducibility

With the same seed and the same `config.json`:

- Main run and batch run 0 are bit-for-bit identical
- Pipeline and notebook produce the same values on all main metrics
- Strata are auto-generated by `generate_strates(N, seed)` with an isolated `RandomState`

-----

## Papers

The full technical paper describing the FPS architecture, equations, design intent, ablations, 
and ethical framework is available in this repository:

📄 [Oscillatory Metastable System — FPS (English)](oscillatory_metastable_system_FPS.md)

The FPS research note which highlights the results of the initial tests

📄 [Research Note — FPS (English)](FPS_Research_Note_eng.pdf)

-----

## Roadmap

### Architecture completion

- **Full S(t) prior library**: implement all seven perceptual priors (currently: neutral and regulation; planned: innovation, effort, resilience, fluidity, stability, CPU cost) — each as a specialized projection of O(t)
- **Metric-driven switching on O(t)**: formalize and harden the prior selection rule (lowest score → corresponding prior, all above threshold → neutral fallback)
- **μn(t) adaptive focus**: make the envelope center dynamic, linking it to γ and G so the system can shift *where* it regulates, not just *how much*

### Planned applications

- **FPS as a self-regulating reservoir**: use FPS oscillators as a reservoir computer (Echo State Networks like), where only the readout layer is trained — testing whether endogenous regulation produces richer, more robust representations on standard benchmarks
- **FPS as an attention modulator**: apply FPS modulation to attention scores in a small transformer, measuring whether bounded self-regulation reduces attention collapse, score variance, and vulnerability to adversarial perturbations
- **Toward oscillatory neural networks**: longer-term exploration where every neuron carries self-regulation by construction — FPS dynamics embedded at the unit level rather than as an external layer


-----

*FPS v3 — Metastable oscillatory system with fractal emergence*
*© 2025 Exybris — Independent research*
*With contributions from A.G., Gepetto, Claude, Semoka & Gemini*
