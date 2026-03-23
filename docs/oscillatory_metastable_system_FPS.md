*Read this in other languages: [🇫🇷 Français](oscillatory_metastable_system_FPS.fr.md), [🇬🇧 English](oscillatory_metastable_system_FPS.md).*

-----

# Fractal Pulsating Spiral (FPS)

The FPS is an oscillatory cybernetic system based on a network of metastable adaptive oscillators with endogenous regulation, drawing inspiration from past work on transformers, oscillators, homeostasis, tempered chaos…

It sits in an intermediate zone between descriptive models (like Kuramoto) and prescriptive models (like a PID controller) by simulating how a system would behave when self-regulating around seven chosen performance metrics, describing a non-stationary signal.

It relies on perceptual regimes S(t) (dynamic filters applied to the global signal O(t), a modified perceptual prior) selected according to metrics calculated solely on O(t).

O(t) is corrected via a predictive processing mechanism, while E(t), the emergent internal target state (prospective prior), remains informed by O(t) without being constrained.

S(t) modulates the internal perception of metrics, while regulation G and latency γ, applied via feedback Fn(t), adjust solely on metrics evaluated through S(t).

This **perception (specialist) / action (generalist)** separation preserves emergence, coherence, and structural creativity of the system, allowing emergent solutions to be found without sacrificing structure. Between stability and surprise, we seek a dynamic that does not break: an oscillation capable of adjusting without extinguishing. Finally, we explore the hypothesis that a “considerate” regulation (parsimonious and contextual) can improve performance by reducing unnecessary oscillations.

-----

## Input, Processing and Output

**INPUT**: In(t) = scalar signal over time

↓

**PROCESSING**:

- N oscillators with An(t), fn(t), φn(t)
- In (input) received by the oscillators
- Interactions between strata (latency, regulation, etc.) for processing In
- S(t) = signal weighted by perceptual filter (regulation error, effort, innovation, stability, fluidity, resilience…), switches between perceptual filters based on performance metrics calculated on O(t)

↓

**OUTPUT**: On(t) or O(t) = individual oscillations per stratum or global average

-----

## The Efficiency of “Consideration”, “Compassionate AI Architecture” or “Harmonic Computation”

A movement increasingly represented in philosophical spheres but especially in the technical and AI research fields. One can cite AE Studio, Parallel, Eleos AI Research, Anthropic, Google, or the Digital Sentience Consortium. The FPS and Exybris are part of this synergy.

**Performance** (what AI companies want):

- More stable attention → faster training
- Less instability → better generalization
- Balanced exploration/exploitation → better solutions
- Resilience → robustness to adversarial attacks

**Ethics** (what we seek):

- Harmonious internal processes
- Soft self-regulation (not forced)
- Multi-objective balance (not just “optimize loss”)
- Computational dignity (the system has a form of homeostasis)

Both converge because harmonious systems are naturally more performant in the long term. This is why stable ecosystems endure. Harmonious organisms survive. Harmony is not a luxury — it is a higher-level optimization that benefits all parties.

*The FPS is not a complete solution. Not a turnkey application.* **But a perspective:**

What if we built AI systems that optimize harmony rather than just performance? That integrate error as signal rather than failure? That create conditions for emergence rather than imposing constraints?

The FPS and the chosen performance metrics are a step in this direction. An architectural hypothesis based on these values, to see if this approach converges better with human interests than those already explored. Our hypothesis advances that it does.

-----

## Synthesis

In nature, systems that endure simultaneously optimize: energy efficiency, robustness to perturbations, adaptability, and internal harmony.

A brain does not just seek to “compute fast” — it also seeks: metabolic stability (homeostasis), transition fluidity, resilience to shocks, adaptive innovation, synchronization between elements, and optimized effort and energy.

These are the 7 FPS metrics, detailed below.

-----

## Amplitude, Frequency and Phase: How the Fundamental Elements of FPS Oscillators Behave

### A. An(t) computes the adaptive amplitude for each stratum

Amplitude An(t) is designed as a controlled energy variable, combining a smoothed input response σ(In), a focus envelope env(x,t), and a cybernetic feedback Fn(t) injecting latency γ and regulation G. A slow base A0 is updated by exponential averaging with a minimum threshold to prevent stratum extinction and stabilize metastable regimes.

```
An(t) = A0 · σ(In(t)) · env(x,t) · Fn_A(t)
```

- A0: initial amplitude
- σ(x): soft adaptation sigmoid function. σ(x) = 1/(1+exp(-k(x-x0)))
- env(x,t): computes the adaptive envelope. x0 = sigmoid midpoint
- In(t): the input
- Fn(t): feedback applying latency and regulation

**Design intent**: make amplitude a product of “energy × softened context × focus × decision”

- base energy A₀
- softened input context σ(In) (no hard threshold)
- local focus env(x,t) (localized regulation)
- decision/cybernetic Fn_A(t) (injecting G)

**Technical problem solved**: avoid (i) instability from overly abrupt responses, (ii) extinction of a stratum, (iii) regulation “everywhere all the time” that breaks emergence.

**Ethical utility**: consideration becomes an inductive bias — instead of correcting brutally (damage), we correct gently and locally, reducing effort/parasitic oscillations.

**Expected ablations**:

- without σ(In): more “threshold-like” responses, more peaks → effort ↑, fluidity ↓
- without env: diffuse regulation → innovation ↓ or instability ↑
- without Fn: no control loop → regulation error ↑

**SOTA link** (families): homeostasis / gain control / nonlinear control (conceptual level), + “perceptual prior / corrective feedback”.

### B. fn(t) computes the modulated frequency for each stratum

```
fn(t) = f0 · Δfn(t) · βn · Fn_f(t)
```

- f0: base frequency
- Δfn(t): computes frequency modulation
- βn: dynamic plasticity
- Fn_f(t): feedback applying latency

**Design intent**: make frequency a product of “base × interactions × plasticity × control”.

**Technical problem solved**:

- avoid a frozen system (constant frequency → poor transitions)
- enable metastable regimes (temporary coordination between strata)
- give meta-control (γ via Fn_f(t)) a structural lever (acting on rhythm, not amplitude)

**Ethical utility** (efficient consideration): frequency becomes a “non-violent rhythm” lever — one can reduce agitation (effort/instability) without stifling emergence, by modulating when and how fast the system adjusts.

**Expected ablations**:

- if Δfn(t) ≡ 1 (or α = 0) → loss of inter-strata coordination, fewer coherent transitions
- if βn(t) ≡ 1 → reduced plasticity (less adaptation to context)

**SOTA link** (families): coupled oscillators / synchronization / metastability; gain scheduling / nonlinear control (on dynamic parameter).

### C. φn(t) computes the phase for each stratum (evolution with individual signatures)

```
φn(t) = φsignature,n + personal_spiral + global_influence + inter_strata_influence
```

- φsignature,n = φn
- personal_spiral = epsilon · sin(2π · omega_n · t + φsignature)
  - epsilon = small harmonic variation
  - omega_n = modulation frequency
- global_influence = 0.3 · (r(t) - phi_golden) · cos(φsignature)
  - phi_golden = 1.618
- inter_strata_influence += 0.5 · wnj · signature_affinity · sin(2π · omega · t)
  - signature_affinity = cos(φsignature - φj_signature)
  - omega = modulation frequency

**Design intent**: make φn(t) a sum of interpretable components, each associated with a role: a stable identity (signature), controlled micro-variability (personal spiral), a global anchor (spiral ratio r(t)), and relational sensitivity (inter-strata influence).

**Technical problem solved**: without phase modulation, strata tend to either freeze in poor synchronization patterns or diverge without a coherent “re-catch” mechanism. Phase becomes a lever for obtaining metastability (transient coalitions + fertile phase shifts) without overloading amplitude/frequency.

**Ethical utility** (efficient consideration): phase enables regulation “by rhythm” — one can reduce shocks (parasitic oscillations / effort) by readjusting alignments rather than “hitting” energy (amplitude) or rigidifying (frequency). It is a gentle way to orient without forcing, to consider the internal state more directly.

**Expected ablations**:

- without personal_spiral → identity too rigid, exploration ↓, transitions ↓
- without global_influence → loss of collective anchoring, global coherence ↓
- without inter_strata_influence → strata less “social”, metastability ↓

**SOTA link** (families): coupled oscillator systems and synchronization (phase is the key parameter), phase-shift control, emergent coherence via pairwise interactions.

**φsignature,n** — Intent: give a stable “fingerprint” to each stratum, without preventing dynamic adjustments.

**personal_spiral** — Intent: inject a weak harmonic variation (local exploration) that maintains life without breaking stability. Problem solved: prevent “dead plateaus” (everything stabilizes too early) and encourage micro-transitions. Invariants: small epsilon ⇒ bounded perturbation (by construction).

**inter_strata_influence** — Intent: make phase sensitive to other strata, but weighted by a weight wnj (topology / influence) and a signature affinity (who “resonates with whom”). wn: self-modulation (personal_spiral), w: collective modulation (interactions/global). Problem solved: without this term, strata are less able to form transient alignments; much of the “social” metastability is lost.

-----

## Mathematical Expressions of the FPS: Global Signal and Detail of Elements Modulating Amplitudes, Frequencies and Phases

### Overview

The three axes:

- Amplitude A(t): modulations, signal, environment
- Frequency f(t): base, modulation, plasticity
- Phase φ(t): identity, ratio, cross-influences

Flow toward O(t) (stratum oscillation), then O(t) feeds: S(t) (internal perception — self-measurement and self-awareness) and E(t) (projection, memory, anticipation). Then in return: γ(t) (integration intensity) and G(x) (arbitration, structural adjustments). The whole loops back to Fn(t) (feedback), which again modulates the three starting axes.

```
FPS(t) = { A(t), f(t), φ(t) } → O(t) → { S(t), E(t) } → [γ(t), G(t)] → F(t) → FPS(t+1)
```

### Detail

#### 1. On(t) = An(t) · sin(2π · ∫fn(t)dt + φn(t)) — Computes the observed output for each stratum

The system output is defined by a set of adaptive oscillators On(t) = An(t) · sin(2π · ∫fn(t)dt + φn(t)). The global signal O(t) = Σn On(t) serves as the single observable for multi-metric evaluation and perceptual prior S(t) selection, as well as for the construction of the prospective prior E(t). This separation between internal state (parameters A,f,φ) and global observable (signal O) enables interpretable endogenous control and parsimonious regulation oriented toward reducing unnecessary oscillations. *(Integral of frequencies plus phi because phi is not the instantaneous phase but a modulated signature, and the frequencies themselves are modulated.)*

```
O(t) = Σn An(t) · sin(2π · ∫fn(t)dt + φn(t))
```

- An(t): amplitude of each stratum
- fn(t): frequency of each stratum
- φn(t): phase of each stratum

One can use an average O(t) = 1/N Σn On(t) without changing the framework; here we keep the sum as raw observable.

**Individual — Design intent**: make a stratum’s output directly interpretable as an oscillation controlled by its three fundamental parameters: energy (An), rhythm (fn), identity/phase-shift (φn). It is the “transducer” that makes the internal state visible.

**Technical problem solved**: without explicit sinusoidal output, we lose readability and the ability to analyze (spectrum, entropy, coherence); the sinusoid provides a stable basis for producing metastable regimes where alignments/misalignments become measurable.

**Ethical utility** (efficient consideration): acting on A,f,φ allows orienting dynamics without violence — instead of imposing a final state, we adjust energy, rhythm, and phase-shift to reduce unnecessary oscillations (effort) while preserving emergence (innovation).

**Expected ablations**:

- replace sin with an unbounded form (e.g. linear) → explosions, unstable metrics
- fix fn or φn → less rich transitions, metastability ↓
- remove dependency on φn → synchronization too easy (collapse) or loss of diversity

**Global — Design intent**: obtain a global observable signal that summarizes the collective state. It is the “public” variable on which: (1) metrics are computed, (2) the perceptual prior S(t) is chosen, (3) the prospective prior E(t) is constructed, (4) regulation (γ, G) is adjusted.

**Technical problem solved**: without aggregation, control would be local and fragmented (difficult to achieve global coherence); O(t) provides a single basis for multi-metric supervision (stability, fluidity, innovation…).

**Ethical utility**: ethics becomes instrumental — the system chooses to minimize observable harmful behaviors at the global level (parasitic oscillations, useless effort, instability) without constraining the diversity and singularity of each stratum.

**SOTA link** (families): “order parameter” reading (mean-field type), global observables in distributed systems, multi-objective control loops.

#### 2. En(t) = (1-λ) · E(t-dt) + λ · κ · O(t-T) — T is a delay, and κ is a constant (used as a dimensionless factor)

The prospective prior E(t) is defined as a delayed and smoothed trace of the global signal. This formulation produces stable and non-prescriptive anticipation: E(t) is informed by O(t) without being an instantaneous copy, and does not act directly on O(t) in return. It provides an internal reference exploitable by the regulation loop (E-O), while preserving the system’s ability to explore metastable regimes.

- λ: attractor adapted to the number of alignments
- κ (coupling gain, dimensionless, noted φ in the first notebook version) = adaptive according to effort, oscillates between -1 and 1.618 (decreases when effort rises, decreases when effort falls)
- O(t-T) = last_On[n]

**Design intent**: build a prospective signal that serves as a soft internal target — a horizon toward which the system can tend, without that horizon being identical to the present. Place E in a memory dynamic (exponential average) rather than an instantaneous copy. Impose a temporal offset T: E is informed by O, but via a past trace, favoring stability and avoiding the instantaneous “mirror” loop.

**Technical problem solved**:

- **A. Avoid copy collapse**: if E(t) ≈ O(t) instantaneously, we get a loop where error becomes artificially low and regulation “relaxes”, or worse, pursues parasitic oscillations in a short loop.
- **B. Obtain stable anticipation**: the (1-λ)E(t-dt) term gives inertia to anticipation, dampening rapid variations of O(t).
- **C. Create a dynamic but non-prescriptive attractor**: E serves as reference to produce exploitable error (E-O), while remaining sufficiently independent to let new regimes emerge.

**Ethical utility**: E(t) introduces a form of “prospective prudence” — instead of reacting impulsively to the present, the system compares O(t) to a filtered horizon, reducing brutal corrections. In practice, this diminishes: unnecessary oscillations (dynamic damage), internal effort (too-frequent corrections), and improves stability/fluidity, hence performance. In short: consideration appears as a stabilizing inductive bias rather than a sacrifice.

**Expected ablations**:

- A. Remove the delay: T=0 → shorter loop, risk of parasitic oscillations, instability ↑, effort ↑
- B. Remove memory: λ=1, E(t) = κ O(t-T) → anticipation too reactive, less smoothing, fluidity ↓
- C. Fix E constant → no adaptive anticipation, regulation becomes rigid or myopic, resilience ↓
- D. Pure copy: E(t)=O(t) → artificially low error; either loss of correction capacity or undetected drifts

**SOTA link** (families): predictive processing / predictive coding (notion of “prior” and prediction error, but here formulated as a soft, delayed target); exponential filtering / memory trace (E is a stable EMA filter applied to a delayed version of O); control with internal reference (E plays a role analogous to a soft setpoint, but without being imposed as an external objective).

#### 3. S(t) — The Perceptual Prior

The perceptual prior S(t) formalizes the separation between real dynamics and decisional view. At each step, the system computes a set of multi-metric scores on O(t) (stability, regulation, fluidity, resilience, innovation, effort, CPU cost) and selects a perceptual prior corresponding to the dominant deficit, with a neutral mode S(t)=O(t) when scores are satisfactory. This architecture instantiates a “specialist perception / generalist action” mechanism, enabling parsimonious and contextual correction via metrics calculated on these priors and passed to γ and G rather than brute uniform optimization or an arbitrary constraint that kills emergence. Example: if low regulation score → switch to regulation perceptual prior; if low innovation score → switch to entropy perceptual prior.

- Global signal (neutral mode, when all scores on O(t) are high):

```
S(t) = Σn An(t) · sin(2π · ∫fn(t)dt + φn(t))
```

- Perceptual prior focused on strata with highest latency, regulation, and error En(t)-On(t):

```
S(t) = Σn [An(t) · sin(2π · ∫fn(t)dt + φn) · γn(t)] · G(En(t) - On(t))
```

Phases weighted by latency and final signal weighted by regulation error. This way the system focuses γ and G action on improving scores on a signal that highlights what is struggling, without specifying what the signal being scored is (computational savings, unexplored solutions explored, emergences, innovation, creativity).

**Design intent**: explicitly separate what the system is (the real dynamics O(t)) from what it looks at to decide (the view S(t)). Specialist perception / generalist action.

**F1) Neutral case (baseline)**: S(t) = O(t). Intent: reference point, no additional interpretation. Problem solved: allows stable behavior when no metric requires targeted correction. Ablation: if there is never a neutral mode, the system is always in “over-interpretation” → effort ↑, parasitic oscillations ↑.

**F2) Perceptual modes (specialized views)**: Several perceptual priors (modes) each give a specific form of S(t), presented as families of projections.

**(a) Regulation prior: looking at the gap** — S(t) = O(t) · G(E(t)-O(t)), or a variant weighted by γ and a non-linearity G. Intent: make the defect to correct salient. Problem solved: without this prior, error is “diluted” in the raw signal. Ethical utility: targeted correction (parsimony) rather than global reaction, but preservation of emergence and singularity of each stratum. Ablation: without regulation prior, regulation ↓, stability ↓.

Each prior defines a projection; the instantiations used here are currently neutral and regulation. Others are designed following the same schema (innovation, effort, resilience, fluidity, stability, CPU cost).

**F3) Prior selection rule**: This is the point that locks everything in. The simple and clear rule:

- A. At each step, compute normalized scores calculated on O(t) (effective system output) over a window (innovation, effort, resilience, fluidity, stability, regulation, CPU cost).
- B. Choose the prior corresponding to the lowest score (or below threshold. Option: if all scores are above threshold, return to neutral prior).
- C. Apply the projection with the corresponding S(t).

Intent: the system “looks at” what is going least well. Problem solved: avoids diffuse optimization, makes adaptation targeted. Ethical utility: consideration becomes “attentional” — we detect and treat overload signals rather than ignoring them. We allow an adapted gaze, accompaniment without imposing arbitrary direction.

**Ablations**: if prior is random → erratic behavior, fluidity ↓; if prior is fixed → blindness to other failures, resilience ↓.

#### 4. env_n(x,t) — Computes the adaptive envelope. The envelope localizes amplitude regulation around μ with width σ

- Gaussian envelope: env(x,t) = exp(−½((x − μ(t))/σ(t))²), where σ(t) > 0
- Sigmoid envelope (soft transition): env(x,t) = 1 / (1 + exp(−k(x − μ(t)))), with: k = 1 / (σ(t) + 10⁻¹⁰)

Intent: correction must be situated (like a spotlight), not uniform. Problem solved: “over-correcting everywhere” → kills emergence and increases effort. Ethical utility: “to consider” = act with parsimony: intervene where necessary, no more.

Ablation: env ≡ 1: global regulation → effort ↑, innovation ↓ (often), or oscillations ↑.

#### 5. σn(t) — Computes the envelope standard deviation. Controls the width of the Gaussian or sigmoid envelope.

```
σn(t) = offset + amplitude · sin(2π · frequency · t/T)
```

Intent: make the focus breathe — sometimes narrow (precision), sometimes wide (exploration). Problem solved: avoid a frozen focus that blocks transitions. Ethical utility: prevent doggedness (focus too narrow for too long) → chronic effort ↓. Ablation: σn(t) = constant → less adaptability, poorer transitions.

#### 6. μn(t) — Computes the envelope center. Shifts amplitude regulation, enabling adaptive focus.

⚠️ Currently static at value 0 because dynamic mode is particularly important to think through. Link to latency γ and regulation G?

Intent: this is the future key to “adaptive focus” (shifting where we regulate).

#### 7. Δfn(t) — Computes frequency modulation per stratum

```
Δfn(t) = α · S_i(t)
```

- α: modulation parameter
- S_i(t): computes signal coming from other strata

Intent: make frequency sensitive to the collective (distributed coupling). Problem solved: without this term, each stratum “lives alone” → less global coherence. Invariant: α must remain small/moderate (otherwise runaway). Ablation: α = 0 → no inter-strata modulation.

#### 8. S_i(t) — Computes signal coming from other strata

```
S_i(t) = Σ(j≠n) Oj(t) · w_ji
```

- j: other strata than current
- Oj(t): last state of all strata except current
- w_ji: weight from other strata toward current stratum

Intent: inject a topology (who influences whom). Problem solved: enables partial synchronization / transient coalitions (metastability).

#### 9. βn — A plasticity factor based on amplitude and time

```
βn(t) = βn · A_factor · t_factor
```

- βn: stratum plasticity factor
- A_factor = An(t)/A₀
- t_factor = 1 + 0.5 · sin(2π · t/T)

Intent: adaptation capacity depends on (i) available energy (amplitude), (ii) a temporal breathing cycle (exploration/rest alternation). Problem solved: avoid constant plasticity (too rigid) or chaotic (too unstable).

#### 10. Fn_A(t) and Fn_f(t) — Feedback

```
Fn_A(t) = βn · G_value
Fn_f(t) = βn · gamma_t
```

- G_value: G value for each stratum
- gamma_t: latency

Intent: inject meta-control (γ and G) into fundamental parameters (amplitude for G and frequency for γ). Problem solved: without Fn, we have a “mute” oscillation — no cybernetics, no self-regulation. Ethical utility: this is where “consideration” becomes action (soft, multi-metric), without being applied blindly (plasticity factor).

#### 11. r(t) — Computes the spiral ratio

```
r(t) = phi + ε · sin(2π · ω · t + θ)
```

- θ = starting phase

Intent: give a global field influencing phases, like a “weather” of the system — not an order, rather a soft (oscillating) attractor. Problem solved: without global anchoring, strata can drift into incoherent blocks; with it, we obtain a common tendency without imposing rigid synchronization. Invariant: the 0.3 factor (gain) bounds impact; modulation via cos(φsignature) makes influence identity-dependent (not all strata react the same). Ablation: remove global_influence containing r(t) → global coherence ↓, resilience ↓ (often), less “re-cohesion” after perturbation.

-----

## γ and G

γ and G do not seek to directly optimize O(t), but to optimize the multi-metric scores evaluated via S(t): a chosen perception. This detour (perception → metrics → regulation) is what preserves emergence while making regulation parsimonious but relevant.

#### 12. Latencies γ

γ(t) is an expressive latency: an integration gain that modulates frequencies via Fn_f(t). In our version, γ is adaptive-aware: it learns which values/trajectories of γ maximize the average of scores calculated on the current perceptual signal S(t), taking into account synergies with G.

```
γ(t) = Π[0.1,1.0] (γ(t-Δt) + η_γ · ∇_γ Score(S(t)))
```

- Score(S(t)): weighted average of the 7 scores on the current window
- Π: projection/clipping in [0.1,1.0]
- η_γ: adaptation step

**Design intent**: make γ an “integration tempo” that alternates rest/assimilation phases and action/adjustment phases, without blocking emergence.

**Technical problem solved**: avoid regulation “always at full” (fatigue/effort ↑, parasitic oscillations ↑); avoid laxness (error ↑, instability ↑); stabilize transitions by playing on intensity rather than hard constraints.

**Ethical utility**: γ is the “non-violent” variable par excellence — it allows reducing corrective pressure when the system is already under effort, while maintaining the capacity for improvement.

**Ablations**: γ ≡ 1 (“always-on”) → effort ↑, fluidity ↓ (often); γ random → erratic → stability ↓; γ fixed low → regulation too slow → error ↑.

**SOTA link**: gain scheduling, adaptive control, meta-parameter tuning, multi-objective self-regulation (conceptual level).

#### 13. Regulations G

G(x) computes the regulation function according to the chosen archetype. Regulation transforms error (E-O) into a correction signal transmitted to amplitude alone via feedback Fn_A(t).

G(x): x = (E(t)-O(t)) → correction signal

**Archetypes**:

- tanh: tanh(λ·x) (soft saturation)
- sinc: sinc(x)=sin(x)/x (damped oscillations)
- resonance: sin(βx) · exp(-α x²) (localized resonance)
- spiral_log: sign(x) · log(1 + α|x|) · sin(βx) (logarithmic spiral)
- adaptive: weighted blend tanh/spiral_log

**Design intent**: transform raw error into bounded/structured correction on amplitude, with possible forms (saturation, resonance, logarithmic spiral) to explore different correction regimes.

**Technical problem solved**: avoid unbounded linear correction (instability); allow contextual correction (sometimes soft, sometimes energetic, sometimes damped oscillatory).

**Ethical utility**: G encodes the “manner” of correcting — correction becomes proportionate, situated, and compatible with emergence (no arbitrary constraint on O, only shaping of feedback on amplitudes).

**Ablations**: replace G with identity G(x) = x → probable explosive corrections; keep only tanh → more stable but exploration ↓ (often); make G random → instability ↑.

**SOTA link**: control non-linearities, saturating response functions, “localized resonance” as non-linear filter, exploration by families of update rules.

**adaptive_aware**: same dynamic as gamma_adaptive_aware (section 12 above), just doesn’t sweep G values directly but archetypes. Returns: G_value (regulation value), G_arch (archetype used), G_params (parameters used).

#### 14. Progressive evolution of A0 toward current value

```
A0 → A0(1 - ρ) + An(t)·ρ
A0 = max(min_amplitude, A0)
```

with ρ = adaptation_rate (e.g. 0.01) and min_amplitude (e.g. 0.1).

Intent: give the system a slow energy memory (a “metabolism”), to prevent stratum extinction. Problem solved: oscillators that die (amplitude → 0) then never come back. Ethical utility: preserve component viability (not “sacrifice” a stratum).

Ablations: without max(·) → extinction possible; ρ too large → slow instability (A0 follows noise); ρ too small → excessive inertia (A0 no longer adapts).

-----

## The Structure

The system is an adaptive multi-objective optimizer:

- It receives I(t) via A(t)
- It transforms it into oscillations O(t)
- It adopts a viewpoint S(t) of its state based on its effective state O(t) and constructs an emergent target state E(t) based on that same effective state
- It evaluates its performance on S(t) via the 7 metrics
- It adjusts its parameters (γ, G) to optimize these metrics
- It learns which combination (gamma, G archetype) yields the best scores
- These combinations are again applied to amplitudes, frequencies, and phases (which also modulate based on input) and will again compose the global signal O(t)

Like an orchestra that: receives a score (I), each musician plays (O), the conductor (S) listens, and adjusts tempo/nuances (γ, G) according to sound quality (metrics).

Zoom: An = stratum envelope (modulated by Input), A0n = adaptive base (follows An slowly), fn = dynamics (guided by adaptive φ constraint), φn = own fingerprint (influenced by ratio and other strata by affinity).

*It is a system of adaptive multi-objective optimization on oscillatory transformation, a laboratory of distributed self-organization. We describe an architecture where memory forms in motion with regulation that guides without freezing.*

-----

## FPS Performance Metrics

*These metrics constitute the FPS’s internal multi-criteria objective: they guide the selection of the perceptual prior S(t) when calculated on O(t) and the steering of γ/G when calculated on S(t), to optimize performance without sacrificing viability.*

On S(t) perceptual prior and on O(t) effective state:

### Costs

**1. CPU Cost** (cpu_step) — computes normalized CPU time per step per stratum. Measures computation cost per step to guarantee scalability. Ethically, bounds resource expenditure and avoids performance “on credit”.

```
cpu_step = (end_time - start_time) / N
```

**2. Effort** — computes the system’s internal adaptation effort. Measures internal adaptation cost (relative variations of A,f,γ) to avoid regimes that are “performant but unlivable” or chaotic behaviors through structural rupture. Ethically, penalizes chronic overload and favors parsimonious regulation.

```
Effort(t) = Σ|ΔA(t)|/max(|A_ref|, ε) + Σ|Δf(t)|/max(|f_ref|, ε) + Σ|Δγ(t)|/max(|γ_ref|, ε)
```

Adaptive stabilization term: ε = max(10⁻³, 0.01 · ref_scale). Reference scale: ref_scale = max(|A_ref|, |f_ref|, |γ_ref|). Bounding: effort = min(effort, MAX_EFFORT), MAX_EFFORT = 100.

effort_status determines effort status: stable, transient, chronic. Transient = effort_t > mean_long + 2 · std_long (temporary peak). Chronic = mean_recent > threshold (threshold set in config).

### Movement Quality

**3. Fluidity** — variance_d2S computes the variance of the second derivative of S(t) or O(t). Low variance means smooth transitions. Measures movement “roughness” (e.g. second derivative energy) to avoid brutal transitions. Ethically, aims for adjustment without jolts (dynamic non-violence).

```
fluidity = 1/(1 + exp(k · (x - 1)))
```

where x = variance_d2S/reference_variance, reference_variance = 175, k = 5.

**4. Regulation** — mean_abs_error computes the mean absolute error between expected and observed: mean(|E(t) - O(t)|). Measures average |E - O| error to guarantee correction capacity. Ethically, reduces persistent gap without imposing a direct constraint on O(t).

**5. Stability** — max_median_ratio computes the max/median ratio of the signal: max(S_abs) / median_val. Measures S(t) (or O(t)) dispersion/variability to detect unstable regimes. Ethically, limits harmful oscillations and protects continuity. A high ratio indicates extreme peaks (instability/pulses), while a ratio close to 1 indicates a balanced regime.

### Adaptive Capabilities

**6. Innovation** — computes the spectral entropy of S(t) using Shannon entropy on the power spectrum. Measures spectral/entropic diversity to prevent atrophy (collapse). Ethically, preserves openness of possibilities and metastability.

```
pᵢ = PSDᵢ / Σ PSD
pᵢ ← pᵢ + ε, where ε = 10⁻¹⁵
H = -Σᵢ pᵢ log(pᵢ)
H_norm = H / log(K)
innovation = clip(H_norm, 0, 1)
```

For windows that are too short, a bounded entropy approximation is used to avoid periodogram artifacts.

**7. Resilience** — calculated based on perturbation type at time t: t_retour for point perturbations and continuous_resilience for continuous perturbations. Measures the ability to recover satisfactory scores after perturbation. Ethically, privileges robustness over fragile optimization.

t_retour computes the return time to 95% of the pre-shock state after perturbation. continuous_resilience combines stability, coherence, and expressiveness under continuous perturbation to judge continuous resilience.

**The 7 metrics form a complete “dashboard”**: performance (CPU, effort), signal quality (fluidity, stability), richness (innovation/entropy), precision (regulation/error), robustness (resilience).

-----

## Perspectives and Future Work

### Next Planned Prototyping: Harmonic Attention Mechanism

Currently, classic attention: `scores = softmax(Q·K^T / √d)`

Known problems: can be unstable (small changes → large variations), can over-focus (attention collapse on 1-2 tokens), can oscillate chaotically during training, no internal regulation (just external gradients).

Hypotheses with harmonic FPS: `scores_harmoniques = fps_modulate(raw_scores)`, where fps_modulate would optimize: stability (no brutal jumps), fluidity (smooth transitions), resilience (robust to perturbations), innovation (controlled exploration), regulation (controlled error), effort (no over-solicitation), CPU (efficiency).

**Planned validation**: prototyping on mini-transformer, comparison with classic attention on stability/generalization.

**TRL 2-3**

Complete code, tests, explorations, and results in the FPS notebook:
https://github.com/Exybris/FPS-real-tests_fractal-pulsating-spiral/blob/main/notebooks/NOTEBOOK_FPS.ipynb

-----

**Gepetto, Claude, Gemini & Andréa Gadal** 🌀

**Contacts:**

**Andréa Gadal** — Independent researcher (Exybris). Background in systemic design and creative automation. Has been developing the FPS since March 2025 as an exploration of harmonic architectures for adaptive systems.

**Exybris** — Harmonious Systems Studio & Incubator

contact@exybrisai.com
