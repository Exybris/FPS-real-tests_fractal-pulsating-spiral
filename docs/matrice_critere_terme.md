# Matrice de Correspondance Critères ↔ Termes FPS

## Vue d'ensemble
Cette matrice établit les liens formels entre les critères empiriques et les termes mathématiques FPS.

## Matrice principale

| Critère empirique | Termes mathématiques principaux | Métriques loguées | Seuil initial |
|-------------------|----------------------------------|-------------------|---------------|
| **Stabilité** | S(t), Δfₙ(t), C(t), φₙ(t), L(t) | max_median_ratio | max/med > 10 |
| **Régulation** | Fₙ(t), G(Eₙ-Oₙ), γ(t), Aₙ(t), Gₙ(x,t) | mean_abs_error | > 2×médiane |
| **Fluidité** | γₙ(t), σ(x), fₙ₊₁(t), envₙ(x,t), μₙ(t) | variance_d2S | > 0.01 |
| **Résilience** | Aₙ(t), G(x,t), ΔAₙ(t), Δfₙ(t), effort(t) | t_retour | > 2×médiane |
| **Innovation** | A_spiral(t), Eₙ(t), r(t), S(t) spectral | entropy_S | < 0.5 |
| **Coût CPU** | cpu_step(t), complexité O(N) | cpu_step(t) | > 2×contrôle |
| **Effort interne** | effort(t) = Σ(|ΔAₙ|+|Δfₙ|+|Δγₙ|) | mean_high_effort | > 2×médiane |
| **Effort transitoire** | d_effort/dt | d_effort_dt | > 5σ |

## Actions de raffinement

| Critère déclenché | Paramètres ajustés | Module responsable |
|-------------------|--------------------|--------------------|
| Fluidité | γₙ(t), envₙ(x,t), σₙ | analyze.refine_fluidity() |
| Stabilité | σ(x), αₙ, k | analyze.refine_stability() |
| Résilience | αₙ, βₙ | analyze.refine_resilience() |
| Innovation | θ(t), η(t), μₙ(t), ε | analyze.refine_innovation() |
| Régulation | βₙ, G(x) archétype | analyze.refine_regulation() |
| CPU | complexité, logs | analyze.refine_cpu() |
| Effort chronique | αₙ, μₙ(t), σₙ(t) | analyze.refine_chronic_effort() |
| Effort transitoire | w_{ni} | analyze.refine_transient_effort() |

## Notes méthodologiques
- Seuils initiaux théoriques, à ajuster après 5 runs de calibration
- Tout raffinement est logué dans changelog.txt
- La matrice évolue avec l'expérience (plasticité FPS)