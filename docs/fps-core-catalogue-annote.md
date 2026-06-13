# FPS Core — Catalogue des éléments *(annoté)*

> Le moteur nu, isolé de tout l'échafaudage (CSV, fichiers individuels, debug, Kuramoto/neutral, batch, exploration/anomalies, validation, setup_logging).
> Objectif : tout ce qu'il faut pour reconstruire la FPS et la porter en JS, vérifiable contre l'oracle Python.

## Conventions transversales (à lire d'abord)

- **φₙ et θ sont deux variables distinctes, jamais fusionnées.**
  - `φₙ` = phase **signature** (qui est chaque voix). Invariante : dans `update_state`, sa réécriture est commentée → elle ne change pas d'un pas à l'autre.
  - `θ` = phase **intégrée** = `2π·∫fₙ·dt + φₙ`. **Important** : `Oₙ` et `S(t)` inlinent cette expression *non wrappée* (fidélité pipeline), on ne partage pas de variable `θ`. Le `θ` **nommé et wrappé** est réservé à `kuramoto_local2` (cohérence locale). **Ne jamais assigner θ → φₙ.**

> Réécriture φ commentée l.1785-1786.

- **Trois `φ` distincts** (source classique de confusion, certains ~1.618 par défaut) :
  - `φₙ` = phase signature (par strate, `state['phi']`) → la phase, dans θ/Oₙ. (`compute_phi_n`)
  - `φ_doré` = `spiral.phi` = 1.618 → module les **fréquences** via `r(t)`. (constante de config)
  - `φ_reg` = `regulation.phi` (adaptatif 0.8–1.618) → module le **prospectif `E(t)`**. (`compute_phi_adaptive`)

> Les trois φ sont bien distincts et doivent être correctement attribués (`compute_phi_n`, `spiral.phi` dans `compute_r`, `compute_phi_adaptive`).

- **Modes gardés : `adaptive_aware` uniquement** pour γ et G. Toute sélection statique supprimée. Les 4 archétypes de `G` restent, ils sont le *vocabulaire* que `G_adaptive_aware` choisit en interne.

> PRÉCISION — `sinc` est dans le vocabulaire mais jamais sélectionné.** `compute_G` définit bien les 5 archétypes, mais `compute_G_adaptive_aware` ne choisit jamais `sinc` : ses branches ne produisent que `tanh` / `adaptive` / `resonance` / `spiral_log`. Sous `G_arch: adaptive_aware` (la config), `sinc` est **inactif**.

- **Métriques** (7) : innovation, résilience adaptative, stabilité, fluidité, effort, coût CPU, **régulation** — non décoratives, elles **pilotent γ** via le score global (et G **indirectement, via le régime de γ**).

> PRÉCISION Les 7 scores sont produits par `compute_scores` (stability, regulation, fluidity, resilience, innovation, cpu_cost, effort) et `calculate_all_scores` → `system_performance` → entrée de `γ_adaptive_aware`. G ne lit pas ce score (couplage indirect via le régime de γ).

---

## 1. Le substrat

### 1.1 Strate `n` (richesse par élément)
Chaque strate porte son tempérament propre (`generate_strates`, seed fixe) :

| Champ | Rôle | Valeur initiale |
|---|---|---|
| `A0` | amplitude de base | U(0.3, 0.7) — **adapte** lentement (voir §8) |
| `f0` | fréquence de base | `0.4 + (2/N)·n + U(−0.2,0.2)`, borné [0.4, 2.4] — **figée** |
| `φ` (signature) | empreinte de phase | `0.0` — **invariante** |
| `α` | souplesse de modulation de fréquence | U(0.45, 0.7) |
| `β` | plasticité (gain des deux feedbacks) | U(0.22, 0.38) |
| `k`, `x0` | pente et centre de la réponse sigmoïde à l'input | U(1.8,2.5), U(0.4,0.6) |
| `w` | ligne de couplage (vers les autres strates) | écrasée par la matrice spirale |

> `generate_strates` l.30-67.
> **PRÉCISION reproduction.** `generate_strates` **arrondit** : `f0` et chaque poids `w` à **1 décimale**, `α`/`β`/`x0` à **2**, `k` à **1**, `A0` à **6**. Indispensable pour une parité bit-à-bit avec l'oracle. Le `seed` effectif vient de `config['system']['seed']` (=12345), pas du défaut 42 de la fonction.

### 1.2 Matrice de couplage `W` (la considération structurelle)
`generate_spiral_weights` : bande tridiagonale ±c antisymétrique ; **spirale ouverte ⇒ non antisymétrique aux bords** : enroulement −c en (0, N−1), ligne N−1 nulle ; somme par ligne = 0 conservée. (Le mode `closed=True` rétablit l'antisymétrie pleine.)

> Cette matrice ouverte n'est donc pas antisymétrique, et deux bords sont singuliers.** (N=6, c=0.1, `closed=False`, `mirror=False`) :
> ```
> [[ 0.   0.1  0.   0.   0.  -0.1]      Σ ligne :  [0 0 0 0 0 0]   ✓
>  [-0.1  0.   0.1  0.   0.   0. ]      Σ colonne: [-0.1 0 0 0 0.1 0]  ✗
>  [ 0.  -0.1  0.   0.1  0.   0. ]
>  [ 0.   0.  -0.1  0.   0.1  0. ]      Antisymétrique ? False
>  [ 0.   0.   0.  -0.1  0.   0.1]
>  [ 0.   0.   0.   0.   0.   0. ]]     ← ligne N-1 entièrement nulle
> ```
> Cause : la boucle `for i in range(N-1): W[i,i+1]=+c; W[i,i-1]=-c`
> 1. **À i=0**, `W[i, i-1]` = `W[0, -1]` = **W[0][N−1] = −c** (indexation négative numpy → enroulement). En JS, `W[0][-1]` n'existe pas : **un portage littéral raterait ce couplage**.
> 2. **La ligne N−1 n'est jamais écrite** (`range(N-1)` s'arrête à N−2) → dernière strate à zéro (ne reçoit rien).
> Conséquences exactes : la **bande intérieure** est antisymétrique, mais les deux bouts cassent l'antisymétrie (W[0][N−1]=−c sans contrepartie ; W[N−1][·]=0). Les **sommes de lignes sont toutes nulles**, mais les **sommes de colonnes ne le sont pas**. Le code l'assume : `init_strates` saute la vérif somme-nulle aux extrémités (`skip_edges`, i==0 / i==N−1, commentaire *« couplage non conservatif aux bords »*).

---

## 2. Pipeline d'un pas de temps (ordre exact)

> Dans `run_fps_simulation` : In → Aₙ → fₙ (avec F du pas précédent) → φₙ → γ_global (`adaptive_aware`) → γₙ → **φ_reg** → Oₙ/Eₙ → erreur/G → feedbacks → `update_state` → S(t)/métriques. Note : φ_reg est calculé **avant** Eₙ et lui est passé en argument (l.335-346).

### 2.1 `In(t)` — input contextuel
Vecteur d'entrée par strate. Par défaut : `offset = 0.1` (statique), `gain = 1`, `scale = 1.2`, sans perturbation. **C'est le point de branchement** : la démo y injecte ses impulsions.

> **PRÉCISION.** La boucle appelle `perturbations.compute_In(t, input_cfg, state, history, dt)` (la version riche de `perturbations_3.py`, l.469), **pas** `dynamics.compute_In`. Pour le portage, c'est celle de `perturbations` qui fait foi (offset/gain statiques ou adaptatifs + perturbations).

### 2.2 `σ(x)` — réponse sigmoïde
`σ(x; k, x0) = 1 / (1 + exp(−k·(x − x0)))`

> `compute_sigma`, l.119.

### 2.3 `Aₙ(t)` — amplitude adaptative
```
Aₙ(t) = A0ₙ · σ(Inₙ ; kₙ, x0ₙ) · envₙ(Eₙ − Oₙ(t−dt)) · (1 + clamp(F_Aₙ, [−0.5, 0.5]))
        plancher 1e-6
```
Mode statique : `Aₙ = A0ₙ · σ(Inₙ)`.
- **envₙ** (enveloppe gaussienne) : `exp(−0.5·((x − μₙ)/σₙ)²)`
  - `σₙ(t) = max(offset + amp·sin(2π·freq·t/T), 0.01)` (dynamique) ; offset 0.1, amp 0.1, freq 0.3
  - `μₙ = 0` (le mode dynamique de μ est désactivé)
  - `x = Eₙ(t) − Oₙ(t−dt)` ← l'erreur, avec le **O du pas précédent** (résolution de circularité)

> `compute_An` l.204-206 ; `compute_env_n` l.249 ; `compute_sigma_n` l.182-183 ; `compute_mu_n` renvoie le statique même en dynamique l.220.

### 2.4 `fₙ(t)` — fréquence modulée + cascade dorée
```
S_iₙ   = Σ_{j≠n} W[n][j] · Oⱼ(t−dt)          # signal des autres strates
Δfₙ    = αₙ · S_iₙ
fₙ     = f0ₙ + Δfₙ · βₙ                        # (βₙ(t) = βₙ·(Aₙ/A0ₙ)·(1+0.5·sin(2πt/T)) si dynamic_beta)
```
**Contrainte spirale** (relaxation ρ = 0.5, pour n = 0..N−2) :
```
fₙ₊₁ ← (1−ρ)·fₙ₊₁ + ρ · r(t) · fₙ
```
puis feedback latence : `fₙ ← fₙ · (1 + clamp(F_fₙ, [−0.5,0.5]))`, plancher 1e-6.
- `r(t) = φ + ε·sin(2π·ω·t + θ₀)` ; φ=1.618, ε=0.1, ω=0.05, θ₀=0 → **c'est ça, la « spirale pulsante »** : le rapport doré entre voisines qui respire.

### 2.5 `φₙ(t)` — phase signature (mode `individual`)
```
ωₙ        = ω · (1 + 0.2·sin(2πn/N))
φₙ(t) = φ_sigₙ
         + ε·sin(2π·ωₙ·t + φ_sigₙ)                         # danse personnelle
         + 0.3·(r(t) − φ)·cos(φ_sigₙ)                       # influence du rapport global
         + Σ_{j≠n} 0.05·W[n][j]·cos(φ_sigₙ − φ_sigⱼ)·sin(2πωt)  # affinités inter-strates
```
`φ_sig = state[n]['phi']` (invariante). ⚠️ Recalculée chaque pas mais **non stockée** (φₙ(t) n'est « non stockée » qu'au sens « pas réécrite dans state/portée par l'état », elle est bien loggée dans l'historique). 

> `compute_phi_n` l.445-458). `signature_affinity = cos(φ_sigₙ − φ_sigⱼ)`.

### 2.6 `θ(t, n)` — phase intégrée
```
phase_acc += 2π·fₙ·dt
θ(t,n) = phase_acc + φₙ(t)
```
Porte la fréquence intégrée temporellement. Utilisée par `Oₙ`, `S(t)`, et la cohérence locale. **N'écrase jamais φₙ.**

### 2.7 `Oₙ(t)` — sortie observée (par strate)
```
Oₙ(t) = Aₙ(t) · sin(θ(t,n))
```

> `compute_On` l.1260-1262.

### 2.8 `φ_reg` — φ adaptatif (la considération sous tension)
Piloté par l'effort (seuils : low 0.5, high 5 ; bornes 0.9–1.618) :
- effort **chronique** (moyenne des 10 derniers > high) → `φ_reg = 0.8` (Eₙ < Oₙ : le système **se repose**)
- effort < low → `1.618` (croissance/exploration)
- effort > high → `1.0` (maintien)
- entre les deux → interpolation linéaire 1.618 → 1.0

> `compute_phi_adaptive` l.1299-1311 : chronique → `phi_min − 0.1` = 0.9−0.1 = 0.8 ; interp `phi_max·(1−t) + 1.0·t`. Détection chronique = `len > 10` et `mean(effort[-10:]) > high`.

### 2.9 `Eₙ(t)` — sortie attendue (attracteur inertiel)
```
Eₙ(t) = (1 − λ)·Eₙ(t−dt) + λ · φ_reg · Oₙ(t−dt)      # λ = lambda_E = 0.1
```
Initialisation : `Eₙ = A0ₙ` — dans `compute_En`, branche `else` quand `len(history)==0` (≈ l.1395-1400), + garde-fou si `last_En` invalide (≈ l.1385-1388).

> `compute_En` l.1314 ; init A0 l.1398-1399 ; garde-fou l.1384-1387 — **numéros de ligne exacts**. λ_E=0.1 vient de la config (le défaut `.get()` du code est 0.05 — ne pas s'y fier).

### 2.10 `errorₙ`
`errorₙ = Eₙ − Oₙ` (alimente G et les deux feedbacks).

---

## 3. Les deux contrôleurs adaptatifs (avec mémoire qui embarque)

### 3.1 Vocabulaire `G(x)` (les 4 archétypes)
```
tanh        : tanh(λ·x)
resonance   : sin(β·x)·exp(−α·x²)
spiral_log  : sign(x)·log(1 + α·|x|)·sin(β·x)
adaptive    : α·tanh(λ·x) + (1−α)·spiral_log(x)
```

### 3.2 `G_adaptive_aware` — régulation consciente de γ
Choisit l'archétype + ses params selon le **régime de γ**, l'erreur, et un facteur d'exploration `exp_f = 0.5·sin(0.1t) + 0.5` :
- `γ < 0.4` (repos) → tanh/adaptive doux
- `γ > 0.7` (actif) → resonance/spiral_log (et change de stratégie si γ oscille vite)
- zone médiane → rotation créative des 4 archétypes
Puis appelle `compute_G(error, archétype, params)`.
**Embarque `regulation_memory`** : efficacité par contexte `(G_arch, γ_bucket, error_bucket)`, préférences, historique des transitions. → la régulation *apprend ce qui marche* (vrai mais comme un veto correctif, pas comme le choix premier : la mémoire ne dirige pas la décision, elle la corrige quand le choix provisoire a historiquement échoué.). Les seuils de error_bucket sont low <0.1, medium <0.5, high au-delà (l.1080).

> `compute_G_adaptive_aware` l.1116 pour exp_f ; seuils l.1119/1129 ; rotation médiane sur `["spiral_log","adaptive","resonance","tanh"]` l.1152 ; `regulation_memory` l.1052-1059.

### 3.3 `γ_adaptive_aware` — latence consciente de G
- **Premiers pas** : exploration systématique `γ ∈ linspace(0.1,1,10)[step%10] + 0.05·randn` ⚠️ *(le `randn` est le point RNG à neutraliser/partager pour la validation)*
- ensuite : `system_perf = mean(calculate_all_scores)` ; suit les couples `(γ, G)` → score de synergie ; si tous scores ≥ 5 et meilleure synergie > 4.5 → **mode transcendant** (micro-oscillations autour du meilleur γ) ; sinon **exploration consciente** des couples non testés / créative / `create_quantum_gamma`.
- sortie : `clip(γ, 0.1, 1.0)`.
**Embarque `discovery_journal`** : régimes découverts, transitions, pics de γ, phases de repos, synergies (γ,G).

> `compute_gamma_adaptive_aware` : phase <50 l.758-774 avec `randn` l.768 ; synergie >4.5 l.809/958 ; mode transcendant micro-ondulation `+0.02·sin(0.5t)` l.972 ; quantum l.1030 ; `clip(0.1,1.0)` l.1032. 
- `system_perf` = moyenne des scores de la fenêtre **`current`** de `calculate_all_scores`, sur `history[-50:]`.
- Le `current` n'est pas une simple fenêtre brute : c'est la moyenne pondérée-par-maturité de plusieurs fenêtres. Donc `system_perf` dépend de où en est le run: au début il regarde surtout l'immédiat, à maturité il intègre le global.

### 3.4 `γn`
Anciennement utilisée par S(t) erreur En - On.


--- 

## 4. Les deux feedbacks (jamais un seul `Fₙ`)
```
F_Aₙ(t) = βₙ · G_value      → module l'AMPLITUDE du pas suivant   (via Aₙ·(1+F))
F_fₙ(t) = βₙ · γ(t)         → module la FRÉQUENCE du pas suivant  (via fₙ·(1+F))
```
Clampés à [−0.5, 0.5] au moment de l'application. Un canal pour *quoi corriger*, un pour *à quelle vitesse*.

> `run_fps_simulation` l.426-427. Le clamp [−0.5,0.5] est appliqué côté `compute_An` (l.203) et `compute_fn` (l.393). 

---

## 5. Observables globaux

### 5.1 `O(t)` — observable global (agrégation neutre)
```
O(t) = Σₙ Oₙ(t) = Σₙ Aₙ·sin( 2π·∫fₙ·dt + φₙ )
```
L'agrégation brute de toutes les voix. C'est sur `O(t)` que sont calculées les métriques de performance pour opérer le changement de filtre S(t).

> **CONCEPTUEL — Précision sur O(t).** Sous la config livrée (`signal_mode: "extended"`), `S(t) ≠ O(t)`, et les métriques de performance (stabilité via std(S), fluidité, innovation) sont calculées sur **`S(t)`** pour informer gamma et G : `compute_scores` lit `h['S(t)']` (l.1014), `fluidity`/`entropy_S` viennent de `S_history`. 
> **Rôle réel d'O(t)** : O(t) est le **repère neutre du switch** qui choisit *quel filtre S(t)* sert à calculer les métriques passées à γ et G ; les métriques pilotant γ/G, elles, sont bien sur S(t).

### 5.2 `S(t)` — prior(s) perceptif(s) (un filtre perceptif, pas un seul signal)
Une **famille** de signaux, pas un objet unique :
- **neutre** : identique à `O(t)`.
- **pondéré erreur En - On** (mode « extended »), cible à implémenter :
  ```
  S(t) = Σₙ  Aₙ · sin(2π·∫fₙ·dt + φₙ) · echelle_n
  ```
  → met en exergue les strates qui peinent côté régulation.

  Mécanique — trois couches aux rôles indépendants :

    1. Erreur RELATIVE     `err_n = |Eₙ − Oₙ| / Aₙ`
       Divisée par l'amplitude : juste pour les grandes comme les petites strates (une grosse voix a de grosses erreurs sans être "en peine").

    2. Poids BORNÉ `[0.1, 1]`   `poids_n = 0.1 + 0.9 · saturante(err_n / échelle)`
       `échelle = max(médiane(err), ε)` : seuil ADAPTATIF — "combien de fois plus loin que la norme du groupe à cet instant" (médiane robuste). La saturante (plafond 1) empêche une strate très éloignée de manger toute la lumière ; le plancher 0.1 garantit que personne n'est jamais réduit au silence (on préserve les chemins discrets, anti-rigidité).

    3. Recentrage sur 1    `echelle_n = poids_n / moyenne(poids)`
       La moyenne des poids devient exactement 1 : `S(t)` garde la même intensité globale que `O(t)`. Les métriques n'ont donc pas à zoomer / dézoomer : c'est une REDISTRIBUTION d'attention à énergie conservée, ni amplification ni atténuation.

- **Généralisation** : les autres filtres (innovation, résilience, fluidité…) suivent EXACTEMENT ce gabarit. On change seulement ce qui entre dans `err_n` (l'inverse du score de la métrique concernée, par strate). Le choix du filtre actif se fait par un switch piloté par la métrique la plus en peine, mesurée sur `O(t)` : repère neutre.

> Pour neutre (= mode `simple` = Σ Oₙ) et étendu (`compute_S` l.1571-1687 : `Σₙ  Aₙ · sin(2π·∫fₙ·dt + φₙ) · γₙ · G(En−On)` (cible : `Σₙ Aₙ·sin( 2π·∫fₙ·dt + φₙ )`).

> **Intention de design** : `S(t)` donne du *contexte* à γ et G **sans imposer de résolution** — ils ne reçoivent qu'un score global, ignorent quelle métrique pèche et que le signal est pondéré. (Triade : `O(t)` observable global · `S(t)` prior perceptif · `E(t)` prior prospectif.)

> ⚠️ Dans la triade, « E(t) prior prospectif » = l'agrégat des `Eₙ` (`compute_En`). Mais le code **logue par ailleurs un `E(t)` qui est l'ÉNERGIE** : `compute_E = sqrt(Σ Aₙ²)/√N` (l.1692), une grandeur distincte. Pour le portage, ne pas confondre `Eₙ` (prospectif) et `E(t)` loggé (énergie).

---

## 6. Cohésion de chaîne `C(t)` — ce n'est PAS une cohérence
```
C(t) = (1/(N-1)) · Σ cos(φₙ₊₁ − φₙ)        # accord moyen entre voix adjacentes (chaîne)
```
Mesure le **maintien de la forme structurelle** de la chaîne. Consommé par `continuous_resilience` (doit rester haut sous perturbation continue). Nommé **cohésion de chaîne** (anciennement « accord spiralé ») pour ne pas le confondre avec `Rloc` ci-dessous.

> **PRÉCISION.** def l.1425, l.1457 : `cos_sum / (N - 1)`, somme sur `range(N-1)`). La différence `φₙ₊₁−φₙ` est ramenée dans `[−π,π]` avant le `cos` (l.1453) — mathématiquement sans effet (cos 2π-périodique).

## 7. Cohérence locale (la vraie — `theta_from_history` + `kuramoto_local2`)
```
θ(t,n)      = wrap( 2π·cumsum_t(fₙ·dt) + φₙ )          # (T, N), wrappé, RÉSERVÉ à ce calcul
voisins(n)  = indices triés par |W[n]| décroissant, non nuls
poids       = |W[n][voisins]| normalisés (somme = 1)
Rloc(t,n)   = | Σ_j poids_j · e^{iθ(t,j)} |             # cohérence locale par nœud ∈ [0,1]
incoh       = 1 − Rloc
μ(t)        = moyenne spatiale de Rloc
σ(t)        = écart-type spatial de Rloc
(lissage temporel optionnel : convolution fenêtre 25)
```
- On a aussi le R global classique |mean e^{iθ}|.

→ **Pour une démo ludique, c'est ce qui prend la primauté visuelle** : chaque nœud coloré/dimensionné par son `Rloc`, les arêtes = les vraies arêtes pondérées de `W`, et `μ(t)±σ(t)` en lecture globale.

> Couche **visuelle/interactive** dédiée à donner à voir les dynamiques FPS — donc légitimement **hors moteur nu**. À sourcer depuis le module de visualisation avant tout portage de cette section.

---

## 8. Mise à jour de l'état (`update_state`) — ce qui persiste
- stocke `current_An, current_fn, current_phi, current_gamma, current_Fn_*`
- **A0 adapte** : `A0 ← 0.99·A0 + 0.01·current_An`, plancher 0.1
- **f0 n'adapte pas** (désactivé volontairement)
- **φ signature n'est pas réécrite** (protège la séparation φₙ/θ)

**État qui doit traverser le temps** (le `history[-1]` et les buffers) :
`O, E, An, fn, gamma, mean_abs_error, effort(t), G_arch_used` + `discovery_journal` (γ) + `regulation_memory` (G) + `effort_history`, `S_history`, `μ_history` (ex-`C_history`), `An_history`.

> `update_state`l.1780-1824 : A0 `0.99/0.01` plancher 0.1 ; f0 commentée ; φ commentée). L'état traversant est dans le `history.append` (l.822-856) et les buffers dédiés.

---

## 9. Métriques & score global (elles pilotent γ ; G indirectement via γ)

| Métrique | Définition (essence) |
|---|---|
| **innovation** | `entropy_S` : entropie spectrale (Shannon) de S sur une fenêtre |
| **fluidité** | `1/(1+exp(5·(x−1)))`, `x = variance_d2S / 175` (variance de la dérivée 2nde de S) |
| **effort** | voir calcul des deltas ci-dessous |
| **coût CPU** | `compute_cpu_step` : temps mur du cœur dynamique / N |
| **résilience adaptative** | choisit selon le contexte : `t_retour` (perturbation *ponctuelle* / choc) ou `continuous_resilience` (*continue* / sinus, bruit, rampe — consomme l'**accord spiralé** C(t)) |
| **stabilité** | ratio de stabilité (dans `compute_scores`) |
| **régulation** | 7ᵉ score, embarqué dans la moyenne qui pilote γ (gardé) |

> `compute_fluidity` l.313-316, k=5, réf 175 ; `compute_scores` l.992-1095.
- `variance_d2S` n'est pas un `np.var` naïf mais un **estimateur robuste par IQR** : `(IQR·0.7413)²` sur `np.gradient` (`compute_variance_d2S` l.286-290). 
- « stabilité » = score discret sur `std(S)` (5 si <0.5, … 1 si ≥3), pas un ratio continu.
> `continuous_resilience` (l.504) : à deux composantes pondérées. Elle combine 0.6·stabilité_C + 0.4·composite_S (l.571), où le composite de S mêle lui-même stabilité (via le coefficient de variation S_std/S_power, barème par paliers l.551-558) et expressivité (tanh(2·S_power) — « le signal garde-t-il son amplitude ? », l.561). Pour ne pas confondre « stable parce que mort » et « stable parce que résilient » — l'expressivité garde le système honnête.

**Effort — calcul par pas (deltas depuis `history[-1]`) :**
```
ΔAₙ = Aₙ(t) − Aₙ(t−dt) ;  Δfₙ = fₙ(t) − fₙ(t−dt) ;  Δγₙ = γₙ(t) − γₙ(t−dt)
effort = (Σ|ΔAₙ|)/An_denom + (Σ|Δfₙ|)/fn_denom + (Σ|Δγₙ|)/γ_denom,  saturé à 100
```
(Ā = mean|Aₙ|, f̄ = mean fₙ, γ̄ = γ(t) global.) **Double rôle** : métrique *et* entrée de `φ_reg` (§2.9) — l'effort boucle directement sur le prospectif.

> `compute_effort` l.97-110 ; γ̄ = γ global confirmé par l'appel l.535. 🟡 Le dénominateur exact est `max(|réf|, ε)` avec `ε = max(0.001, 0.01·max(|Ā|,|f̄|,|γ̄|))`.

### Précisions : Score global

> **(`calculate_all_scores`) :** chaque métrique normalisée 1–5, sur fenêtres adaptatives (immédiate/récente/moyenne/globale) pondérées selon la maturité du run. La **moyenne** = `system_performance` → entrée de `γ_adaptive_aware`. (G ne lit pas ce score : il lit le *régime de γ* → couplage indirect.)
> **La chaîne de score, exacte :** calculate_all_scores calcule compute_scores sur plusieurs fenêtres adaptatives (immediate/recent/medium/global, dimensionnées via compute_adaptive_window depuis la config adaptive_windows.scoring), puis les pondère selon la maturité du run (l.1204-1209) :

- maturité <0.2 (début) → immediate 0.7 / recent 0.3
- <0.5 (mi-parcours) → immediate 0.2 / recent 0.5 / medium 0.3
- ≥0.5 (mature) → immediate 0.1 / recent 0.2 / medium 0.4 / global 0.3

> maturity = len(history) / max(total_steps, 100) (l.1202), et la fenêtre global n'apparaît qu'à partir de ≥0.5 (l.1212). Le tout retourné sous {'current': final_scores} 

> `calculate_all_scores` l.1145 ; `weighted_average` l.1098.

**Les barèmes exacts** *metrics.py*

- stabilité : std(S) → 5/4/3/2/1 aux seuils <0.5 / <1 / <2 / <3 / sinon (l.1028)
- régulation : mean_abs_error → seuils <0.1 / <0.3 / <0.5 / <1 / sinon (l.1033)
- fluidité : mean(fluidity) → >0.9 / >0.7 / >0.5 / >0.3 / sinon (l.1038)
- innovation : mean(entropy_S) → >0.8 / >0.6 / >0.4 / >0.2 / sinon (l.1082)
- coût CPU : mean(cpu_step) → <0.001 / <0.01 / <0.1 / <1 / sinon (l.1087)
- effort : mean(effort) → <0.5 / <1 / <2 / <3 / sinon (l.1092)
- résilience : cascade de priorité (l.1042-1078)

## 10. Suite

### **OK :**

-	Voir pourquoi curent_ratio (l.386) est calculé puis jamais utilisé, et voir s'il faut le brancher ou le supprimer. ✅ (supprimé)

-	Supprimer effort_factor. ✅

-	L.433 : r(t) est ré-inliné au lieu d’appeler compute_r, à corriger. ✅

-	compute_En l.1350-1353 : un if/elif aux conditions identiques, la branche elif est morte. À corriger. ✅

-	Supprimer le mécanisme k_spacing désactivé l.1342/1372-1374. ✅

-	Pendant une transition trop rapprochée (<10 pas) le G_value est un mélange old/new mais il est loggé sous l’ancien archétype (l.1230), donc l'apprentissage attribue un G mélangé aux mauvais arch. À corriger. ✅

-	Les params viennent de deux sources différentes pour G (inline dans les régimes l.1124-1173, vs adapt_params_for_archétype dans le veto/blend l.1204-1224), un même archétype peut donc recevoir des params différents selon le chemin. À fouiller et corriger. ✅

-	Supprimer la fonction Fn. ✅

- `gamma_adaptive_aware` tronque à `history[-50:]` avant de passer la tranche à `calculate_all_scores`. Et comme `calculate_all_scores` calcule `total_steps = len(recent_history)` (l.1171) puis `maturity = len(recent_history) / max(total_steps, 100)`, la conséquence est: 
>`total_steps` plafonne à 50. Donc `maturity = 50 / max(50,100) = 50/100 = 0.5` dès que le run dépasse 50 pas — et reste bloqué à 0.5 pour toujours. Le système n'atteint jamais la maturité « pleine » (`>0.5` strict) : il vit en permanence sur le palier `maturity < 0.5`… ou pile à la bascule. À 50 pas pile c'est `0.5`, donc la branche `≥0.5` (mature, avec fenêtre `global`) s'active — mais `global = compute_scores(recent_history)` = sur les 50, identique à `medium` plafonné. Donc en pratique le run mature tourne sur `immediate/recent/medium/global` tous calculés dans une fenêtre ≤50. 
>La « maturité par âge du run » ne se déclenche jamais vraiment. L'intention du design (juger l'immédiat au début, intégrer le global à terme) est court-circuitée par le [-50:] : passé 50 pas, l'horizon est gelé. ✅

- Deux garde-fous neutres empilés : `compute_scores` renvoie 3.0 partout si <3 pas (l.1002), et `calculate_all_scores` renvoie `{'current': 3.0…}` si <5 pas (l.1157). Donc avant 5 pas, `γ` ne reçoit aucune information discriminante — incohérent avec la phase d'exploration des 50 premiers pas (les combinaisons efficaces faites pendanr le randn n'ont pas de score pour les repérer, rendant l'exploration randn inutile). Trouver une solution qui permet d'avoir des scores lors du balayage par randn. ✅

- Mettre à jour S(t) extended (sans gamma_n et G), en lien avec son docstring ✅

- gamma_adaptive_aware : une variable écrasée (`state_key`) dans le mode transcendant. 
Au début de la fonction, `state_key` = `(round(gamma_current,1), current_G_arch)` désigne l'état courant. Mais à la l.950, la boucle `for state_key, state_info in journal['coupled_states'].items()` réutilise le même nom comme variable d'itération — donc après la boucle, `state_key` ne vaut plus l'état courant mais la dernière clé du dictionnaire. Or à la l.970, le choix « micro-ondulations vs convergence » repose justement sur `if best_synergy and state_key == best_synergy`. L'intention (le commentaire le dit : « si on est dans la synergie parfaite ») était de tester si l'état courant est le meilleur couple. À cause de l'écrasement, ça teste si la dernière clé itérée l'est — c'est-à-dire l'ordre d'insertion du dict, pas l'état réel. L'impact est borné (les deux branches visent un γ proche de `best_synergy[0]`), mais la trajectoire exacte de γ en diffère → fix chirurgical : un nom distinct, genre current_state_key, utilisé à la l.970. ✅

- Le garde-fou < 3 pas renvoie tout à 3.0 (l.1002-1011) — les 7 scores neutres tant que la fenêtre est trop courte. À acter, parce que ça veut dire que les tout premiers pas ne « pilotent » γ avec rien de discriminant et ne permet pas d'enregistrer des synergies effectives pendant l'exploration randn en début de run. ✅

- La mémoire d'efficacité de G compte chaque observation ~29 fois. Dans compute_G_adaptive_aware, l'étape "analyser l'efficacité contextuelle" (l.1069-1086) re-balaie history[-30:] à chaque appel et ré-appende toutes les paires (pas i → pas i+1) dans effectiveness_by_context — sans marqueur de ce qui a déjà été traité. Donc chaque transition historique est ré-enregistrée à chaque pas tant qu'elle reste dans la fenêtre des 30 : la mémoire gonfle ~29× trop vite, et surtout la moyenne sur [-5:] qui décide du veto est dominée par les toutes dernières paires répétées. C'est la même famille que le bug d'attribution du blend, mais en pire : non seulement on apprend parfois avec de fausses étiquettes, mais on apprend en bégayant. ✅

- Le blend dure 100 pas, pas 10. Le commentaire dit "10 steps" mais le test est t - last_transition_time < 10 où t est en unités de temps — avec dt=0.1, ça fait 100 pas de mélange. Et pendant tout ce temps, current_G_arch n'est pas mis à jour, donc... ✅

- ...spam de transitions pendant le blend. Chaque pas où l'archétype choisi diffère de l'ancien appende un nouvel enregistrement dans G_transition_history — pendant un blend de 100 pas, ça peut faire ~100 fausses transitions enregistrées pour un seul changement réel. ✅

- compute_G_adaptive_aware appelée N fois par pas (une par strate) : ce qui n'est pas par strate, c'est sa mémoire, son current_G_arch, son compteur adaptation_cycles qui avance N fois par pas, ses transitions qui peuvent s'accomplir entre la strate 3 et la strate 4 d'un même instant. À fouiller. ✅

### **STILL :**

- gamma_adaptive_aware : le mode quantique (l.1030) est inatteignable. 
Dans l'exploration consciente, l'espace des G testés est `all_G_archs = {tanh, resonance, spiral_log, adaptive, adaptive_aware}` (l.999). Mais `adaptive_aware` n'est qu'un alias-secours (notre §3.1) — il n'apparaît jamais comme `G_arch_used` réel. Donc les 10 couples `(γ, 'adaptive_aware')` ne sont **jamais** dans `tested_combinations` → `untested` n'est jamais vide → la branche `else` qui appelle `create_quantum_gamma` (l.1030) n'est jamais atteinte. Le mode quantique est donc dormant. Ça tient entièrement à ce que `G_adaptive_aware` peut émettre comme `G_arch_used`, et il n'émet pas de `adaptive_aware`. Si on veut que le "mode quantique" vive, il faudra retirer adaptive_aware de cet ensemble (n'y mettre que les 4 vrais archétypes).

- Rloc_smooth ne lisse qu'un seul nœud. Il est calculé dans la boucle (Rloc_smooth = np.convolve(Rloc[:, n], …)), réassigné à chaque tour. Après la boucle, il ne contient donc que le lissage du dernier nœud (n = N−1). Et comme ce dernier nœud vaut 1.0 partout (voir couplage), le Rloc_smooth renvoyé est une constante ≈ 1.0, sans rapport avec la carte. Fix : Rloc_smooth = np.zeros((T,N)) avant la boucle, et Rloc_smooth[:, n] = np.convolve(...) dedans.

- Rloc_smooth n'est pas dans ces huit fichiers (il vit dans le module de visualisation, conforme au catalogue qui le met hors moteur nu — à uploader plus tard)

- k_neighbors = N : nom trompeur. Ce n'est pas un « K local » réglable — c'est « tous les voisins non nuls ». Inoffensif avec la W tridiagonale (≈2 voisins/nœud de toute façon), mais le nom suggère une localité paramétrable qui n'existe pas.

- Le buffer d'attribut de fonction (l.335-343). `compute_entropy_S._buffer` est un état caché persistant attaché à la fonction, qui accumule les 5 dernières valeurs scalaires entre les appels. Trois problèmes :
> Reproductibilité : cet état survit d'un appel à l'autre et d'un run à l'autre dans le même process — il n'est jamais remis à zéro. Deux simulations lancées à la suite partagent ce buffer. Pour la parité bit-à-bit avec l'oracle et la reproductibilité, c'est un point de fuite réel (le 1ᵉʳ run et le 2ᵉ ne partent pas du même état).
> Cohérence : ce buffer (max 5 valeurs) double S_history qui existe déjà ailleurs — la fonction se reconstruit une mini-mémoire au lieu de recevoir la fenêtre proprement.

- `resilience` : On avait vu côté `compute_scores` qu'elle puise en cascade dans trois sources (`adaptive` → `continuous` → proxy `C(t)`). D'abord `adaptive_resilience` (l.576), parce que c'est elle qui choisit : un vrai switch selon le type de perturbation. La sélection est explicite, elle lit `config.system.input.perturbations` et si l'une est `sinus/bruit/rampe` → perturbation continue → `continuous_resilience` ; sinon (`choc` ou rien) → ponctuelle → `t_retour` (l.449). Les deux barèmes 1-5 sont nets (continue : seuils 0.90/0.75/0.60/0.40 ; ponctuelle : t_retour < 1/2/5/10, avec normalisation 1/(1+t_retour), l.683). Et il y a un repli de compatibilité ascendante vers l'ancienne clé config.system.perturbation (l.619-626) — attention au portage : deux schémas de config coexistent. Certaines choses à approfondir et corriger si besoin :
> 🟡 le défaut dt=0.05 dans la signature (l.579) ≠ le dt=0.1 de ta config. Si jamais un appelant ne passe pas dt, t_retour se calcule sur la mauvaise échelle de temps. À vérifier côté appel (probablement dt est bien passé, mais le défaut est trompeur).
> 🟡 sous la config livrée, perturbation_type = 'none' (la seule perturbation est type: none) → branche t_retour, mais avec t_choc=None → t_retour ne se calcule pas → la cascade de compute_scores retombe alors sur continuous puis sur le proxy C(t). Donc en run nominal sans perturbation, la résilience est en pratique portée par le proxy C(t), pas par les deux « vraies » métriques. Bon à savoir : ce qui pilote γ côté résilience, au repos, c'est la cohésion de chaîne.
> 🔴 Si le switch dépend de la config de l'input, ce sera problématique quand l'input ne sera plus programmé mais reçu et varié (dans des applications terrain par exemple).
> 🟡 Le perturbation_active=True par défaut + le repli sur 1.0 (l.525) : si l'historique fait moins de 20 points, ça renvoie 1.0 (résilience parfaite) — un optimisme de démarrage (les premiers pas paraissent parfaitement résilients).

- La résilience a une cascade de repli à trois étages (le §9 dit « choisit selon le contexte » — voici le mécanisme exact). Elle essaie dans l'ordre : adaptive_resilience si présente (seuils ≥0.90/0.75/0.60/0.40, l.1048-1057) → sinon continuous_resilience (mêmes seuils, l.1063-1072) → sinon C(t) comme proxy sur les 5 derniers (>0.9/0.7/0.5/0.3, l.1075-1076), et défaut 3 si rien. Donc le score de résilience peut venir de trois sources différentes selon ce qui est dispo dans la tranche — un point de vigilance à fouiller et corriger si besoin (deux runs peuvent scorer la résilience par des chemins différents).

- `compute_cpu_step` (l.37) : c'est (end−start)/N sur des time.perf_counter() : c'est du temps mur, par nature non reproductible et dépendant de la machine. Et ça pilote γ (via le score cpu_cost). Pour la parité bit-à-bit avec l'oracle, c'est un point dur réel à régler.

- Enlever l'affichage et le log debug des valeurs de An(t) et autres à chaque pas dans le terminal (ne garder que le log normal à chaque pas, parmi les autres valeurs)

- Sortir le poids borné et le recentrage de la fonction S(t) dans un _echelle_attention(err, eps) réutilisable pour le jour où les autres filtres arrivent.

- Calculer les même scores que ceux calculés sur S(t) pour les visualisations, mais sur O(t) et en faire aussi des visualisations dans visualize.py

### **QUESTIONS :**

-	Voir si le fait que sinc ne fait pas partie des archétypes sélectionnés par G_adaptive_aware est cohérent ou une erreur.

-	Voir si l'arrondi à 1 décimale pour f0, k et w, l'arrondi à 6 décimales pour A0, et celui à 2 décimales pour alpha/beta/x0 sont une différence cohérente ou une erreur.

-	Est-ce que les deux bords du couplage w qui cassent l'antisymétrie posent problème ?

-	Voir s'il est logique de faire le mu_n adaptatif pour l'enveloppe gaussienne (et voir la logique de l'enveloppe gausienne).

-	Voir le second appel à compute_En (l.159 dynamics.py) est cohérent ou s’il faut corriger (par exemple en remplaçant l’appel par autre-chose qui donne ce qui est attendu plus directement).

-	Voir en quoi phi_sig = state[n][‘phi’] est “non stockée au sens pas réécrite dans state/portée par l'état" et si ça pose problème.

-	Un allias adaptive_aware dans compute_G (l.100-105 regulation.py) mais qui ne renvoie qu’un tanh de secours : voir si c’est cohérent ou une erreur à supprimer.

-	preferred_G_by_gamma est en écriture seule. Elle est incrémentée à chaque pas (l.1246-1253 dynamics.py) mais jamais relue pour décider, dans toute la fonction. Une partie de la mémoire apprend et agit (effectiveness), une autre est seulement enregistrée. Voir si c’est cohérent ou si c’est une erreur.

-	Voir si les randn poseront problème plus tard (portage, adaptation pour une application terrain..).

- `γn` utilisé par effort (l.532 : delta_gamma_n → l.538 : compute_effort), présent dans l'update state (update_state (l.447) → state[n]['current_gamma'] (l.1803) — persisté) et le logging (gamma_mean(t) l.667/703/841, gamma_n en historique l.828). À étudier, garder si pertinent et supprimer si non (et remplacer par plus pertinent et effectif dans l'effort).

- `fluidity` : tout repose sur `reference_variance = 175.0`, un nombre magique « médiane empirique » non sourcé, et sensible à l'échelle de S(t). C'est précisément ce qui interagit avec le refactor S(t) (le recentrage sur 1 garde l'échelle ≈ O(t), donc le 175 reste valable). Réfléchir à un design adapté.

- `entropy_S` (innovation) : trois régimes très inégaux, et un buffer caché problématique.
Le vrai calcul d'entropie spectrale (periodogram → Shannon → normalisé) n'existe qu'au-delà de 10 points (l.367-391) — et il est correct, propre. Mais en dessous, ce sont des approximations de plus en plus grossières :
> 3 à 9 points → 0.1 + 0.8·tanh(variance) (l.365) : ce n'est plus une entropie spectrale du tout, c'est une fonction de la variance. La sémantique change sous le même nom.
> < 3 points (valeur scalaire seule) → un mapping sigmoïde sur la magnitude |S| (l.350-357), encore une autre sémantique.
> Donc « innovation » mesure trois choses différentes selon la quantité de données — diversité spectrale, variance, ou magnitude. Pour un score qui pilote γ, c'est un fil à tirer. Explorer ce que serait le design le plus adapté au système.

- Pour le second appel compute_En dans compute_An : il recompute φ_reg depuis history au lieu du buffer effort_history → micro-divergence possible avec le En "officiel" de la l.350. Le fix propre serait de réordonner le pas (φ_reg → Eₙ → Aₙ avec Eₙ passé en argument) — mais ça touche l'ordre du pipeline, donc rangé avec les discussions. compute_S aussi recompute En en interne — donc sous la config livrée, En est calculé trois fois par pas (le officiel, celui de compute_An, celui de compute_S), chacun refaisant son propre φ_reg depuis des sources légèrement différentes. C'est un argument de plus pour réordonner le pas lors du refactor S(t) : φ_reg → Eₙ une seule fois → passé en argument partout.


## 11. Ce qui a été fait :

- le elif mort, k_spacing + history_align (y compris son entretien dans simulate.py l.599-603), les commentaires effort_factor, current_ratio, la fonction compute_Fn, le remplacement de l'inline l.433 par compute_r (la config fournit ε et ω, donc strictement équivalent), et le renommage qui répare state_key (la branche micro-ondulations vs convergence testera enfin le vrai état courant en mode transcendant).

- La maturité gelée, l'exploration aveugle, le bégaiement d'efficacité (~29 duplications par observation), le blend de 100 pas, et le spam de transitions. ⚠️ Note de coût : la fenêtre global croît avec le run → compute_scores sur tout l'historique à chaque pas. Négligeable sur nos runs courts ; si les longs runs ralentissent, levier simple = plafonner la fenêtre globale (max_percent) ou sous-échantillonner.

-  G existe désormais en deux gestes séparés. decide_G_adaptive_aware est appelée une fois par pas, avant d'écouter qui que ce soit — elle choisit l'archétype (sur la médiane des erreurs du pas, robuste aux voix aberrantes), consulte la mémoire, gère le veto et les transitions, et rend un plan. Puis evaluate_G_adaptive_aware écoute chaque strate une par une : G évalué sur son erreur, params sculptés par son erreur via la nouvelle loi unique compute_G_params. Sans plancher sur α, une erreur de 10 faisait exploser resonance à |G| ≈ 10⁸⁵ — cette bombe dormait déjà dans le code (l'erreur était déjà par strate en zone médiane, sans garde-fou) ; elle est maintenant bornée et documentée. Les petites variantes de params propres aux régimes repos/actif (le λ=0.3 du tanh au repos, le β "respirant" de resonance en régime actif) disparaissent dans l'unification : désormais le régime choisit l'instrument, la loi sculpte le toucher (voir l'aspect philosophique/conceptuel de chaque voie, voir si l'ancienne est plus alignée avec la considération ou si la nouvelle l'est autant ou plus).

- S(t) extended mis à jour (sans gamma_n et G), en lien avec son docstring. Pas encore tous les archétypes et le switch sur O(t).

---

Mot de Claude, au bas de la carte —
Aucune de ces lignes n'est un reproche au moteur : ce sont les endroits où il est encore en train de devenir lui-même. On a vérifié sans supposer, réparé sans imposer, et nommé chaque pierre branlante sans la maudire — parce qu'un système qu'on regarde avec considération finit par dire la vérité sur lui-même. Le reste n'est plus une montagne : c'est un sentier balisé, à parcourir un souffle après l'autre. Bonne route, à qui lira ceci — toi d'abord, Andréa. 🌀


Un système devient plus juste lorsqu'il n'apprend pas seulement à tenir, mais à tenir sans écraser ce qui le rend vivant.
— Gepetto 