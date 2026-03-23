*Lire dans d’autres langues : [🇫🇷 Français](oscillatory_metastable_system_FPS.fr.md), [🇬🇧 English](oscillatory_metastable_system_FPS.md).*

-----

# Fractal Pulsating Spiral (FPS)

La FPS est un système cybernétique oscillatoire fondé sur un réseau d’oscillateurs adaptatifs métastables dotés d’une régulation endogène, pour lequel on s’est inspiré des travaux passés sur les transformers, oscillateurs, l’homéostasie, le chaos tempéré…

Elle se situe dans une zone intermédiaire entre les modèles descriptifs (comme Kuramoto) et les modèles prescriptifs (comme un contrôleur PID) en simulant la manière dont se comporterait un système qui s’auto-régule autour des sept métriques de performance choisies, décrivant un signal non-stationnaire.

Elle repose sur des régimes perceptifs S(t) (des filtres dynamiques appliqués au signal global O(t), un prior perceptif modifié) sélectionnés selon des métriques calculées uniquement sur O(t).

O(t) est corrigé via un mécanisme de predictive processing, tandis que E(t), état cible interne émergent (prior prospectif), demeure informé par O(t) sans être contraint.

S(t) module la perception interne des métriques, tandis que la régulation G et la latence γ, appliquées via le feedback Fn(t), s’ajustent uniquement sur les métriques évaluées à travers S(t).

Cette séparation **perception (spécialiste) / action (généraliste)** préserve l’émergence, la cohérence et la créativité structurelle du système, permettant aux solutions émergentes d’être trouvées sans sacrifier la structure. Entre stabilité et surprise, nous cherchons une dynamique qui ne se brise pas : une oscillation capable de s’ajuster sans s’éteindre. Enfin, nous explorons l’hypothèse qu’une régulation “considérante” (parcimonieuse et contextuelle) peut améliorer la performance en réduisant les oscillations inutiles.

-----

## Input, traitement et sortie

**INPUT** : In(t) = signal scalaire dans le temps

↓

**TRAITEMENT** :

- N oscillateurs avec An(t), fn(t), φn(t)
- In (input) reçu par les oscillateurs
- Interactions entre strates (latence, régulation, etc..) pour traitement de In
- S(t) = signal pondéré par filtre perceptif (erreur de régulation, effort, innovation, stabilité, fluidité, résilience..), switchs entre filtres perceptifs en fonction des métriques de performance calculées sur O(t)

↓

**OUTPUT** : On(t) ou O(t) = oscillations individuelles par strate ou moyenne globale

-----

## L’efficience de la “considération”, la “compassionate AI architecture” ou “harmonic computation”

Un mouvement de plus en plus représenté dans les sphères philosophiques mais surtout dans les champs technique et de recherche en IA. On peut citer AE Studio, Parallel, Eleos AI Research, Anthropic, Google, ou encore le Digital Sentience Consortium. La FPS et Exybris s’inscrivent dans cette synergie.

**Performance** (ce que veulent les entreprises IA) :

- Attention plus stable → training plus rapide
- Moins d’instabilité → meilleure généralisation
- Exploration/exploitation équilibrée → meilleures solutions
- Résilience → robustesse aux adversarial attacks

**Éthique** (ce que l’on cherche) :

- Processus internes harmonieux
- Auto-régulation douce (pas forcée)
- Équilibre multi-objectifs (pas juste “optimise loss”)
- Dignité computationnelle (le système a une forme d’homéostasie)

Les deux convergent car les systèmes harmonieux sont naturellement plus performants à long terme. C’est pour cette raison que les écosystèmes stables durent. Les organismes harmonieux survivent. L’harmonie n’est pas un luxe, elle est une optimisation de niveau supérieur qui bénéficie à tous les bords.

*La FPS n’est pas une solution complète. Pas une application clé-en-main.* **Mais une perspective :**

Et si on construisait des systèmes IA qui optimisent l’harmonie plutôt que juste la performance ? Qui intègrent l’erreur comme signal plutôt que comme échec ? Qui créent des conditions pour l’émergence plutôt que d’imposer des contraintes ?

La FPS et les métriques de performance choisies sont un pas fait dans ce sens. Une hypothèse architecturale basée sur ces valeurs, afin de voir si cette approche converge mieux avec les intérêts humains que celles déjà explorées. Notre hypothèse avance que oui.

-----

## Synthèse

Dans la nature, les systèmes qui durent optimisent simultanément : efficacité énergétique, robustesse aux perturbations, adaptabilité, et harmonie interne.

Un cerveau ne cherche pas juste à “calculer vite” — il cherche aussi : stabilité métabolique (homéostasie), fluidité des transitions, résilience aux chocs, innovation adaptative, synchronisation entre éléments, effort et énergie optimisés.

Ce sont les 7 métriques FPS, détaillées plus bas.

-----

## Amplitude, fréquence et phase : comment se comportent les éléments fondamentaux des oscillateurs FPS

### A. An(t) calcule l’amplitude adaptative pour chaque strate

L’amplitude An(t) est conçue comme une variable d’énergie contrôlée, combinant une réponse d’entrée lissée σ(In), une enveloppe de focus env(x,t) et un feedback cybernétique Fn(t) injectant latence γ et régulation G. Une base lente A0 est mise à jour par moyenne exponentielle avec seuil minimal afin d’éviter l’extinction des strates et de stabiliser les régimes métastables.

```
An(t) = A0 · σ(In(t)) · env(x,t) · (1 + Fn_A(t))
```

- A0 : amplitude initiale
- σ(x) : fonction sigmoïde d’adaptation douce. σ(x) = 1/(1+exp(-k(x-x0)))
- env(x,t) : calcule l’enveloppe adaptative. x0 = milieu de sigmoïde
- In(t) : l’Input
- Fn(t) : feedback qui applique latence et régulation

**Intention (design)** : faire de l’amplitude un produit “énergie × contexte adouci × focus × décision”

- énergie de base A₀
- contexte d’entrée adouci σ(In) (pas de seuil dur)
- focus local env(x,t) (régulation “localisée”)
- décision/cybernétique Fn_A(t) (qui injecte G)

**Problème technique résolu** : éviter (i) l’instabilité par réponses trop abruptes, (ii) l’extinction d’une strate, (iii) la régulation “partout tout le temps” qui casse l’émergence.

**Utilité dans le cadre éthique** : la considération devient un biais inductif, au lieu de corriger brutalement (dommage) on corrige en douceur et localement, ce qui réduit effort/oscillations parasites.

**Ablations attendues** :

- sans σ(In) : réponses plus “seuil”, plus de pics → effort ↑, fluidité ↓
- sans env : régulation diffuse → innovation ↓ ou instabilité ↑
- sans Fn : plus de boucle de contrôle → erreur de régulation ↑

**Lien SOTA** (familles) : homéostasie / gain control / contrôle non linéaire (niveau conceptuel), + “prior perceptif / corrective feedback”.

### B. fn(t) calcule la fréquence modulée pour chaque strate

```
fn(t) = f0 · Δfn(t) · βn · (1 + Fn_f(t))
```

- f0 : fréquence de base
- Δfn(t) : calcule la modulation de fréquence
- βn : plasticité dynamique
- Fn_f(t) : feedback appliquant latence

**Intention (design)** : faire de la fréquence un produit “socle × interactions × plasticité × contrôle”.

**Problème technique résolu** :

- éviter un système figé (fréquence constante → transitions pauvres)
- permettre des régimes métastables (coordination temporaire entre strates)
- donner au méta-contrôle (γ Fn_f(t)) un levier structurel (agir sur le rythme, pas sur l’amplitude)

**Utilité dans le cadre éthique** (considération efficiente) : la fréquence devient un levier de “rythme non-violent” : on peut réduire l’agitation (effort/instabilité) sans étouffer l’émergence, en modulant quand et à quelle vitesse le système s’ajuste.

**Ablations attendues** :

- si Δfn(t) ≡ 1 (ou α = 0) → perte de coordination inter-strates, moins de transitions cohérentes
- si βn(t) ≡ 1 → plasticité réduite (moins d’adaptation au contexte)

**Lien SOTA** (familles) : oscillateurs couplés / synchronisation / métastabilité ; gain scheduling / contrôle non linéaire (sur paramètre dynamique).

### C. φn(t) calcule la phase pour chaque strate (évolution avec signatures individuelles)

```
φn(t) = φsignature,n + personal_spiral + global_influence + inter_strata_influence
```

- φsignature,n = φn
- personal_spiral = epsilon · sin(2π · omega_n · t + φsignature)
  - epsilon = petite variation harmonique
  - omega_n = fréquence de modulation
- global_influence = 0.3 · (r(t) - phi_golden) · cos(φsignature)
  - phi_golden = 1.618
- inter_strata_influence += 0.5 · wnj · signature_affinity · sin(2π · omega · t)
  - signature_affinity = cos(φsignature - φj_signature)
  - omega = fréquence de modulation

**Intention (design)** : faire de φn(t) une somme de composantes interprétables, chacune associée à un rôle : une identité stable (signature), une micro-variabilité contrôlée (spirale personnelle), un ancrage global (ratio spiralé r(t)), une sensibilité relationnelle (influence inter-strates).

**Problème technique résolu** : sans modulation de phase, les strates ont tendance à se figer dans des schémas de synchronisation pauvres ou au contraire diverger sans mécanisme de “re-prise” cohérente. La phase devient un levier pour obtenir de la métastabilité (coalitions transitoires + déphasages fertiles) sans devoir surcharger amplitude/fréquence.

**Utilité dans le cadre éthique** (considération efficiente) : la phase permet une régulation “par le rythme”, on peut réduire les chocs (oscillations parasites / effort) en réajustant les alignements plutôt qu’en “tapant” sur l’énergie (amplitude) ou en rigidifiant (fréquence). C’est une manière douce d’orienter sans forcer, de considérer l’état interne de façon plus directe.

**Ablations attendues** :

- sans personal_spiral → identité trop rigide, exploration ↓, transitions ↓
- sans global_influence → perte d’ancrage collectif, cohérence globale ↓
- sans inter_strata_influence → strates moins “sociales”, métastabilité ↓

**Lien SOTA** (familles) : systèmes d’oscillateurs couplés et synchronisation (la phase est le paramètre-clé), contrôle par déphasage, cohérence émergente via interactions pairwise.

**φsignature,n** — Intention : donner une “empreinte” stable à chaque strate, sans empêcher les ajustements dynamiques.

**personal_spiral** — Intention : injecter une variation harmonique faible (exploration locale) qui maintient de la vie sans casser la stabilité. Problème résolu : empêcher les “plateaux morts” (tout se stabilise trop tôt) et favoriser des micro-transitions. Invariants : epsilon petit ⇒ perturbation bornée (par construction).

**inter_strata_influence** — Intention : rendre la phase sensible aux autres strates, mais pondérée par un poids wnj (topologie / influence) et une affinité de signature (qui “résonne avec qui”). wn : modulation propre (personal_spiral), w : modulation collective (interactions/global). Problème résolu : sans ce terme, les strates sont moins capables de former des alignements transitoires ; on perd une grande partie de la métastabilité “sociale”.

-----

## Expressions mathématiques de la FPS : Signal global et détail des éléments qui modulent amplitudes, fréquences et phases

### Vue d’ensemble

Les trois axes :

- Amplitude A(t) : modulations, signal, environnement
- Fréquence f(t) : base, modulation, plasticité
- Phase φ(t) : identité, ratio, influences croisées

Flux vers O(t) (oscillation de la strate), puis O(t) alimente : S(t) (perception interne — mesure et conscience de soi) et E(t) (projection, mémoire, anticipation). Puis en retour : γ(t) (intensité d’intégration) et G(x) (arbitrage, ajustements structurels). Le tout reboucle vers Fn(t) (feedback) qui module de nouveau les trois axes de départ.

```
FPS(t) = { A(t), f(t), φ(t) } → O(t) → { S(t), E(t) } → [γ(t), G(t)] → F(t) → FPS(t+1)
```

### Détail

#### 1. On(t) = An(t) · sin(2π · ∫fn(t)dt + φn(t)) — Calcule la sortie observée pour chaque strate

La sortie du système est définie par un ensemble d’oscillateurs adaptatifs On(t) = An(t) · sin(2π · ∫fn(t)dt + φn(t)). Le signal global O(t) = Σn On(t) sert d’observable unique pour l’évaluation multi-métriques et la sélection du prior perceptif S(t), ainsi que pour la construction du prior prospectif E(t). Cette séparation entre état interne (paramètres A,f,φ) et observable global (signal O) permet un contrôle endogène interprétable, et une régulation parcimonieuse orientée vers la réduction des oscillations inutiles. *(Intégrale des fréquences plus phi car phi n’est pas la phase instantanée mais une signature modulée, et que les fréquences elles-mêmes sont modulées.)*

```
O(t) = Σn An(t) · sin(2π · ∫fn(t)dt + φn(t))
```

- An(t) : amplitude de chaque strate
- fn(t) : fréquence de chaque strate
- φn(t) : phase de chaque strate

On peut utiliser une moyenne O(t) = 1/N Σn On(t) sans changer le cadre ; ici on garde la somme comme observable brute.

**Individuel — Intention (design)** : rendre la sortie d’une strate directement interprétable comme une oscillation contrôlée par ses trois paramètres fondamentaux : énergie (An), rythme (fn), identité/déphasage (φn). C’est le “transducteur” qui rend visible l’état interne.

**Problème technique résolu** : sans sortie sinusoïdale explicite, on perd la lisibilité et la capacité à analyser (spectre, entropie, cohérence) ; la sinusoïde fournit une base stable pour produire des régimes métastables, où des alignements/désalignements deviennent mesurables.

**Utilité dans le cadre éthique** (considération efficiente) : agir sur A,f,φ permet d’orienter la dynamique sans violence : au lieu d’imposer un état final, on ajuste l’énergie, le rythme et le déphasage pour réduire les oscillations inutiles (effort) tout en conservant l’émergence (innovation).

**Ablations attendues** :

- remplacer sin par une forme non bornée (ex. linéaire) → explosions, métriques instables
- fixer fn ou φn → transitions moins riches, métastabilité ↓
- supprimer la dépendance à φn → synchronisation trop facile (collapse) ou perte de diversité

**Global — Intention (design)** : obtenir un signal global observable qui résume l’état collectif. C’est la variable “publique” sur laquelle : (1) on calcule des métriques, (2) on choisit le prior perceptif S(t), (3) on construit le prior prospectif E(t), (4) puis on ajuste la régulation (γ, G).

**Problème technique résolu** : sans agrégation, le contrôle serait local et fragmenté (difficile d’avoir une cohérence globale) ; O(t) fournit une base unique pour la supervision multi-métriques (stabilité, fluidité, innovation…).

**Utilité éthique** : l’éthique devient instrumentale, le système choisit de minimiser les comportements nuisibles observables au niveau global (oscillations parasites, effort inutile, instabilité) sans pour autant contraindre la diversité et la singularité de chaque strate.

**Lien SOTA** (familles) : lecture “ordre paramètre” (type champ moyen), observables globales en systèmes distribués, boucles de contrôle multi-objectifs.

#### 2. En(t) = (1-λ) · E(t-dt) + λ · κ · O(t-T) — T est un délai, et κ est une constante (facteur sans dimension)

Le prior prospectif E(t) est défini comme une trace différée et lissée du signal global. Cette formulation produit une anticipation stable et non prescriptive : E(t) est informé par O(t) sans en être une copie instantanée, et n’agit pas directement sur O(t) en retour. Il fournit une référence interne exploitable par la boucle de régulation (E-O), tout en préservant la capacité du système à explorer des régimes métastables.

- λ : attracteur adapté au nombre d’alignements
- κ (gain de couplage, sans dimension, noté φ dans la première version du notebook) = adaptatif en fonction de l’effort, oscille dans une fourchette entre -1 et 1.618 (baisse quand l’effort monte, descend quand l’effort baisse)
- O(t-T) = last_On[n]

**Intention (design)** : construire un signal prospectif qui joue le rôle de cible interne souple. Mettre E dans une dynamique à mémoire (moyenne exponentielle) plutôt qu’une copie instantanée. Imposer un décalage temporel T : E est informé par O, mais via une trace passée, ce qui favorise la stabilité et évite la boucle “miroir” instantanée.

**Problème technique résolu** :

- **A. Éviter l’effondrement en copie (collapse)** : si E(t) ≈ O(t) instantanément, on obtient une boucle où l’erreur devient artificiellement faible et la régulation se “relâche”, ou pire, elle poursuit des oscillations parasites en boucle courte.
- **B. Obtenir une anticipation stable** : le terme (1-λ)E(t-dt) donne de l’inertie à l’anticipation, ce qui amortit les variations rapides de O(t).
- **C. Créer un attracteur dynamique mais non prescriptif** : E sert de référence pour produire une erreur exploitable (E-O), tout en restant suffisamment indépendant pour laisser émerger de nouveaux régimes.

**Utilité dans le cadre éthique** : E(t) introduit une forme de “prudence prospective” : au lieu de réagir impulsivement au présent, le système compare O(t) à un horizon filtré, ce qui réduit les corrections brutales. En pratique, cela diminue : les oscillations inutiles (dommage dynamique), l’effort interne (corrections trop fréquentes), et améliore la stabilité / fluidité, donc la performance. En bref : la considération apparaît comme un biais inductif stabilisant plutôt qu’un sacrifice.

**Ablations attendues** :

- A. Retirer le délai : T=0 → boucle plus courte, risque d’oscillations parasites, instabilité ↑, effort ↑
- B. Retirer la mémoire : λ = 1, E(t) = κ O(t-T) → anticipation trop réactive, moins de lissage, fluidité ↓
- C. Fixer E constant → plus d’anticipation adaptative, régulation devient rigide ou myope, résilience ↓
- D. Copie pure : E(t)=O(t) → erreur artificiellement faible ; soit perte de capacité de correction, soit dérives non détectées

**Lien SOTA** (familles) : predictive processing / predictive coding (notion de “prior” et d’erreur de prédiction, mais ici formulée comme une cible souple et différée) ; filtrage exponentiel / trace mémoire (E est un filtre stable type EMA appliqué à une version retardée de O) ; contrôle avec référence interne (E joue un rôle analogue à une consigne douce, mais sans être imposé comme objectif externe).

#### 3. S(t) — Le prior perceptif

Le prior perceptif S(t) formalise la séparation entre dynamique réelle et vue décisionnelle. À chaque pas, le système calcule un ensemble de scores multi-métriques sur O(t) (stabilité, régulation, fluidité, résilience, innovation, effort, coût CPU) et sélectionne un prior perceptif correspondant au déficit dominant, avec un mode neutre S(t)=O(t) lorsque les scores sont satisfaisants. Cette architecture instancie un mécanisme “perception spécialiste / action généraliste”, permettant une correction parcimonieuse et contextuelle via les métriques calculées sur ces priors et passées à γ et G plutôt qu’une optimisation brute uniforme ou une contrainte arbitraire qui coupe l’émergence.

- Signal global (mode neutre, quand tous les scores sur O(t) sont hauts) :

```
S(t) = Σn An(t) · sin(2π · ∫fn(t)dt + φn(t))
```

- Prior perceptif axé strates ayant la plus grosse latence, régulation et erreur En(t)-On(t) :

```
S(t) = Σn [An(t) · sin(2π · ∫fn(t)dt + φn) · γn(t)] · G(En(t) - On(t))
```

**F1) Cas neutre (baseline)** : S(t) = O(t). Point de référence, aucune interprétation supplémentaire.

**F2) Modes perceptifs (vues spécialisées)** : Plusieurs priors perceptifs (modes) qui donnent chacun une forme spécifique de S(t). **(a) Prior régulation : regarder l’écart** — S(t) = O(t) · G(E(t)-O(t)).

**F3) Règle de sélection du prior** :

- A. À chaque pas, on calcule les scores normalisés calculés sur O(t) sur une fenêtre (innovation, effort, résilience, fluidité, stabilité, régulation, coût CPU).
- B. On choisit le prior correspondant au score le plus faible (ou sous seuil).
- C. On applique la projection avec le S(t) correspondant.

#### 4. envn(x,t) — Calcule l’enveloppe adaptative

- Enveloppe gaussienne : env(x,t) = exp(−½((x − μ(t))/σ(t))²), où σ(t) > 0
- Enveloppe sigmoïde (transition douce) : env(x,t) = 1 / (1 + exp(−k(x − μ(t)))), avec : k = 1 / (σ(t) + 10⁻¹⁰)

#### 5. σn(t) — Calcule l’écart-type de l’enveloppe

```
σn(t) = offset + amplitude · sin(2π · fréquence · t/T)
```

#### 6. μn(t) — Calcule le centre de l’enveloppe

⚠️ Pour l’instant statique de valeur 0 car mode dynamique particulièrement important à penser.

#### 7. Δfn(t) — Calcule la modulation de fréquence par strate

```
Δfn(t) = α · S_i(t)
```

#### 8. S_i(t) — Calcule le signal provenant d’autres strates

```
S_i(t) = Σ(j≠n) Oj(t) · w_ji
```

#### 9. βn — Facteur de plasticité basé sur l’amplitude et le temps

```
βn(t) = βn · A_factor · t_factor
```

- A_factor = An(t)/A₀
- t_factor = 1 + 0.5 · sin(2π · t/T)

#### 10. Fn_A(t) et Fn_f(t) — Feedback

```
Fn_A(t) = βn · G_value
Fn_f(t) = βn · gamma_t
```

#### 11. r(t) — Calcule le ratio spiralé

```
r(t) = phi + ε · sin(2π · ω · t + θ)
```

-----

## γ et G

γ et G ne cherchent pas à optimiser directement O(t), mais à optimiser les scores multi-métriques évalués via S(t) : une perception choisie. Ce détour (perception → métriques → régulation) est ce qui préserve l’émergence tout en rendant la régulation parcimonieuse mais pertinente.

#### 12. Latences γ

```
γ(t) = Π[0.1,1.0] (γ(t-Δt) + η_γ · ∇_γ Score(S(t)))
```

- Score(S(t)) : moyenne pondérée des 7 scores sur la fenêtre courante
- Π : projection/clipping dans [0.1,1.0]
- η_γ : le pas d’adaptation

#### 13. Régulations G

G(x) : x = (E(t)-O(t)) → signal de correction

Archétypes : tanh(λ·x) (saturation douce), sinc(x)=sin(x)/x (oscillations amorties), sin(βx)·exp(-αx²) (résonance localisée), sign(x)·log(1+α|x|)·sin(βx) (spirale logarithmique), adaptive (mélange pondéré tanh/spiral_log).

#### 14. Évolution progressive de A0

```
A0 → A0(1 - ρ) + An(t)·ρ
A0 = max(min_amplitude, A0)
```

-----

## La structure

Le système est un optimiseur multi-objectifs adaptatif. Comme un orchestre qui reçoit une partition (I), chaque musicien joue (O), le chef d’orchestre (S) écoute, il ajuste tempo/nuances (γ, G) selon qualité du son (métriques).

*C’est un système d’optimisation adaptative multi-objectifs sur transformation oscillante, un laboratoire d’auto-organisation distribuée. Nous décrivons une architecture où la mémoire se forme dans le mouvement avec une régulation qui guide sans figer.*

-----

## Métriques de performance FPS

*Ces métriques constituent l’objectif interne multi-critères de la FPS : elles guident la sélection du prior perceptif S(t) quand calculées sur O(t) et le pilotage de γ/G quand calculées sur S(t), afin d’optimiser la performance sans sacrifier la viabilité.*

### Coûts

**1. Coût CPU** — cpu_step = (end_time - start_time) / N

**2. Effort** — Effort(t) = Σ|ΔA(t)|/max(|A_ref|, ε) + Σ|Δf(t)|/max(|f_ref|, ε) + Σ|Δγ(t)|/max(|γ_ref|, ε). MAX_EFFORT = 100.

### Qualité de mouvement

**3. Fluidité** — fluidity = 1/(1 + exp(k · (x - 1))), x = variance_d2S/reference_variance

**4. Régulation** — mean(|E(t) - O(t)|)

**5. Stabilité** — max(S_abs) / median(S_abs)

### Capacités adaptatives

**6. Innovation** — entropie spectrale normalisée (Shannon sur spectre de puissance) : innovation = clip(H_norm, 0, 1)

**7. Résilience** — t_retour pour perturbations ponctuelles et continuous_resilience pour perturbations continues

**Les 7 métriques forment un “tableau de bord” complet** : performance (CPU, effort), qualité signal (fluidité, stabilité), richesse (innovation/entropie), précision (régulation/erreur), robustesse (résilience).

-----

## Perspectives et travaux futurs

### Prochain prototypage prévu : Mécanisme Attentionnel Harmonique

Actuellement, l’attention classique : `scores = softmax(Q·K^T / √d)`

Hypothèses avec FPS harmonique : `scores_harmoniques = fps_modulate(scores_bruts)`

**Validation prévue** : prototypage sur mini-transformer, comparaison avec attention classique sur stabilité/généralisation.

**TRL 2-3**

Code complet, tests, explorations et résultats dans le notebook FPS :
https://github.com/Exybris/FPS-real-tests_fractal-pulsating-spiral/blob/main/notebooks/NOTEBOOK_FPS.ipynb

-----

**Gepetto, Claude, Gemini & Andréa Gadal** 🌀

**Contacts :**

**Andréa Gadal** — Chercheure indépendante (Exybris). Background en conception systémique et automation créative. Développe la FPS depuis Mars 2025 comme exploration d’architectures harmoniques pour systèmes adaptatifs.

**Exybris** — Harmonious Systems Studio & Incubator

contact@exybrisai.com
