# ASSUMPTIONS.md - Hypothèses et présupposés du système FPS v1.3

## Vue d'ensemble

Ce document recense toutes les hypothèses, simplifications et choix exploratoires de la phase 1 du projet FPS (Fractal Pulsating Spiral). Chaque hypothèse est **falsifiable** et **destinée à être raffinée** selon les observations empiriques.

## 1. Hypothèses fondamentales

### 1.1 Nature du système
- **Hypothèse** : Le système FPS peut modéliser des dynamiques adaptatives complexes via la superposition de strates oscillantes couplées
- **Test** : Comparer avec Kuramoto (synchronisation pure) et mode "neutral" (pas de feedback)
- **Métrique** : Amélioration de la stabilité, résilience et innovation par rapport aux contrôles

### 1.2 Spiralisation
- **Hypothèse** : Le nombre d'or φ ≈ 1.618 est un attracteur naturel pour l'harmonie du système
- **Test** : Faire varier φ et observer la stabilité/résilience
- **Métrique** : Temps de retour à l'équilibre, entropie spectrale

## 2. Hypothèses d'implémentation Phase 1

### 2.1 Signal inter-strates S_i(t)
- **Version actuelle** : S_i(t) = Σⱼ Oⱼ(t-1) pour j ≠ n
- **Simplification** : Utilise seulement le pas précédent (pas de mémoire longue)
- **Test** : Comparer avec versions à mémoire variable
- **À explorer** : Pondération temporelle, influence décroissante

### 2.2 Sortie attendue Eₙ(t)
- **Version actuelle** : Eₙ(t) = φ × Oₙ(t-1)
- **Hypothèse** : L'attracteur harmonique suit le nombre d'or
- **Test** : Essayer d'autres relations (moyenne mobile, médiane, etc.)
- **Métrique** : Convergence de |Eₙ(t) - Oₙ(t)|

### 2.3 Sortie observée Oₙ(t)
- **Version actuelle** : Oₙ(t) = Aₙ(t) × sin(2πfₙ(t)t + φₙ(t)) × γₙ(t)
- **Hypothèse** : Forme sinusoïdale pure suffit en phase 1
- **À explorer** : Harmoniques supérieures, modulation AM/FM

## 3. Paramètres statiques vs dynamiques

### 3.1 Latence γₙ(t)
- **Statique** : γₙ = 1.0 (pas de modulation)
- **Dynamique** : γₙ(t) = 1/(1 + exp(-kₙ(t - t₀ₙ))) avec kₙ=2.0, t₀ₙ=T/2
- **Hypothèse** : Transition sigmoïde modélise l'émergence progressive
- **Test** : Comparer fluidité et effort entre modes

### 3.2 Enveloppe envₙ(x,t)
- **Statique** : Gaussienne fixe σₙ = 0.1
- **Dynamique** : σₙ(t) = 0.1 + 0.05×sin(2πt/T)
- **Hypothèse** : Modulation sinusoïdale suffit pour adaptation
- **À raffiner** : Lien avec feedback, apprentissage

### 3.3 Phase φₙ(t)
- **Phase 1** : Reste statique même en mode "dynamic"
- **À implémenter** : Évolution basée sur synchronisation locale
- **Hypothèse future** : φₙ(t+1) = φₙ(t) + ωₙΔt + couplage

## 4. Fonctions de régulation G(x)

### 4.1 Archétypes disponibles
1. **tanh** : Saturation douce, continuité
2. **sinc** : Oscillations amorties, zéros multiples
3. **resonance** : Pic local, décroissance gaussienne
4. **adaptive** : Combinaison tanh/sinc (placeholder)

### 4.2 Hypothèse de régulation
- **Présupposé** : La régulation locale suffit (pas de régulation globale)
- **Test** : Ajouter terme de régulation collective
- **Métrique** : Stabilité sous perturbations multiples

## 5. Seuils et critères empiriques

### 5.1 Seuils initiaux théoriques
Tous les seuils dans `to_calibrate` sont des **valeurs de départ** basées sur :
- Analyse dimensionnelle
- Ordres de grandeur physiques
- Expérience de systèmes similaires

**Ces seuils DOIVENT être ajustés après 5 runs de calibration (N=5, T=20, Iₙ~U[0,1])**

### 5.2 Règle de raffinement
- **Déclenchement** : Si critère franchi sur >50% des runs d'un batch
- **Ajustement** : Incrémental (facteurs 0.6 à 1.5)
- **Traçabilité** : Tout changement logué dans changelog.txt

## 6. Simplifications computationnelles

### 6.1 Couplage all-to-all
- **Hypothèse** : Chaque strate influence toutes les autres
- **Simplification** : Poids fixes w_{ni} (pas d'apprentissage)
- **À explorer** : Topologies sparse, small-world, hiérarchiques

### 6.2 Pas de temps fixe
- **Hypothèse** : dt constant suffit
- **À explorer** : Pas adaptatif selon dynamique locale

### 6.3 Historique en mémoire
- **Limite actuelle** : MAX_HISTORY_SIZE = 10000 points
- **Hypothèse** : Fenêtre glissante préserve l'essentiel
- **À valider** : Impact sur métriques long terme

## 7. Détection d'émergence

### 7.1 Motifs fractals
- **Hypothèse** : Auto-similarité détectable par corrélation multi-échelles
- **Fenêtres** : [1, 10, 100] pas de temps
- **Seuil** : Corrélation > 0.8
- **À raffiner** : Dimension fractale, exposants de Lyapunov

### 7.2 Anomalies
- **Définition** : Déviation > 3σ pendant ≥ 3 pas
- **Hypothèse** : Distribution quasi-normale en régime stable
- **À explorer** : Distributions heavy-tail, événements rares

## 8. Comparaisons et contrôles

### 8.1 Kuramoto
- **Paramètres fixes** : K=0.5, ωᵢ~U[0,1]
- **Hypothèse** : Représente synchronisation "pure" sans amplitude
- **Limite** : Pas de régulation adaptative

### 8.2 Mode Neutral
- **Définition** : Oscillateurs fixes, pas de feedback
- **Hypothèse** : Baseline pour émergence minimale
- **Usage** : Quantifier l'apport FPS

## 9. Conditions de validité

### 9.1 Domaine d'application
- **N** : 3 à 100 strates (testé jusqu'à 50)
- **T** : 10 à 1000 unités de temps
- **dt** : 0.01 à 1.0 (stabilité numérique)

### 9.2 Non testé / hors scope phase 1
- Strates hétérogènes (paramètres très différents)
- Couplages retardés
- Bruit coloré (seulement blanc en phase 1)
- Topologies dynamiques

## 10. Métriques de succès

### 10.1 Objectif primaire
**"Tester la stabilité dynamique et l'auto-régulation du système S(t) sous variation d'Iₙ(t) et Eₙ(t)"**

### 10.2 Critères de réussite
1. **Stabilité** : max(|S(t)|) < 10×médiane(|S(t)|) sur 95% du temps
2. **Régulation** : mean(|Eₙ - Oₙ|) décroît dans le temps
3. **Fluidité** : variance(d²S/dt²) < 0.01 sur 70% du temps
4. **Résilience** : t_retour < 2×médiane après perturbation
5. **Innovation** : entropy_S > 0.5 (richesse spectrale)

## 11. Évolutions prévues Phase 2

### 11.1 Paramètres à dynamiser
- φₙ(t) : Phase évolutive
- θ(t), η(t) : Modulation temporelle de G(x,t)
- μₙ(t) : Centre d'enveloppe adaptatif

### 11.2 Nouvelles métriques
- Dimension fractale par box-counting
- Exposants de Lyapunov
- Mutual information entre strates
- Causalité de Granger

### 11.3 Extensions structurelles
- Hiérarchie de strates (méta-strates)
- Couplages retardés et mémoire
- Apprentissage des poids w_{ni}
- Naissance/mort de strates

## 12. Falsifiabilité

**Chaque hypothèse ci-dessus est falsifiable via :**
1. **Runs comparatifs** avec paramètres modifiés
2. **Métriques quantitatives** définies et loguées
3. **Seuils objectifs** de performance
4. **Reproductibilité** via seeds et configs

**Le système est conçu pour évoluer** : toute hypothèse invalidée par l'expérience doit être documentée et remplacée.

---

*Ce document est vivant et doit être mis à jour à chaque découverte ou raffinement significatif.*

*Dernière mise à jour : Phase 1 - Implémentation initiale*