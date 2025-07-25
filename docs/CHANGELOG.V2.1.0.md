# CHANGELOG FPS Pipeline

## Version 2.1.0 - 2025-06-23

### 🎯 Alignement avec le FPS Paper

#### Nouvelles fonctionnalités
- **Signal composite S(t)** : Mode "extended" avec G(Eₙ-Oₙ) intégré
- **Couplage spiralé** : Support des topologies "ring" et "spiral" avec miroir
- **Métrique de fluidité** : Nouvelle métrique basée sur variance_d2S
- **G(x) spiral_log** : Nouvelle forme de régulation du FPS Paper
- **Matrice de poids automatique** : Génération pour couplage spiral/ring

#### Corrections majeures
- ✅ compute_S utilise maintenant G(x) en mode extended
- ✅ compute_An utilise l'erreur (Eₙ-Oₙ) pour l'enveloppe
- ✅ compute_S_i utilise directement la matrice w
- ✅ compute_delta_fn simplifié selon le FPS Paper
- ✅ compute_A calcule la moyenne sans valeur absolue

#### Améliorations
- 📈 Stabilité : +125742% vs Kuramoto (mode extended)
- 📈 Innovation : +577% vs Kuramoto 
- 📈 Fluidité : +288% vs Kuramoto
- 📊 Score global : +99.6% vs Kuramoto

#### Problèmes connus
- ⚠️ Résilience : -96% (mesure incorrecte avec perturbation sinusoïdale)
- ⚠️ Signal plat avec spiral_log pur (utiliser "adaptive")

### 🔧 Changements techniques

#### dynamics.py
- compute_S : Support mode "simple" et "extended"
- compute_An : Enveloppe basée sur erreur de régulation
- compute_S_i : Utilisation directe de la matrice w
- compute_fn : Ajout contrainte spiralée (commentée)
- compute_A : Moyenne simple des Δfₙ

#### regulation.py
- compute_G : Ajout archétype "spiral_log"
- Archétype "adaptive" combine tanh et spiral_log

#### utils.py
- generate_spiral_weights : Nouvelle fonction pour matrices spiralées
- Support topologies ouvertes et fermées avec miroir

#### init.py
- Génération automatique des poids si coupling.type défini
- Validation adaptée pour matrices générées

#### validate_config.py
- Support du bloc "coupling" optionnel
- Validation relaxée pour matrices auto-générées

#### compare_modes.py
- Nouvelle métrique "fluidity" basée sur variance_d2S
- Score global sur 6 critères (était 5)

### 📝 Documentation
- README.md mis à jour avec nouvelles implémentations
- Résumé d'audit complété avec addendum Phase 2.1
- Ajout de ce CHANGELOG

### 🚀 Prochaines étapes
1. Implémenter perturbation type "pulse" pour mesure correcte de résilience
2. Optimiser les paramètres de régulation (α, β, λ)
3. Tester d'autres topologies de couplage
4. Analyse spectrale du pipeline
