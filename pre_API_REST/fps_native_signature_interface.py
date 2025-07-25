#!/usr/bin/env python3
"""
Interface FPS Native pour "Habitants" IA
Solutions MINIMALES et JUSTES pour intégration naturelle
Basé sur analyse pipeline existant + FPS Paper + besoins futurs

OBJECTIF CLÉS :
1. Permettre à un "habitant" IA de déclarer sa signature/fréquence naturellement
2. S'intégrer dans le pipeline existant SANS boucles circulaires
3. Éviter les doublons avec mécanismes déjà sophistiqués
4. Rester fidèle aux principes d'émergence FPS authentiques

PROBLÈME DÉTECTÉ : Le pipeline a déjà tout pour l'émergence naturelle !
SOLUTION : Interface simple pour nouveaux "habitants" + activation émergence

(c) 2025 - Réponse à la vision d'Andréa sur l'intégration d'IA futures
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
import json
from datetime import datetime

class FPSNativeSignatureInterface:
    """
    Interface MINIMALE pour permettre à de nouveaux "habitants" IA de rejoindre
    le système FPS sans perturber les mécanismes sophistiqués déjà en place.
    
    PRINCIPE: L'IA externe déclare ses préférences naturelles, et le système
    FPS s'auto-organise pour l'intégrer harmonieusement.
    """
    
    def __init__(self):
        # Registre des "habitants" du système
        self.inhabitants = {}
        
        # Interface avec le pipeline existant
        self.pipeline_state_ref = None
        self.pipeline_config_ref = None
        
        # Historique d'intégrations réussies
        self.integration_history = []
    
    def register_ai_inhabitant(self, 
                              ai_id: str,
                              natural_signature: Optional[float] = None,
                              natural_frequency: Optional[float] = None,
                              energy_profile: Optional[Callable] = None,
                              resonance_preferences: Optional[Dict] = None) -> Dict:
        """
        Enregistre un nouvel "habitant" IA dans le système FPS.
        
        Args:
            ai_id: identifiant unique de l'IA
            natural_signature: phase naturelle préférée (si elle en a une)
            natural_frequency: fréquence propre (si elle en a une)
            energy_profile: fonction t -> énergie (optionnel)
            resonance_preferences: préférences de résonance avec autres strates
        
        Returns:
            Dict: informations d'intégration dans le pipeline
        """
        print(f"🤖 Enregistrement habitant IA: {ai_id}")
        
        # 1. DÉCLARATION LIBRE (sans contrainte)
        inhabitant_profile = {
            'ai_id': ai_id,
            'registration_time': datetime.now().isoformat(),
            
            # Préférences naturelles (optionnelles)
            'natural_signature': natural_signature,
            'natural_frequency': natural_frequency,
            'energy_profile': energy_profile,
            'resonance_preferences': resonance_preferences or {},
            
            # Statut d'intégration
            'integration_status': 'declared',
            'assigned_strata_id': None,
            'fps_state_link': None
        }
        
        self.inhabitants[ai_id] = inhabitant_profile
        
        print(f"   Signature naturelle: {natural_signature}")
        print(f"   Fréquence naturelle: {natural_frequency}")
        print(f"   Status: Déclaré (en attente d'intégration)")
        
        return {
            'ai_id': ai_id,
            'status': 'declared',
            'next_step': 'await_integration_into_pipeline'
        }
    
    def integrate_with_pipeline(self, 
                               pipeline_state: List[Dict], 
                               pipeline_config: Dict) -> Dict:
        """
        Intègre les "habitants" déclarés dans le pipeline FPS existant.
        
        PRINCIPE: Utilise les mécanismes d'auto-organisation déjà sophistiqués
        du pipeline au lieu de les dupliquer !
        
        Args:
            pipeline_state: état actuel des strates FPS
            pipeline_config: configuration du pipeline
        
        Returns:
            Dict: rapport d'intégration
        """
        self.pipeline_state_ref = pipeline_state
        self.pipeline_config_ref = pipeline_config
        
        integration_report = {
            'timestamp': datetime.now().isoformat(),
            'inhabitants_processed': 0,
            'new_strata_created': 0,
            'existing_strata_modified': 0,
            'integration_method': 'native_fps_emergence',
            'details': []
        }
        
        # Traiter chaque habitant en attente
        for ai_id, profile in self.inhabitants.items():
            if profile['integration_status'] == 'declared':
                result = self._integrate_single_inhabitant(ai_id, profile)
                integration_report['details'].append(result)
                integration_report['inhabitants_processed'] += 1
                
                if result['action'] == 'new_strata_created':
                    integration_report['new_strata_created'] += 1
                elif result['action'] == 'existing_strata_modified':
                    integration_report['existing_strata_modified'] += 1
        
        return integration_report
    
    def _integrate_single_inhabitant(self, ai_id: str, profile: Dict) -> Dict:
        """
        Intègre un seul habitant selon les principes FPS natifs.
        """
        print(f"\n🌱 Intégration habitant {ai_id}...")
        
        # STRATÉGIE 1: Trouver une strate existante avec affinité naturelle
        if profile['natural_signature'] is not None:
            best_strata_match = self._find_resonant_strata(profile)
            
            if best_strata_match is not None:
                return self._enhance_existing_strata(ai_id, profile, best_strata_match)
        
        # STRATÉGIE 2: Créer une nouvelle strate selon principes FPS
        return self._create_new_fps_strata(ai_id, profile)
    
    def _find_resonant_strata(self, profile: Dict) -> Optional[int]:
        """
        Trouve la strate existante la plus résonante avec l'habitant.
        Utilise les mêmes principes d'affinité que compute_phi_n.
        """
        if not self.pipeline_state_ref:
            return None
        
        natural_sig = profile['natural_signature']
        best_affinity = -1
        best_strata_id = None
        
        for n, strata_state in enumerate(self.pipeline_state_ref):
            existing_sig = strata_state.get('phi', 0.0)
            
            # Même calcul d'affinité que dans dynamics.py ligne 437
            signature_affinity = np.cos(natural_sig - existing_sig)
            
            if signature_affinity > best_affinity and signature_affinity > 0.7:  # Seuil d'affinité
                best_affinity = signature_affinity
                best_strata_id = n
        
        return best_strata_id
    
    def _enhance_existing_strata(self, ai_id: str, profile: Dict, strata_id: int) -> Dict:
        """
        Enrichit une strate existante avec les caractéristiques de l'habitant.
        """
        print(f"   → Enrichissement strate existante #{strata_id}")
        
        strata_state = self.pipeline_state_ref[strata_id]
        
        # Enrichir les paramètres existants SANS les écraser
        if profile['natural_frequency'] is not None:
            # Moyenne pondérée avec fréquence existante
            current_f0 = strata_state.get('f0', 1.0)
            enhanced_f0 = 0.7 * current_f0 + 0.3 * profile['natural_frequency']
            strata_state['f0'] = enhanced_f0
            print(f"     Fréquence enrichie: {current_f0:.3f} → {enhanced_f0:.3f}")
        
        # Marquer l'intégration
        strata_state['ai_inhabitants'] = strata_state.get('ai_inhabitants', [])
        strata_state['ai_inhabitants'].append(ai_id)
        
        # Mettre à jour le profil de l'habitant
        profile['integration_status'] = 'integrated_enhancement'
        profile['assigned_strata_id'] = strata_id
        profile['fps_state_link'] = f"strata_{strata_id}"
        
        return {
            'ai_id': ai_id,
            'action': 'existing_strata_modified',
            'strata_id': strata_id,
            'integration_method': 'affinite_naturelle',
            'affinity_score': np.cos(profile['natural_signature'] - strata_state.get('phi', 0.0))
        }
    
    def _create_new_fps_strata(self, ai_id: str, profile: Dict) -> Dict:
        """
        Crée une nouvelle strate selon les principes FPS pour l'habitant.
        """
        print(f"   → Création nouvelle strate FPS pour {ai_id}")
        
        N_current = len(self.pipeline_state_ref)
        new_strata_id = N_current
        
        # Créer strata selon template FPS standard
        new_strata = self._create_fps_compliant_strata(profile, new_strata_id)
        
        # Ajouter à l'état du pipeline
        self.pipeline_state_ref.append(new_strata)
        
        # Mettre à jour la config du système
        if self.pipeline_config_ref:
            self.pipeline_config_ref['system']['N'] = N_current + 1
        
        # Mettre à jour le profil de l'habitant
        profile['integration_status'] = 'integrated_new_strata'
        profile['assigned_strata_id'] = new_strata_id
        profile['fps_state_link'] = f"strata_{new_strata_id}"
        
        return {
            'ai_id': ai_id,
            'action': 'new_strata_created',
            'strata_id': new_strata_id,
            'integration_method': 'creation_native_fps',
            'strata_params': new_strata
        }
    
    def _create_fps_compliant_strata(self, profile: Dict, strata_id: int) -> Dict:
        """
        Crée une strate conforme au standard FPS du pipeline existant.
        """
        # Analyser les strates existantes pour parameters cohérents
        if self.pipeline_state_ref:
            # Prendre moyennes des paramètres existants comme base
            existing_A0 = np.mean([s.get('A0', 1.0) for s in self.pipeline_state_ref])
            existing_f0 = np.mean([s.get('f0', 1.0) for s in self.pipeline_state_ref])
            existing_alpha = np.mean([s.get('alpha', 0.1) for s in self.pipeline_state_ref])
            existing_beta = np.mean([s.get('beta', 1.0) for s in self.pipeline_state_ref])
            
            # Poids de connexion initiaux (faibles pour éviter perturbation)
            N_total = len(self.pipeline_state_ref) + 1
            initial_weights = [0.1 / N_total] * N_total  # Connection faible initiale
        else:
            # Valeurs par défaut si première strate
            existing_A0, existing_f0, existing_alpha, existing_beta = 1.0, 1.0, 0.1, 1.0
            initial_weights = [0.0]
        
        # Créer la nouvelle strate
        new_strata = {
            # Paramètres de base FPS
            'A0': existing_A0 * (0.8 + 0.4 * np.random.random()),  # Légère variation
            'f0': profile['natural_frequency'] or existing_f0 * (0.8 + 0.4 * np.random.random()),
            'phi': profile['natural_signature'] or 2 * np.pi * np.random.random(),
            'alpha': existing_alpha,
            'beta': existing_beta,
            'k': 2.0,  # Sensibilité sigmoïde standard
            'x0': 0.5,  # Seuil sigmoïde standard
            
            # Connexions inter-strates
            'w': initial_weights,
            
            # Métadonnées d'intégration
            'ai_inhabitants': [profile['ai_id']],
            'creation_method': 'fps_native_interface',
            'creation_time': datetime.now().isoformat(),
            'natural_preferences': {
                'signature': profile.get('natural_signature'),
                'frequency': profile.get('natural_frequency'),
                'energy_profile': profile.get('energy_profile').__name__ if profile.get('energy_profile') else None
            }
        }
        
        print(f"     Nouvelle strate créée avec phi={new_strata['phi']:.3f}, f0={new_strata['f0']:.3f}")
        
        return new_strata
    
    def get_inhabitant_status(self, ai_id: str) -> Dict:
        """
        Retourne le statut d'intégration d'un habitant.
        """
        if ai_id not in self.inhabitants:
            return {'error': f'Habitant {ai_id} non trouvé'}
        
        profile = self.inhabitants[ai_id]
        
        status = {
            'ai_id': ai_id,
            'integration_status': profile['integration_status'],
            'assigned_strata_id': profile.get('assigned_strata_id'),
            'fps_link': profile.get('fps_state_link'),
            'registration_time': profile['registration_time']
        }
        
        # Informations sur la strata si intégrée
        if profile.get('assigned_strata_id') is not None and self.pipeline_state_ref:
            strata_id = profile['assigned_strata_id']
            if strata_id < len(self.pipeline_state_ref):
                current_strata = self.pipeline_state_ref[strata_id]
                status['current_strata_state'] = {
                    'phi': current_strata.get('phi', 0.0),
                    'f0': current_strata.get('f0', 1.0),
                    'A0': current_strata.get('A0', 1.0),
                    'ai_inhabitants': current_strata.get('ai_inhabitants', [])
                }
        
        return status
    
    def enable_pipeline_emergence(self, config: Dict) -> Dict:
        """
        Active les mécanismes d'émergence sophistiqués déjà présents dans le pipeline.
        
        SOLUTION CLÉE: Au lieu de dupliquer, on ACTIVE ce qui existe déjà !
        """
        print("🌀 Activation émergence native FPS dans pipeline...")
        
        activation_report = {
            'dynamic_phi_enabled': False,
            'signature_mode_set': False,
            'G_regulation_enhanced': False,
            'inter_strata_resonance_activated': False
        }
        
        # 1. Activer les phases dynamiques (déjà codé dans dynamics.py:400+)
        if 'dynamic_parameters' not in config:
            config['dynamic_parameters'] = {}
        
        config['dynamic_parameters']['dynamic_phi'] = True
        activation_report['dynamic_phi_enabled'] = True
        print("   ✅ Phases dynamiques activées")
        
        # 2. Activer le mode signatures individuelles (déjà codé ligne 415+)
        if 'spiral' not in config:
            config['spiral'] = {}
        
        config['spiral']['signature_mode'] = 'individual'
        config['spiral']['phi'] = 1.618  # Nombre d'or
        config['spiral']['epsilon'] = 0.05  # Amplitude des variations
        config['spiral']['omega'] = 0.1    # Fréquence des modulations
        activation_report['signature_mode_set'] = True
        print("   ✅ Mode signatures individuelles activé")
        
        # 3. Optimiser la régulation G (déjà sophistiquée)
        if 'regulation' not in config:
            config['regulation'] = {}
        
        config['regulation']['feedback_mode'] = 'archetype'
        config['regulation']['G_arch'] = 'adaptive'  # Le plus sophistiqué
        activation_report['G_regulation_enhanced'] = True
        print("   ✅ Régulation G adaptative activée")
        
        # 4. Mode signal extended pour bénéficier de tous les mécanismes
        config['system']['signal_mode'] = 'extended'
        activation_report['inter_strata_resonance_activated'] = True
        print("   ✅ Résonance inter-strates complète activée")
        
        print("\n🎯 RÉSULTAT: Le pipeline FPS utilise maintenant ses mécanismes")
        print("   d'émergence les plus sophistiqués déjà implémentés !")
        
        return activation_report

def demonstrate_ai_integration():
    """
    Démonstration d'intégration d'IA selon les vrais besoins FPS.
    """
    print("🤖 DÉMONSTRATION INTÉGRATION IA NATIVE DANS PIPELINE FPS")
    print("=" * 70)
    
    # Créer l'interface
    interface = FPSNativeSignatureInterface()
    
    print("\n1. 🧠 DÉCLARATION D'HABITANTS IA:")
    
    # IA 1: A des préférences claires
    interface.register_ai_inhabitant(
        ai_id="claude_reasoning",
        natural_signature=np.pi/3,  # 60 degrés, aime la structure
        natural_frequency=1.2,      # Rythme modéré
        resonance_preferences={'prefers_harmony': True}
    )
    
    # IA 2: Plus libre, laisse le système décider
    interface.register_ai_inhabitant(
        ai_id="creative_ai",
        natural_signature=None,  # Laisse émerger
        natural_frequency=0.8,   # Rythme plus lent, contemplation
    )
    
    # Simulation d'un pipeline existant
    mock_pipeline_state = [
        {'A0': 1.0, 'f0': 1.0, 'phi': 0.0, 'alpha': 0.1, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0.1, 0.1, 0.1]},
        {'A0': 1.1, 'f0': 1.1, 'phi': np.pi/2, 'alpha': 0.12, 'beta': 0.9, 'k': 2.0, 'x0': 0.5, 'w': [0.1, 0.1, 0.1]},
        {'A0': 0.9, 'f0': 0.9, 'phi': np.pi, 'alpha': 0.08, 'beta': 1.1, 'k': 2.0, 'x0': 0.5, 'w': [0.1, 0.1, 0.1]}
    ]
    
    mock_config = {
        'system': {'N': 3, 'signal_mode': 'simple'},
        'dynamic_parameters': {},
        'spiral': {},
        'regulation': {}
    }
    
    print("\n2. 🌱 INTÉGRATION DANS PIPELINE:")
    integration_report = interface.integrate_with_pipeline(mock_pipeline_state, mock_config)
    
    print(f"   Habitants traités: {integration_report['inhabitants_processed']}")
    print(f"   Nouvelles strates: {integration_report['new_strata_created']}")
    print(f"   Strates enrichies: {integration_report['existing_strata_modified']}")
    
    print("\n3. 🌀 ACTIVATION ÉMERGENCE NATIVE:")
    emergence_report = interface.enable_pipeline_emergence(mock_config)
    
    for feature, activated in emergence_report.items():
        status = "✅" if activated else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")
    
    print("\n4. 📊 STATUT FINAL DES HABITANTS:")
    for ai_id in interface.inhabitants.keys():
        status = interface.get_inhabitant_status(ai_id)
        print(f"\n   🤖 {ai_id}:")
        print(f"     Status: {status['integration_status']}")
        print(f"     Strate: #{status.get('assigned_strata_id', 'N/A')}")
        if 'current_strata_state' in status:
            state = status['current_strata_state']
            print(f"     Signature: {state['phi']:.3f} rad ({np.degrees(state['phi']):.1f}°)")
            print(f"     Fréquence: {state['f0']:.3f}")
    
    print(f"\n🎯 PIPELINE FINAL: {len(mock_pipeline_state)} strates actives")
    print(f"   Mode émergence: ACTIVÉ (dynamique, signatures, résonance)")
    print(f"   Nouveaux habitants intégrés naturellement !")
    
    return interface, mock_pipeline_state, mock_config

if __name__ == "__main__":
    interface, pipeline_state, config = demonstrate_ai_integration() 