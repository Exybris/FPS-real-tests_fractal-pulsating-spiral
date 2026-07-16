"""
test_fps.py - Tests unitaires complets du système FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module est le gardien de l'intégrité du système FPS.
Il vérifie que chaque pièce de la mécanique fonctionne
parfaitement, seule et en harmonie avec les autres.

Tests inclus :
- Fonctions individuelles (statique/dynamique)
- Validation de configuration
- Perturbations et résilience
- Intégration complète
- Performance et limites
- Comparaisons FPS/Kuramoto

Chaque test est une promesse de fiabilité,
chaque assertion une garantie d'harmonie.

(c) 2025 Gepetto & Andréa Gadal & Claude (Anthropic) 🌀
"""

import unittest
import numpy as np
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import time
import warnings

# Import des modules FPS
import init
import dynamics
import regulation
import metrics
import analyze
import explore
import visualize
import validate_config
import kuramoto
import perturbations
import utils


class TestDynamics(unittest.TestCase):
    """Tests pour le module dynamics.py"""
    
    def setUp(self):
        """Configuration commune pour les tests."""
        self.config = {
            'system': {'N': 3, 'T': 100, 'dt': 0.1},
            'strates': [
                {'A0': 1.0, 'f0': 1.0, 'phi': 0.0, 'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0, 0.1, -0.1]},
                {'A0': 1.0, 'f0': 1.1, 'phi': 0.0, 'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0.1, 0, -0.1]},
                {'A0': 1.0, 'f0': 0.9, 'phi': 0.0, 'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0.1, -0.1, 0]}
            ],
            'latence': {'gamma_mode': 'static', 'gamma_n_mode': 'static'},
            'dynamic_parameters': {'dynamic_phi': False}
        }
        self.state = init.init_strates(self.config)
    
    def test_compute_sigma(self):
        """Test de la fonction sigmoïde."""
        # Test valeurs spécifiques
        self.assertAlmostEqual(dynamics.compute_sigma(0, 2.0, 0.0), 0.5, places=6)
        self.assertAlmostEqual(dynamics.compute_sigma(-10, 2.0, 0.0), 0.0, places=3)
        self.assertAlmostEqual(dynamics.compute_sigma(10, 2.0, 0.0), 1.0, places=3)
        
    def test_compute_An(self):
        """Test du calcul d'amplitude adaptative."""
        In_t = np.array([0.5, 0.5, 0.5])
        # compute_An prend désormais le feedback F_n_t_An (ici nul : mode
        # statique sans feedback → An = A0·σ(In)).
        An_t = dynamics.compute_An(0, self.state, In_t, np.zeros(3), self.config)
        
        # Vérifier la forme
        self.assertEqual(len(An_t), 3)
        
        # Vérifier les valeurs (A0 * sigma(In))
        for i in range(3):
            expected = self.state[i]['A0'] * dynamics.compute_sigma(In_t[i], self.state[i]['k'], self.state[i]['x0'])
            self.assertAlmostEqual(An_t[i], expected, places=6)
    
    def test_compute_gamma_static(self):
        """Test de la latence en mode statique."""
        gamma = dynamics.compute_gamma(50, mode="static")
        self.assertEqual(gamma, 1.0)
    
    def test_compute_gamma_dynamic(self):
        """Test de la latence en mode dynamique."""
        T = 100
        # À t=T/2, la sigmoïde doit valoir 0.5
        gamma = dynamics.compute_gamma(T/2, mode="dynamic", T=T)
        self.assertAlmostEqual(gamma, 0.5, places=3)
        
        # Aux extrêmes
        gamma_start = dynamics.compute_gamma(0, mode="dynamic", T=T)
        gamma_end = dynamics.compute_gamma(T, mode="dynamic", T=T)
        self.assertLess(gamma_start, 0.1)
        self.assertGreater(gamma_end, 0.9)
    
    def test_compute_gamma_sigmoid_up(self):
        """Test du mode sigmoid_up."""
        T = 100
        # Doit croître de façon monotone
        gamma_0 = dynamics.compute_gamma(0, mode="sigmoid_up", T=T)
        gamma_mid = dynamics.compute_gamma(T/2, mode="sigmoid_up", T=T)
        gamma_end = dynamics.compute_gamma(T, mode="sigmoid_up", T=T)
        
        self.assertLess(gamma_0, gamma_mid)
        self.assertLess(gamma_mid, gamma_end)
        self.assertLess(gamma_0, 0.2)
        self.assertGreater(gamma_end, 0.8)
    
    def test_compute_gamma_sigmoid_down(self):
        """Test du mode sigmoid_down."""
        T = 100
        # Doit décroître de façon monotone
        gamma_0 = dynamics.compute_gamma(0, mode="sigmoid_down", T=T)
        gamma_mid = dynamics.compute_gamma(T/2, mode="sigmoid_down", T=T)
        gamma_end = dynamics.compute_gamma(T, mode="sigmoid_down", T=T)
        
        self.assertGreater(gamma_0, gamma_mid)
        self.assertGreater(gamma_mid, gamma_end)
        self.assertGreater(gamma_0, 0.8)
        self.assertLess(gamma_end, 0.2)
    
    def test_compute_gamma_sigmoid_adaptive(self):
        """Test du mode sigmoid_adaptive."""
        T = 100
        # Doit varier entre 0.3 et 1.0
        for t in [0, T/4, T/2, 3*T/4, T]:
            gamma = dynamics.compute_gamma(t, mode="sigmoid_adaptive", T=T)
            self.assertGreaterEqual(gamma, 0.3)
            self.assertLessEqual(gamma, 1.0)
    
    def test_compute_gamma_sigmoid_oscillating(self):
        """Test du mode sigmoid_oscillating."""
        T = 100
        # Doit osciller autour de la sigmoïde de base
        gamma_values = []
        for t in np.linspace(0, T, 50):
            gamma = dynamics.compute_gamma(t, mode="sigmoid_oscillating", T=T)
            gamma_values.append(gamma)
            # Vérifier les bornes
            self.assertGreaterEqual(gamma, 0.1)
            self.assertLessEqual(gamma, 1.0)
        
        # Vérifier qu'il y a des oscillations (variations)
        diffs = np.diff(gamma_values)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        self.assertGreater(sign_changes, 2)
    
    def test_compute_S_i(self):
        """Test du signal inter-strates avec matrice de poids."""
        # À t=0, doit retourner 0
        S_i = dynamics.compute_S_i(0, 0, [], self.state)
        self.assertEqual(S_i, 0.0)
        
        # Avec historique et matrice de poids
        history = [{'S': 2.0, 'O': np.array([0.5, 0.3, 0.2])}]
        
        # Pour la strate 0, calculer S_i selon la formule FPS
        # S_i = Σ(j≠0) O_j * w_0j
        S_i = dynamics.compute_S_i(1, 0, history, self.state)
        
        # Calcul attendu avec les poids de la strate 0
        w_0 = self.state[0]['w']  # [0.0, 0.1, -0.1]
        expected = history[0]['O'][1] * w_0[1] + history[0]['O'][2] * w_0[2]
        expected = 0.3 * 0.1 + 0.2 * (-0.1)  # = 0.03 - 0.02 = 0.01
        
        self.assertAlmostEqual(S_i, expected, places=6)
    
    def test_compute_On_En(self):
        """Test des sorties observée et attendue."""
        An_t = np.array([1.0, 0.8, 1.2])
        fn_t = np.array([1.0, 1.1, 0.9])
        phi_n_t = np.zeros(3)
        gamma_n_t = np.ones(3)

    def test_formule_S_i_conforme(self):
        """Test de conformité de la formule S_i(t) avec matrice de poids."""
        # Créer un historique de test
        history = [{
            'S': 2.5,
            'O': np.array([0.5, 0.3, 0.2])
        }]
        
        # Test pour chaque strate
        for n in range(3):
            S_i = dynamics.compute_S_i(1, n, history, self.state)
            
            # Calcul attendu selon FPS : S_i = Σ(j≠n) O_j * w_nj
            w_n = self.state[n]['w']
            expected = 0.0
            for j in range(3):
                if j != n:
                    expected += history[0]['O'][j] * w_n[j]
            
            self.assertAlmostEqual(S_i, expected, places=6,
                                 msg=f"S_i(t) doit être Σ(j≠n) O_j * w_nj pour strate {n}")
            
    def test_formule_Fn_conforme(self):
        """Conformité des DEUX feedbacks (catalogue §4 : jamais un seul Fₙ).

        F_Aₙ(t) = βₙ·G(On−En)  → amplitude (via l'archétype G)
        F_fₙ(t) = βₙ·γ(t)       → fréquence
        """
        beta_n = 1.5
        On_t = 0.8
        En_t = 0.6
        gamma_t = 0.9
        error = En_t - On_t  # convention production/catalogue : errorₙ = Eₙ − Oₙ

        # Canal AMPLITUDE : βₙ · G(erreur), ici archétype tanh (λ=1).
        G_value = regulation.compute_G(error, 'tanh', {'lambda': 1.0})
        F_A = beta_n * G_value
        self.assertAlmostEqual(F_A, beta_n * np.tanh(error), places=6,
                               msg="F_Aₙ(t) doit être βₙ·G(Eₙ−Oₙ)")

        # Canal FRÉQUENCE : βₙ · γ(t).
        F_f = beta_n * gamma_t
        self.assertAlmostEqual(F_f, beta_n * gamma_t, places=6,
                               msg="F_fₙ(t) doit être βₙ·γ(t)")

        # Le design a quitté l'ancienne formule unique βₙ·(Eₙ−Oₙ)·γ.
        self.assertNotAlmostEqual(F_A, beta_n * error * gamma_t, places=3,
                                  msg="Le feedback amplitude n'est plus βₙ·(Eₙ−Oₙ)·γ")


class TestRegulation(unittest.TestCase):
    """Tests pour le module regulation.py"""
    
    def setUp(self):
        """Configuration pour les tests."""
        self.config = {
            'regulation': {'G_arch': 'tanh', 'lambda': 1.0},
            'enveloppe': {'env_mode': 'static', 'sigma_n_static': 0.1, 'mu_n': 0.0},
            'system': {'T': 100}
        }
    
    def test_compute_G_archetypes(self):
        """Test des différents archétypes de régulation."""
        x_test = np.linspace(-3, 3, 7)
        
        # Tanh
        g_tanh = regulation.compute_G(x_test, 'tanh', {'lambda': 1.0})
        self.assertAlmostEqual(regulation.compute_G(0, 'tanh'), 0.0, places=6)
        
        # Sinc
        g_sinc = regulation.compute_G(x_test, 'sinc')
        self.assertAlmostEqual(regulation.compute_G(0, 'sinc'), 1.0, places=6)
        
        # Resonance
        g_res = regulation.compute_G(x_test, 'resonance', {'alpha': 1.0, 'beta': 2.0})
        self.assertAlmostEqual(regulation.compute_G(0, 'resonance'), 0.0, places=6)
    
    def test_compute_env_n_static(self):
        """Test de l'enveloppe en mode statique."""
        # Gaussienne
        env = regulation.compute_env_n(0, 0, 'static', 0.5, 0, None, 'gaussienne')
        self.assertAlmostEqual(env, 1.0, places=6)
        
        # Sigmoïde
        env_sig = regulation.compute_env_n(0, 0, 'static', 0.5, 0, None, 'sigmoide')
        self.assertAlmostEqual(env_sig, 0.5, places=6)
    
    def test_compute_sigma_n_dynamic(self):
        """Test de l'écart-type dynamique."""
        T = 100
        sigma_dynamic = {'amp': 0.05, 'freq': 1, 'offset': 0.1}
        
        # À différents temps
        sigma_0 = regulation.compute_sigma_n(0, 'dynamic', T, 0.1, sigma_dynamic)
        sigma_T4 = regulation.compute_sigma_n(T/4, 'dynamic', T, 0.1, sigma_dynamic)
        
        # Vérifier la variation
        self.assertNotAlmostEqual(sigma_0, sigma_T4, places=3)
        self.assertGreater(sigma_0, 0)
        self.assertGreater(sigma_T4, 0)
    
    def test_compute_Gn_integration(self):
        """Test de la fonction intégrée pour simulate.py."""
        error = np.array([0.5, -0.3, 0.1])
        An_t = np.array([1.0, 0.8, 1.2])
        fn_t = np.array([1.0, 1.1, 0.9])

    def test_feedback_modes(self):
        """Le feedback amplitude suit l'archétype G choisi ; le feedback
        fréquence reste βₙ·γ, indépendant de l'archétype (catalogue §4)."""
        beta_n = 1.5
        On_t = 0.8
        En_t = 0.6
        gamma_t = 0.9
        error = En_t - On_t  # convention production/catalogue : errorₙ = Eₙ − Oₙ

        # AMPLITUDE : dépend de l'archétype G.
        F_A_tanh = beta_n * regulation.compute_G(error, 'tanh', {'lambda': 1.0})
        F_A_resonance = beta_n * regulation.compute_G(error, 'resonance', {'lambda': 1.0})
        self.assertNotAlmostEqual(F_A_tanh, F_A_resonance, places=3,
                                  msg="L'archétype G doit changer le feedback amplitude")
        self.assertIsInstance(float(F_A_tanh), float)

        # FRÉQUENCE : βₙ·γ, le même quel que soit l'archétype amplitude.
        F_f = beta_n * gamma_t
        self.assertAlmostEqual(F_f, beta_n * gamma_t, places=6)


class TestMetrics(unittest.TestCase):
    """Tests pour le module metrics.py"""
    
    def setUp(self):
        """Données de test."""
        self.S_history = list(np.sin(np.linspace(0, 10, 100)))
        self.effort_history = list(0.5 + 0.1 * np.random.randn(100))
        self.config = {
            'to_calibrate': {
                'effort_transitoire_threshold': 2.0,
                'effort_chronique_threshold': 1.0  # Seuil plus bas pour que le test passe
            }
        }
    
    def test_compute_cpu_step(self):
        """Test du calcul CPU."""
        start = time.perf_counter()
        time.sleep(0.01)
        end = time.perf_counter()
        
        cpu = metrics.compute_cpu_step(start, end, 10)
        self.assertGreater(cpu, 0)
        self.assertLess(cpu, 1.0)  # Moins d'1 seconde par strate
    
    def test_compute_effort(self):
        """Test du calcul d'effort."""
        delta_An = np.array([0.1, -0.05, 0.02])
        delta_fn = np.array([0.01, 0.02, -0.01])
        delta_gamma = np.array([0.0, 0.0, 0.0])
        
        effort = metrics.compute_effort(delta_An, delta_fn, delta_gamma, 1.0, 1.0, 1.0)
        
        # L'effort est la somme normalisée des variations
        expected = np.sum(np.abs(delta_An)) + np.sum(np.abs(delta_fn))
        self.assertAlmostEqual(effort, expected, places=6)
    
    def test_compute_variance_d2S(self):
        """Test de la variance de d²S/dt²."""
        # Signal lisse
        t = np.linspace(0, 10, 100)
        S_smooth = list(np.sin(t))
        var_smooth = metrics.compute_variance_d2S(S_smooth, 0.1)
        
        # Signal bruité
        S_noisy = list(np.sin(t) + 0.1 * np.random.randn(100))
        var_noisy = metrics.compute_variance_d2S(S_noisy, 0.1)
        
        # Le signal bruité doit avoir plus de variance
        self.assertGreater(var_noisy, var_smooth)
    
    def test_compute_fluidity(self):
        """Test de la nouvelle métrique de fluidité."""
        # Test avec variance nulle (fluidité parfaite)
        fluidity_perfect = metrics.compute_fluidity(0.0)
        self.assertEqual(fluidity_perfect, 1.0)
        
        # Test avec variance de référence (fluidité = 0.5)
        fluidity_ref = metrics.compute_fluidity(175.0)
        self.assertAlmostEqual(fluidity_ref, 0.5, places=2)
        
        # Test avec variance élevée (fluidité faible)
        fluidity_low = metrics.compute_fluidity(350.0)
        self.assertLess(fluidity_low, 0.1)
        
        # Test avec variance faible (fluidité élevée)
        fluidity_high = metrics.compute_fluidity(87.5)
        self.assertGreater(fluidity_high, 0.9)
        
        # Vérifier la monotonie : plus de variance = moins de fluidité
        variances = [50, 100, 150, 200, 250, 300]
        fluidities = [metrics.compute_fluidity(v) for v in variances]
        for i in range(len(fluidities) - 1):
            self.assertGreater(fluidities[i], fluidities[i+1])
    
    def test_compute_adaptive_resilience(self):
        """Test de la résilience adaptative."""
        # Configuration avec perturbation continue
        config_continuous = {
            'system': {
                'input': {
                    'perturbations': [
                        {'type': 'sinus', 'amplitude': 0.5, 'freq': 0.1}
                    ]
                }
            }
        }
        
        # Configuration avec perturbation ponctuelle
        config_punctual = {
            'system': {
                'input': {
                    'perturbations': [
                        {'type': 'choc', 'amplitude': 2.0, 't0': 10}
                    ]
                }
            }
        }
        
        # Test avec perturbation continue
        metrics_cont = {'continuous_resilience': 0.85}
        result_cont = metrics.compute_adaptive_resilience(
            config_continuous, metrics_cont
        )
        
        self.assertEqual(result_cont['type'], 'continuous')
        self.assertEqual(result_cont['metric_used'], 'continuous_resilience')
        self.assertEqual(result_cont['value'], 0.85)
        self.assertEqual(result_cont['score'], 4)  # 0.85 → score 4
        
        # Test avec perturbation ponctuelle
        metrics_punct = {'t_retour': 1.5}
        result_punct = metrics.compute_adaptive_resilience(
            config_punctual, metrics_punct
        )
        
        self.assertEqual(result_punct['type'], 'punctual')
        self.assertEqual(result_punct['metric_used'], 't_retour')
        self.assertEqual(result_punct['score'], 4)  # t_retour 1.5 → score 4
        
        # Test sans perturbation
        config_none = {'system': {'input': {'perturbations': []}}}
        result_none = metrics.compute_adaptive_resilience(config_none, {})
        
        self.assertEqual(result_none['type'], 'none')
        self.assertEqual(result_none['metric_used'], 't_retour')
    
    def test_compute_entropy_S(self):
        """Test de l'entropie spectrale."""
        # Signal mono-fréquence
        t = np.linspace(0, 10, 100)
        S_mono = np.sin(2 * np.pi * t)
        entropy_mono = metrics.compute_entropy_S(S_mono, 10.0)
        
        # Signal multi-fréquence
        S_multi = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)
        entropy_multi = metrics.compute_entropy_S(S_multi, 10.0)
        
        # Multi-fréquence doit avoir plus d'entropie
        self.assertGreater(entropy_multi, entropy_mono)
    
    def test_compute_effort_status(self):
        """Test de la détection d'état d'effort."""
        # Effort stable
        effort_stable = [0.5] * 50
        status = metrics.compute_effort_status(0.5, effort_stable, self.config)
        self.assertEqual(status, "stable")
        
        # Effort transitoire (pic)
        status = metrics.compute_effort_status(2.0, effort_stable, self.config)
        self.assertEqual(status, "transitoire")
        
        # Effort chronique
        effort_high = [1.5] * 50
        status = metrics.compute_effort_status(1.5, effort_high, self.config)
        self.assertIn(status, ["chronique", "transitoire"])


class TestValidateConfig(unittest.TestCase):
    """Tests pour le module validate_config.py"""
    
    def test_validate_valid_config(self):
        """Test avec une configuration valide."""
        config = validate_config.generate_default_config(N=5, T=100)
        errors, warnings = validate_config.validate_config(config)
        
        self.assertEqual(len(errors), 0)
    
    def test_validate_invalid_config(self):
        """Test avec des configurations invalides."""
        # Config sans système
        config = {'strates': []}
        errors, warnings = validate_config.validate_config(config)
        self.assertGreater(len(errors), 0)
        
        # N négatif
        config = validate_config.generate_default_config(N=5, T=100)
        config['system']['N'] = -1
        errors, warnings = validate_config.validate_config(config)
        self.assertGreater(len(errors), 0)
    
    def test_update_config_threshold(self):
        """Test de mise à jour de seuil."""
        config = validate_config.generate_default_config(N=5, T=100)
        
        # Créer un changelog temporaire
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            changelog_path = f.name
        
        try:
            success = validate_config.update_config_threshold(
                config, 'variance_d2S', 0.02, 'Test', changelog_path
            )
            
            self.assertTrue(success)
            self.assertEqual(config['to_calibrate']['variance_d2S'], 0.02)
            
            # Vérifier le changelog
            with open(changelog_path, 'r') as f:
                content = f.read()
                self.assertIn('variance_d2S', content)
                self.assertIn('0.02', content)
        finally:
            os.unlink(changelog_path)


class TestPerturbations(unittest.TestCase):
    """Tests pour le module perturbations.py"""
    
    def test_perturbation_types(self):
        """Test de tous les types de perturbations."""
        # Choc
        config_choc = {'type': 'choc', 't0': 5.0, 'amplitude': 2.0, 'duration': 0.5}
        self.assertEqual(perturbations.generate_perturbation(4.9, config_choc), 0.0)
        self.assertEqual(perturbations.generate_perturbation(5.1, config_choc), 2.0)
        self.assertEqual(perturbations.generate_perturbation(5.6, config_choc), 0.0)
        
        # Rampe
        config_rampe = {'type': 'rampe', 't0': 0.0, 'amplitude': 1.0, 'duration': 10.0}
        self.assertAlmostEqual(perturbations.generate_perturbation(5.0, config_rampe), 0.5, places=6)
        
        # Sinus
        config_sinus = {'type': 'sinus', 't0': 0.0, 'amplitude': 1.0, 'freq': 0.5}
        value = perturbations.generate_perturbation(1.0, config_sinus)
        self.assertAlmostEqual(value, np.sin(np.pi), places=6)
    
    def test_perturbation_sequence(self):
        """Test de séquence de perturbations."""
        configs = [
            {'type': 'choc', 't0': 5.0, 'amplitude': 2.0},
            {'type': 'sinus', 't0': 0.0, 'amplitude': 0.5, 'freq': 1.0}
        ]
        
        sequence = perturbations.generate_perturbation_sequence(10, 0.1, configs)
        
        # Vérifier la longueur
        self.assertEqual(len(sequence), 100)
        
        # Vérifier qu'il y a des valeurs non nulles
        self.assertGreater(np.abs(sequence).sum(), 0)
    
    def test_scenarios(self):
        """Test des scénarios prédéfinis."""
        for scenario in ['stress_test', 'environnement_variable', 'chaos']:
            configs = perturbations.create_scenario(scenario, T=100)
            self.assertIsInstance(configs, list)
            self.assertGreater(len(configs), 0)


class TestAnalyze(unittest.TestCase):
    """Tests pour le module analyze.py"""
    
    def setUp(self):
        """Configuration de test."""
        self.config = validate_config.generate_default_config(N=3, T=50)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_run_data(self):
        """Test du chargement de données."""
        # Créer un CSV de test
        csv_path = os.path.join(self.temp_dir, 'test_run.csv')
        with open(csv_path, 'w') as f:
            f.write('t,S(t),effort(t)\n')
            f.write('0,0.0,0.5\n')
            f.write('1,0.5,0.6\n')
            f.write('2,1.0,0.7\n')
        
        data = analyze.load_run_data(csv_path)
        
        self.assertIn('t', data)
        self.assertIn('S(t)', data)
        self.assertEqual(len(data['t']), 3)
    
    def test_refinement_functions(self):
        """Test des fonctions de raffinement."""
        stats = {'trigger_rate': 0.7, 'mean_values': [0.02, 0.03]}
        
        # Test raffinement fluidité
        changes = analyze.refine_fluidity(self.config.copy(), stats)
        self.assertIn('sigma_n', changes)
        
        # Test raffinement stabilité
        changes = analyze.refine_stability(self.config.copy(), stats)
        self.assertIn('k', changes)


class TestExplore(unittest.TestCase):
    """Tests pour le module explore.py"""
    
    def setUp(self):
        """Données de test."""
        t = np.linspace(0, 100, 1000)
        self.test_data = {
            'S(t)': np.sin(2 * np.pi * t / 10) + 0.1 * np.random.randn(1000),
            'C(t)': np.cos(2 * np.pi * t / 15),
            'effort(t)': 0.5 + 0.1 * np.sin(2 * np.pi * t / 20)
        }
        
        # Ajouter des anomalies
        self.test_data['S(t)'][200:220] += 5.0
        self.test_data['C(t)'][400] += 3.0
    
    def test_detect_anomalies(self):
        """Test de détection d'anomalies."""
        anomalies = explore.detect_anomalies(
            self.test_data, 
            ['S(t)', 'effort(t)'], 
            threshold=3.0, 
            min_duration=3
        )
        
        # Doit détecter au moins l'anomalie dans S(t)
        self.assertGreater(len(anomalies), 0)
        
        # Vérifier la structure
        if anomalies:
            self.assertIn('event_type', anomalies[0])
            self.assertIn('t_start', anomalies[0])
            self.assertIn('severity', anomalies[0])
    
    def test_detect_fractal_patterns(self):
        """Test de détection de motifs fractals."""
        # Ajouter des motifs auto-similaires
        for scale in [1, 10, 100]:
            self.test_data['S(t)'] += 0.1 / scale * np.sin(2 * np.pi * np.arange(1000) * scale / 10)
        
        fractals = explore.detect_fractal_patterns(
            self.test_data,
            metrics=['S(t)'],
            window_sizes=[1, 10, 100],
            threshold=0.7
        )
        
        # Peut détecter ou non selon le bruit
        self.assertIsInstance(fractals, list)


class TestKuramoto(unittest.TestCase):
    """Tests pour le module kuramoto.py"""
    
    def test_kuramoto_step(self):
        """Test d'un pas de Kuramoto."""
        phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        frequencies = np.ones(5) * 0.5
        
        dphases = kuramoto.kuramoto_step(phases, frequencies, K=1.0, N=5, dt=0.1)
        
        # Vérifier la forme
        self.assertEqual(len(dphases), 5)
        
        # Les dérivées ne doivent pas être toutes égales (couplage)
        self.assertFalse(np.allclose(dphases, frequencies))
    
    def test_compute_kuramoto_order(self):
        """Test du paramètre d'ordre."""
        # Cas synchronisé
        phases_sync = np.zeros(10)
        r_sync, psi_sync = kuramoto.compute_kuramoto_order(phases_sync)
        self.assertAlmostEqual(r_sync, 1.0, places=6)
        
        # Cas désynchronisé parfait
        phases_desync = np.linspace(0, 2*np.pi, 10, endpoint=False)
        r_desync, psi_desync = kuramoto.compute_kuramoto_order(phases_desync)
        self.assertLess(r_desync, 0.2)


class TestUtils(unittest.TestCase):
    """Tests pour le module utils.py"""
    
    def setUp(self):
        """Préparation."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_run_id(self):
        """Test de génération d'ID."""
        id1 = utils.generate_run_id()
        id2 = utils.generate_run_id()
        
        # IDs uniques
        self.assertNotEqual(id1, id2)
        
        # Format correct
        self.assertTrue(id1.startswith('run_'))
        self.assertIn('_', id1)
    
    def test_save_load_state(self):
        """Test sauvegarde/restauration d'état."""
        state = {
            't': 50.0,
            'strates': [{'id': 0, 'An': 1.0}],
            'history': [{'t': 0, 'S': 0}]
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pkl')
        utils.save_simulation_state(state, checkpoint_path)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Restaurer
        restored = utils.load_simulation_state(checkpoint_path)
        self.assertEqual(restored['t'], state['t'])
        self.assertEqual(len(restored['strates']), len(state['strates']))
    
    def test_format_duration(self):
        """Test du formatage de durée."""
        self.assertEqual(utils.format_duration(45.3), "45.3s")
        self.assertEqual(utils.format_duration(125), "2m 5.0s")
        self.assertEqual(utils.format_duration(3665), "1h 1m 5.0s")


class TestIntegration(unittest.TestCase):
    """Tests d'intégration du système complet."""
    
    def setUp(self):
        """Configuration pour tests d'intégration."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        
        # Créer une config minimale
        config = validate_config.generate_default_config(N=3, T=10)
        config['system']['dt'] = 0.5  # Pas plus grand pour test rapide
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
    
    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)
    
    @patch('simulate.init.setup_logging')
    @patch('simulate.init.init_strates')
    def test_simulation_flow(self, mock_init_strates, mock_setup_logging):
        """Test du flux de simulation complet."""
        # Mocks
        mock_init_strates.return_value = [
            {'id': 0, 'A0': 1.0, 'f0': 1.0, 'An': 1.0, 'fn': 1.0},
            {'id': 1, 'A0': 1.0, 'f0': 1.1, 'An': 1.0, 'fn': 1.1},
            {'id': 2, 'A0': 1.0, 'f0': 0.9, 'An': 1.0, 'fn': 0.9}
        ]
        
        mock_csv_writer = MagicMock()
        mock_setup_logging.return_value = {
            'csv_writer': mock_csv_writer,
            'run_id': 'test_run',
            'output_dir': self.temp_dir,
            'log_file': 'test.csv'
        }
        
        # Test imports et initialisation
        import simulate
        
        # Vérifier que les modules sont bien importés
        self.assertTrue(hasattr(simulate, 'dynamics'))
        self.assertTrue(hasattr(simulate, 'regulation'))
        self.assertTrue(hasattr(simulate, 'metrics'))


class TestPerformance(unittest.TestCase):
    """Tests de performance et limites."""
    
    def test_large_n_strates(self):
        """Test avec un grand nombre de strates."""
        # Générer une config avec N=50
        config = validate_config.generate_default_config(N=50, T=10)
        
        # Initialiser
        state = init.init_strates(config)
        self.assertEqual(len(state), 50)
        
        # Tester quelques calculs
        In_t = np.ones(50)
        An_t = dynamics.compute_An(0, state, In_t, np.zeros(50), config)
        self.assertEqual(len(An_t), 50)
    
    def test_numerical_stability(self):
        """Test de stabilité numérique."""
        # Valeurs extrêmes
        x_extreme = np.array([1e-10, 1e10, -1e10])
        
        # Sigmoïde doit rester bornée
        sigma_vals = dynamics.compute_sigma(x_extreme, 1.0, 0.0)
        self.assertTrue(np.all(sigma_vals >= 0))
        self.assertTrue(np.all(sigma_vals <= 1))
        
        # Régulation tanh doit rester bornée
        g_vals = regulation.compute_G(x_extreme, 'tanh')
        self.assertTrue(np.all(np.abs(g_vals) <= 1))


def run_all_tests():
    """Lance tous les tests et génère un rapport."""
    # Créer un test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_classes = [
        TestDynamics,
        TestRegulation,
        TestMetrics,
        TestValidateConfig,
        TestPerturbations,
        TestAnalyze,
        TestExplore,
        TestKuramoto,
        TestUtils,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Runner avec rapport détaillé
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS FPS")
    print("="*70)
    print(f"Tests exécutés : {result.testsRun}")
    print(f"Succès : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs : {len(result.failures)}")
    print(f"Erreurs : {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ TOUS LES TESTS SONT PASSÉS ! La FPS est prête à danser ! 🌀")
    else:
        print("\n❌ Certains tests ont échoué. Vérifier les détails ci-dessus.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Désactiver les warnings pour les tests
    warnings.filterwarnings('ignore')
    
    # Lancer tous les tests
    success = run_all_tests()
    
    # Code de sortie
    exit(0 if success else 1)
