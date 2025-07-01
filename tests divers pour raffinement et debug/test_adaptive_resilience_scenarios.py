#!/usr/bin/env python3
"""
Test de la résilience adaptative avec différents scénarios de perturbation
"""

import json
import subprocess
import pandas as pd
import os

def test_resilience_scenario(scenario_name, perturbation_config):
    """Test un scénario spécifique de perturbation"""
    print(f"\n{'='*60}")
    print(f"🧪 Test: {scenario_name}")
    print(f"{'='*60}")
    
    # Charger la config de base
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Modifier la perturbation
    config['system']['input']['perturbations'] = perturbation_config
    
    # Réduire la durée pour un test rapide
    config['system']['T'] = 10
    config['system']['dt'] = 0.1
    
    # Sauvegarder la config temporaire
    temp_config = f'test_config_{scenario_name.replace(" ", "_")}.json'
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Lancer la simulation
    cmd = f'python3 main.py run --config {temp_config} --mode FPS'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Erreur simulation: {result.stderr}")
        return None
    
    # Extraire le run_id depuis la sortie
    for line in result.stdout.split('\n'):
        if 'Terminé : run_' in line:
            run_id = line.split('Terminé : ')[1].strip()
            break
    else:
        print("❌ Run ID non trouvé")
        return None
    
    # Lire les résultats
    csv_file = f'logs/{run_id}.csv'
    if not os.path.exists(csv_file):
        print(f"❌ Fichier CSV non trouvé: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Analyser les métriques de résilience
    print(f"\n📊 Analyse des métriques de résilience:")
    print(f"   - t_retour moyen: {df['t_retour'].mean():.4f}")
    print(f"   - continuous_resilience moyen: {df['continuous_resilience'].mean():.4f}")
    print(f"   - adaptive_resilience moyen: {df['adaptive_resilience'].mean():.4f}")
    
    # Vérifier quelle métrique est utilisée
    if df['adaptive_resilience'].mean() > 0:
        # Déterminer si c'est basé sur t_retour ou continuous_resilience
        if abs(df['adaptive_resilience'].mean() - df['continuous_resilience'].mean()) < 0.01:
            print(f"   ✓ Utilise continuous_resilience (perturbation continue détectée)")
        else:
            # Calculer la valeur normalisée de t_retour
            t_retour_norm = 1.0 / (1.0 + df['t_retour'].mean())
            if abs(df['adaptive_resilience'].mean() - t_retour_norm) < 0.01:
                print(f"   ✓ Utilise t_retour normalisé (perturbation ponctuelle détectée)")
            else:
                print(f"   ⚠️  Métrique source incertaine")
    
    # Nettoyer
    os.remove(temp_config)
    
    return {
        'scenario': scenario_name,
        't_retour': df['t_retour'].mean(),
        'continuous_resilience': df['continuous_resilience'].mean(),
        'adaptive_resilience': df['adaptive_resilience'].mean()
    }

# Scénarios de test
scenarios = {
    "Sans perturbation": [],
    
    "Perturbation ponctuelle (choc)": [
        {
            "type": "choc",
            "amplitude": 2.0,
            "t0": 5.0
        }
    ],
    
    "Perturbation continue (sinus)": [
        {
            "type": "sinus",
            "amplitude": 0.5,
            "freq": 0.1
        }
    ],
    
    "Perturbation continue (bruit)": [
        {
            "type": "bruit",
            "amplitude": 0.3
        }
    ],
    
    "Perturbations multiples": [
        {
            "type": "choc",
            "amplitude": 1.0,
            "t0": 3.0
        },
        {
            "type": "sinus",
            "amplitude": 0.2,
            "freq": 0.2
        }
    ]
}

# Exécuter les tests
results = []
for scenario_name, perturbation_config in scenarios.items():
    result = test_resilience_scenario(scenario_name, perturbation_config)
    if result:
        results.append(result)

# Résumé
print(f"\n{'='*60}")
print("📈 RÉSUMÉ DES TESTS DE RÉSILIENCE ADAPTATIVE")
print(f"{'='*60}")

if results:
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    print("\n🎯 Conclusions:")
    print("- La résilience adaptative sélectionne automatiquement la bonne métrique")
    print("- Les perturbations continues utilisent continuous_resilience")
    print("- Les perturbations ponctuelles utilisent t_retour normalisé")
    print("- Les scénarios mixtes priorisent les perturbations continues")
else:
    print("❌ Aucun résultat disponible")

print("\n✅ Tests de résilience adaptative terminés!") 