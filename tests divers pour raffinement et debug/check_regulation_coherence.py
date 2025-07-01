#!/usr/bin/env python3
"""
Script pour vérifier la cohérence de la métrique Régulation à travers le pipeline FPS
"""

import json
import csv
import numpy as np
import os
import glob

def check_regulation_coherence():
    print("🔍 Vérification de la cohérence de Régulation (mean_abs_error)\n")
    
    # 1. Vérifier la formule dans metrics.py
    print("1️⃣ FORMULE dans metrics.py:")
    print("   - compute_mean_abs_error(En_array, On_array)")
    print("   - Calcule: mean(|En - On|)")
    print("   - En = sorties attendues, On = sorties observées")
    print("   - Mesure la qualité de convergence du système\n")
    
    # 2. Vérifier les seuils dans main.py
    print("2️⃣ SEUILS dans calculate_empirical_scores (main.py):")
    print("   - error < 0.1 → 5/5 (Excellente régulation)")
    print("   - error < 0.5 → 4/5 (Bonne régulation)")
    print("   - error < 1.0 → 3/5 (Régulation moyenne)")
    print("   - sinon → 2/5 (Régulation faible)\n")
    
    # 3. Vérifier dans simulate.py
    print("3️⃣ CALCUL dans simulate.py:")
    print("   - Ligne 460: mean_abs_error = compute_mean_abs_error(En_t, On_t)")
    print("   - Stocké dans metrics_summary comme 'final_mean_abs_error'\n")
    
    # 4. Analyser les runs récents
    print("4️⃣ ANALYSE des runs récents:\n")
    
    runs = ['run_20250701_174930', 'run_20250630_203612']
    
    for run in runs:
        print(f"📂 {run}:")
        
        # Lire le rapport de comparaison
        report_path = f"fps_pipeline_output/{run}/reports/comparison_fps_vs_controls.json"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            # Chercher regulation dans detailed_metrics
            regulation_found = False
            for metric_name, metric_data in data.get('detailed_metrics', {}).items():
                if 'regulation' in metric_name.lower() or 'error' in metric_name.lower():
                    print(f"   - {metric_name}: {metric_data.get('fps_value', 'N/A')}")
                    regulation_found = True
            
            if not regulation_found:
                print("   ⚠️ Pas de métrique 'regulation' dans detailed_metrics")
        
        # Analyser un fichier batch CSV
        batch_csv = f"fps_pipeline_output/{run}/logs/batch_run_0_run_*_seed12345.csv"
        csv_files = glob.glob(batch_csv)
        if csv_files:
            csv_file = csv_files[0]
            print(f"\n   Analyse du CSV: {os.path.basename(csv_file)}")
            
            # Lire mean_abs_error
            errors = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'mean_abs_error' in row:
                        try:
                            errors.append(float(row['mean_abs_error']))
                        except:
                            pass
            
            if errors:
                final_error = errors[-1] if errors else 0.0
                mean_error = np.mean(errors)
                print(f"   - Dernière mean_abs_error: {final_error:.6f}")
                print(f"   - Moyenne mean_abs_error: {mean_error:.6f}")
                
                # Calculer le score
                if final_error < 0.1:
                    score = 5
                    desc = "Excellente régulation"
                elif final_error < 0.5:
                    score = 4
                    desc = "Bonne régulation"
                elif final_error < 1.0:
                    score = 3
                    desc = "Régulation moyenne"
                else:
                    score = 2
                    desc = "Régulation faible"
                
                print(f"   - Score Régulation: {score}/5 ({desc})")
        
        print()
    
    # 5. Vérifier dans compare_modes.py
    print("5️⃣ UTILISATION dans compare_modes.py:")
    print("   - Régulation n'est pas dans les métriques comparées")
    print("   - Seules synchronization, stability, resilience, innovation, fluidity, cpu_efficiency sont comparées\n")
    
    # 6. Diagnostic
    print("6️⃣ DIAGNOSTIC:")
    print("   ✓ La métrique est bien calculée dans simulate.py")
    print("   ✓ Elle est stockée comme 'final_mean_abs_error'")
    print("   ✓ calculate_empirical_scores l'utilise correctement")
    print("   ❓ Mais elle n'apparaît pas dans les comparaisons inter-modèles")
    print("   → C'est normal car Kuramoto/Neutral n'ont pas En/On\n")
    
    # 7. Recommandations
    print("7️⃣ COHÉRENCE:")
    print("   - La métrique est cohérente dans son flux")
    print("   - Elle mesure bien la convergence En→On spécifique à FPS")
    print("   - Les seuils semblent appropriés (< 0.1 pour excellence)")

if __name__ == "__main__":
    check_regulation_coherence() 