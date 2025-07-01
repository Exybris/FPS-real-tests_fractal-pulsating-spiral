#!/usr/bin/env python3
"""
Script pour vérifier la cohérence de la métrique Stabilité à travers le pipeline FPS
"""

import json
import csv
import numpy as np
import os

def check_stability_coherence():
    print("🔍 Vérification de la cohérence de Stabilité\n")
    
    # 1. Vérifier les métriques dans metrics.py
    print("1️⃣ MÉTRIQUES dans metrics.py:")
    print("   - compute_max_median_ratio: ratio max(|S|)/median(|S|)")
    print("   - Pas de fonction compute_std_S directe")
    print("   - std_S calculé dans simulate.py ligne 745: np.std(S_history)\n")
    
    # 2. Vérifier les seuils dans main.py
    print("2️⃣ SEUILS dans calculate_empirical_scores (main.py):")
    print("   - std_S < 0.5 → 5/5 (Très stable)")
    print("   - std_S < 1.0 → 4/5 (Stable)")
    print("   - std_S < 2.0 → 3/5 (Moyennement stable)")
    print("   - sinon → 2/5 (Peu stable)\n")
    
    # 3. Vérifier dans compare_modes.py
    print("3️⃣ CALCUL dans compare_modes.py:")
    print("   - stability = 1/(std_S + 1e-3)")
    print("   - Plus std_S est bas, plus stability est élevé\n")
    
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
            
            # Récupérer stability depuis detailed_metrics
            stability = data.get('detailed_metrics', {}).get('stability', {}).get('fps_value', None)
            
            if stability:
                # Calculer std_S depuis stability
                # stability = 1/(std_S + 1e-3), donc std_S = 1/stability - 1e-3
                std_S_calc = 1/stability - 1e-3
                print(f"   - stability (detailed_metrics): {stability}")
                print(f"   - std_S calculé: {std_S_calc:.6f}")
                
                # Déterminer le score selon les seuils
                if std_S_calc < 0.5:
                    score = 5
                    desc = "Très stable"
                elif std_S_calc < 1.0:
                    score = 4
                    desc = "Stable"
                elif std_S_calc < 2.0:
                    score = 3
                    desc = "Moyennement stable"
                else:
                    score = 2
                    desc = "Peu stable"
                
                print(f"   - Score Stabilité: {score}/5 ({desc})")
            
            # Vérifier si std_S est dans raw_metrics
            raw_fps = data.get('raw_metrics', {}).get('fps', {})
            if 'std_S' in raw_fps:
                print(f"   ✓ std_S dans raw_metrics: {raw_fps['std_S']}")
            else:
                print(f"   ⚠️ std_S NON TROUVÉ dans raw_metrics!")
                print(f"      → calculate_empirical_scores ne peut pas calculer le score correctement")
            
            # Vérifier stability_ratio (max_median_ratio)
            if 'stability_ratio' in raw_fps:
                print(f"   - stability_ratio (max_median_ratio): {raw_fps['stability_ratio']}")
        
        # Analyser un fichier batch CSV pour vérifier les valeurs réelles
        batch_csv = f"fps_pipeline_output/{run}/logs/batch_run_0_run_*_seed12345.csv"
        import glob
        csv_files = glob.glob(batch_csv)
        if csv_files:
            csv_file = csv_files[0]
            print(f"\n   Analyse du CSV: {os.path.basename(csv_file)}")
            
            # Lire les données S(t)
            S_values = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'S(t)' in row:
                        try:
                            S_values.append(float(row['S(t)']))
                        except:
                            pass
            
            if S_values:
                std_S_actual = np.std(S_values)
                print(f"   - std(S) calculé depuis CSV: {std_S_actual:.6f}")
                print(f"   - Nombre de points: {len(S_values)}")
        
        print()
    
    # 5. Diagnostic du problème
    print("5️⃣ DIAGNOSTIC:")
    print("   ❌ PROBLÈME IDENTIFIÉ: std_S n'est pas inclus dans raw_metrics")
    print("   → calculate_empirical_scores reçoit metrics.get('std_S', float('inf'))")
    print("   → Avec float('inf'), le score est toujours 2/5")
    print("   → La grille empirique affiche donc un score incorrect\n")
    
    # 6. Solution proposée
    print("6️⃣ SOLUTION PROPOSÉE:")
    print("   Dans simulate.py, ajouter std_S à metrics_summary:")
    print("   metrics_summary = {")
    print("       'mean_S': np.mean(S_history),")
    print("       'std_S': np.std(S_history),  # ← DÉJÀ PRÉSENT ligne 745")
    print("       ...")
    print("   }")
    print("\n   Le problème est que std_S est bien calculé mais n'est pas propagé")
    print("   jusqu'à calculate_empirical_scores via les métriques finales.")

if __name__ == "__main__":
    check_stability_coherence() 