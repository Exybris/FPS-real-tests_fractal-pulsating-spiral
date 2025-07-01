#!/usr/bin/env python3
"""
Script pour vérifier la cohérence de la métrique Coût CPU à travers le pipeline FPS
"""

import json
import csv
import numpy as np
import os
import glob

def check_cpu_coherence():
    print("🔍 Vérification de la cohérence de Coût CPU\n")
    
    # 1. Vérifier les formules dans metrics.py
    print("1️⃣ FORMULES dans metrics.py:")
    print("   A) compute_cpu_step(start, end, N):")
    print("      - Calcule: (end_time - start_time) / N")
    print("      - Temps CPU normalisé par strate")
    print("   B) compute_correlation_effort_cpu(effort_history, cpu_history):")
    print("      - Calcule la corrélation de Pearson entre effort et CPU")
    print("      - Permet de voir si effort → charge computationnelle\n")
    
    # 2. Vérifier les seuils dans main.py
    print("2️⃣ SEUILS dans calculate_empirical_scores (main.py):")
    print("   - cpu < 0.001 → 5/5 (Très efficace)")
    print("   - cpu < 0.01 → 4/5 (Efficace)")
    print("   - cpu < 0.1 → 3/5 (Moyen)")
    print("   - sinon → 2/5 (Lent)\n")
    
    # 3. Vérifier dans simulate.py
    print("3️⃣ CALCUL dans simulate.py:")
    print("   - Ligne 335: cpu_step = compute_cpu_step(core_start, time.perf_counter(), N)")
    print("   - Stocké dans metrics_summary comme 'mean_cpu_step' (moyenne sur tout le run)\n")
    
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
            
            # Chercher cpu_efficiency dans detailed_metrics
            cpu_eff = data.get('detailed_metrics', {}).get('cpu_efficiency', {})
            if cpu_eff:
                fps_cpu = cpu_eff.get('fps_value', 'N/A')
                kura_cpu = cpu_eff.get('kuramoto_value', 'N/A')
                neutral_cpu = cpu_eff.get('neutral_value', 'N/A')
                print(f"   - cpu_efficiency FPS: {fps_cpu}")
                print(f"   - cpu_efficiency Kuramoto: {kura_cpu}")
                print(f"   - cpu_efficiency Neutral: {neutral_cpu}")
                print(f"   - Note: Ces valeurs semblent être des efficiences (1/cpu), pas des temps!")
        
        # Analyser un fichier batch CSV
        batch_csv = f"fps_pipeline_output/{run}/logs/batch_run_0_run_*_seed12345.csv"
        csv_files = glob.glob(batch_csv)
        if csv_files:
            csv_file = csv_files[0]
            print(f"\n   Analyse du CSV: {os.path.basename(csv_file)}")
            
            # Lire cpu_step(t)
            cpu_steps = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'cpu_step(t)' in row:
                        try:
                            cpu_steps.append(float(row['cpu_step(t)']))
                        except:
                            pass
            
            if cpu_steps:
                mean_cpu = np.mean(cpu_steps)
                print(f"   - Moyenne cpu_step: {mean_cpu:.6f} secondes")
                print(f"   - Min cpu_step: {min(cpu_steps):.6f}")
                print(f"   - Max cpu_step: {max(cpu_steps):.6f}")
                
                # Calculer le score
                if mean_cpu < 0.001:
                    score = 5
                    desc = "Très efficace"
                elif mean_cpu < 0.01:
                    score = 4
                    desc = "Efficace"
                elif mean_cpu < 0.1:
                    score = 3
                    desc = "Moyen"
                else:
                    score = 2
                    desc = "Lent"
                
                print(f"   - Score Coût CPU: {score}/5 ({desc})")
                
                # Calculer l'efficience pour comparaison
                if mean_cpu > 0:
                    efficiency = 1 / mean_cpu
                    print(f"   - Efficience calculée: {efficiency:.2f}")
        
        print()
    
    # 5. Vérifier dans compare_modes.py
    print("5️⃣ UTILISATION dans compare_modes.py:")
    print("   - 'cpu_efficiency' est comparée entre les modèles")
    print("   - ATTENTION: C'est l'efficience (1/cpu), pas le temps CPU!")
    print("   - Plus l'efficience est haute, mieux c'est\n")
    
    # 6. Diagnostic
    print("6️⃣ DIAGNOSTIC:")
    print("   ⚠️ INCOHÉRENCE DÉTECTÉE:")
    print("   - calculate_empirical_scores utilise mean_cpu_step (temps en secondes)")
    print("   - compare_modes utilise cpu_efficiency (1/temps)")
    print("   - Les valeurs dans le rapport JSON sont des efficiences (~45000)")
    print("   - Mais les seuils dans main.py sont pour des temps (<0.001s)\n")
    
    # 7. Recommandations
    print("7️⃣ RECOMMANDATIONS:")
    print("   1. Clarifier si on veut afficher temps ou efficience")
    print("   2. Si temps: garder mean_cpu_step partout")
    print("   3. Si efficience: adapter les seuils dans calculate_empirical_scores")
    print("   4. Documenter clairement l'unité (secondes vs 1/secondes)")

if __name__ == "__main__":
    check_cpu_coherence() 