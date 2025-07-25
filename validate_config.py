import json
import os
from datetime import datetime

# 1. — METRIQUES VALIDES —
# 1. — METRIQUES VALIDES —
METRIQUES_VALIDES = {
    "t", "S(t)", "A_mean(t)", "f_mean(t)", "effort(t)", "cpu_step(t)",
    "C(t)", "E(t)", "L(t)", "variance_d2S", "fluidity", "entropy_S", "effort_status",
    "mean_abs_error", "mean_high_effort", "d_effort_dt", "t_retour",
    "max_median_ratio", "A_spiral(t)", "continuous_resilience", "adaptive_resilience",
    "En_mean(t)", "On_mean(t)", "gamma", "gamma_mean(t)", "In_mean(t)",
    "An_mean(t)", "fn_mean(t)", "gamma_regime", "G_arch_used"
}

CRITERES_VALIDES = {
    "fluidity", "stability", "resilience", "innovation",
    "regulation", "cpu_cost", "effort_internal", "effort_transient"
}

# NOTE FPS - Seuils initiaux théoriques
# Ces seuils sont basés sur la théorie FPS et doivent être ajustés
# après les 5 premiers runs de calibration (N=5, T=20, In(t)~U[0,1])
SEUILS_THEORIQUES_INITIAUX = {
    "variance_d2S": 0.01,        # Fluidité : variance de d²S/dt²
    "fluidity_threshold": 0.3,   # Fluidité : seuil minimum (0=saccadé, 1=parfait)
    "stability_ratio": 10,        # Stabilité : max(S(t))/median(S(t))
    "resilience": 2,             # Résilience : t_retour / médiane
    "entropy_S": 0.5,            # Innovation : entropie spectrale
    "mean_high_effort": 2,       # Effort chronique : moyenne haute
    "d_effort_dt": 5,            # Effort transitoire : dérivée (en σ)
    "t_retour": 2,               # Temps retour équilibre (× médiane)
    "gamma_n": 1.0,              # Latence par strate
    "env_n": "gaussienne",       # Type enveloppe
    "sigma_n": 0.1,              # Écart-type enveloppe
    "cpu_step_ctrl": 2,          # Coût CPU vs contrôle
    "max_chaos_events": 5        # Nombre max événements chaotiques
}

# 2. — ERROR COLLECTOR —
class ValidationErrorCollector:
    def __init__(self):
        self.errors = []
        self.warnings = []
    def add_error(self, msg): self.errors.append(msg)
    def add_warning(self, msg): self.warnings.append(msg)
    def has_errors(self): return len(self.errors) > 0
    def report(self):
        print("\n[CONFIG VALIDATION]")
        if self.errors:
            print("\nErreurs :")
            for e in self.errors:
                print(f" - {e}")
        else:
            print("Aucune erreur.")
        if self.warnings:
            print("\nAvertissements :")
            for w in self.warnings:
                print(f" - {w}")

# 3. — VALIDATE SECTIONS —

def validate_main_blocks(config, collector):
    BLOCS_REQUIS = [
        "system", "strates", "spiral", "regulation", "latence", "enveloppe",
        "exploration", "to_calibrate", "validation", "analysis", "dynamic_parameters"
    ]
    for b in BLOCS_REQUIS:
        if b not in config:
            collector.add_error(f"Bloc manquant : {b}")

def validate_system(system, collector):
    N = system.get("N")
    T = system.get("T")
    dt = system.get("dt")
    seed = system.get("seed")
    mode = system.get("mode")
    logging = system.get("logging", {})
    input_cfg = system.get("input", {})

    if not is_int(N) or N <= 0:
        collector.add_error("system.N doit être un entier > 0")
    if not is_int(T) or T <= 0:
        collector.add_error("system.T doit être un entier > 0")
    if not is_float(dt) or dt <= 0:
        collector.add_error("system.dt doit être un float > 0")
    if not is_int(seed):
        collector.add_error("system.seed doit être un entier")
    if mode not in ["FPS", "Kuramoto", "neutral"]:
        collector.add_error("system.mode doit être 'FPS', 'Kuramoto' ou 'neutral'")

    # Logging
    level = logging.get("level")
    output = logging.get("output")
    log_metrics = logging.get("log_metrics", [])
    if level not in ["INFO", "DEBUG", "WARNING"]:
        collector.add_error("system.logging.level doit être 'INFO', 'DEBUG' ou 'WARNING'")
    if output not in ["csv", "hdf5"]:
        collector.add_error("system.logging.output doit être 'csv' ou 'hdf5'")
    if not (isinstance(log_metrics, list) and len(log_metrics) > 0):
        collector.add_error("system.logging.log_metrics doit être une liste non vide")
    else:
        for m in log_metrics:
            if not check_metric(m):
                collector.add_error(f"system.logging.log_metrics contient une métrique invalide : {m}")

    # Input (nouvelle architecture)
    baseline = input_cfg.get("baseline", {})
    perturbations = input_cfg.get("perturbations", [])
    
    # Validation baseline
    if baseline:
        offset_mode = baseline.get("offset_mode", "static")
        if offset_mode not in ["static", "adaptive"]:
            collector.add_error("system.input.baseline.offset_mode doit être 'static' ou 'adaptive'")
        if offset_mode == "static" and baseline.get("offset", 0) <= 0:
            collector.add_error("system.input.baseline.offset doit être > 0")
        
        gain_mode = baseline.get("gain_mode", "static")
        if gain_mode not in ["static", "adaptive"]:
            collector.add_error("system.input.baseline.gain_mode doit être 'static' ou 'adaptive'")
        if gain_mode == "static" and baseline.get("gain", 0) <= 0:
            collector.add_error("system.input.baseline.gain doit être > 0")
    
    # Validation perturbations
    if perturbations:
        for i, pert in enumerate(perturbations):
            pert_type = pert.get("type")
            if pert_type not in ["choc", "rampe", "sinus", "bruit", "none"]:
                collector.add_error(f"system.input.perturbations[{i}].type doit être 'choc', 'rampe', 'sinus', 'bruit' ou 'none'")
            if "t0" in pert and (not (is_float(pert["t0"]) or is_int(pert["t0"])) or pert["t0"] < 0):
                collector.add_error(f"system.input.perturbations[{i}].t0 doit être >= 0")
            if "amplitude" not in pert:
                collector.add_error(f"system.input.perturbations[{i}].amplitude doit être défini")
            if pert_type == "sinus" and "freq" not in pert:
                collector.add_error(f"system.input.perturbations[{i}].freq requis si type='sinus'")

def validate_strates(strates, N, collector, coupling_type=None):
    if not isinstance(strates, list):
        collector.add_error("strates doit être une liste")
        return
    if N is None:
        collector.add_error("Impossible de valider strates : N non défini")
        return
    if len(strates) != N:
        collector.add_error(f"strates doit contenir exactement N={N} éléments (actuellement {len(strates)})")
    for i, s in enumerate(strates):
        if s.get("A0", -1) <= 0:
            collector.add_error(f"strate[{i}].A0 doit être > 0")
        if s.get("f0", -1) <= 0:
            collector.add_error(f"strate[{i}].f0 doit être > 0")
        if "phi" not in s:
            collector.add_error(f"strate[{i}].phi doit être défini")
        if s.get("alpha", -1) < 0:
            collector.add_error(f"strate[{i}].alpha doit être >= 0")
        if s.get("beta", -1) < 0:
            collector.add_error(f"strate[{i}].beta doit être >= 0")
        if s.get("k", -1) <= 0:
            collector.add_error(f"strate[{i}].k doit être > 0")
        if "x0" not in s:
            collector.add_error(f"strate[{i}].x0 doit être défini")
        w = s.get("w")
        if coupling_type in {"spiral", "ring"} and w is None:
            # Les poids seront générés dynamiquement : on ne valide pas ici
            w = [0.0]*N
        if not isinstance(w, list) or len(w) != N:
            collector.add_error(f"strate[{i}].w doit être une liste de taille N={N}")
        # Optionnel : check diagonale nulle, et sum(w[i]) == 0
        if isinstance(w, list) and len(w) == N:
            if coupling_type not in {"spiral", "ring"} and abs(w[i]) > 1e-8:
                collector.add_error(f"strate[{i}].w[{i}] (diagonale) doit être 0")
            if coupling_type not in {"spiral", "ring"} and abs(sum(w)) > 1e-8:
                collector.add_warning(f"strate[{i}].w : la somme n'est pas exactement 0 (somme={sum(w):.5g}) — vérifier le couplage")
        # Optionnel : checks booléens
        for boolkey in ("dynamic_phi", "dynamic_alpha", "dynamic_beta"):
            if boolkey in s and not is_bool(s[boolkey]):
                collector.add_error(f"strate[{i}].{boolkey} doit être booléen s'il est présent")

def validate_spiral(spiral, collector):
    phi = spiral.get("phi")
    epsilon = spiral.get("epsilon")
    omega = spiral.get("omega")
    theta = spiral.get("theta")
    if abs(phi - 1.618) > 0.001:
        collector.add_error(f"spiral.phi doit être ≈ 1.618 (φ, tolérance ±0.001)")
    if epsilon is None or epsilon < 0:
        collector.add_error("spiral.epsilon doit être >= 0")
    if omega is None or omega <= 0:
        collector.add_error("spiral.omega doit être > 0")
    if theta is None:
        collector.add_error("spiral.theta doit être défini")

def validate_regulation(regulation, collector):
    G_arch = regulation.get("G_arch")
    dynamic_G = regulation.get("dynamic_G")
    if G_arch not in ["tanh", "sinc", "resonance", "spiral_log", "adaptive", "adaptive_aware"]:
        collector.add_error("regulation.G_arch doit être parmi ['tanh', 'sinc', 'resonance', 'spiral_log', 'adaptive', 'adaptive_aware']")
    if G_arch == "tanh":
        if regulation.get("lambda", 0) <= 0:
            collector.add_error("regulation.lambda doit être > 0 si G_arch == 'tanh'")
    if G_arch == "resonance":
        if regulation.get("alpha", 0) <= 0 or regulation.get("beta", 0) <= 0:
            collector.add_error("regulation.alpha et beta doivent être > 0 si G_arch == 'resonance'")
    if not is_bool(dynamic_G):
        collector.add_error("regulation.dynamic_G doit être booléen")

def validate_latence(latence, collector):
    gamma_mode = latence.get("gamma_mode")
    gamma_static_value = latence.get("gamma_static_value")
    gamma_dynamic = latence.get("gamma_dynamic", {})
    strata_delay = latence.get("strata_delay", False)
    
    # Définir les modes valides au début
    valid_gamma_modes = ["static", "dynamic", "sigmoid_up", "sigmoid_down", "sigmoid_adaptive", "sigmoid_oscillating", "sinusoidal", "adaptive_aware"]

    if gamma_mode not in valid_gamma_modes:
        collector.add_error(f"latence.gamma_mode doit être l'un de: {', '.join(valid_gamma_modes)}")
    if gamma_mode == "static" and (gamma_static_value is None or gamma_static_value <= 0):
        collector.add_error("latence.gamma_static_value doit être > 0 si gamma_mode == 'static'")
    if gamma_mode == "dynamic":
        if gamma_dynamic.get("k", 0) <= 0 or gamma_dynamic.get("t0", -1) < 0:
            collector.add_error("latence.gamma_dynamic (k>0, t0>=0) doit être défini si gamma_mode=='dynamic'")
    
    # Vérifier strata_delay
    if not isinstance(strata_delay, bool):
        collector.add_error("latence.strata_delay doit être un booléen")

def validate_enveloppe(enveloppe, collector):
    env_mode = enveloppe.get("env_mode")
    mu_n = enveloppe.get("mu_n")
    sigma_n_static = enveloppe.get("sigma_n_static")
    sigma_n_dynamic = enveloppe.get("sigma_n_dynamic", {})
    if env_mode not in ["static", "dynamic"]:
        collector.add_error("enveloppe.env_mode doit être 'static' ou 'dynamic'")
    if mu_n is None:
        collector.add_error("enveloppe.mu_n doit être défini")
    if sigma_n_static is None or sigma_n_static <= 0:
        collector.add_error("enveloppe.sigma_n_static doit être > 0")
    if env_mode == "dynamic":
        for p in ("amp", "freq", "offset", "T"):
            v = sigma_n_dynamic.get(p)
            if v is None or v <= 0:
                collector.add_error(f"enveloppe.sigma_n_dynamic.{p} doit être > 0 si env_mode=='dynamic'")

def validate_exploration(exploration, collector):
    metrics = exploration.get("metrics", [])
    window_sizes = exploration.get("window_sizes", [])
    fractal_threshold = exploration.get("fractal_threshold")
    anomaly_threshold = exploration.get("anomaly_threshold")
    min_duration = exploration.get("min_duration")
    for boolkey in ["detect_fractal_patterns", "detect_anomalies", "detect_harmonics"]:
        if boolkey in exploration and not is_bool(exploration[boolkey]):
            collector.add_error(f"exploration.{boolkey} doit être booléen")
    if not (isinstance(metrics, list) and len(metrics) > 0):
        collector.add_error("exploration.metrics doit être une liste non vide")
    else:
        for m in metrics:
            if not check_metric(m):
                collector.add_error(f"exploration.metrics contient une métrique invalide : {m}")
    if not (isinstance(window_sizes, list) and all(is_int(x) and x > 0 for x in window_sizes)):
        collector.add_error("exploration.window_sizes doit être une liste d'entiers > 0")
    if not (is_float(fractal_threshold) and 0 < fractal_threshold < 1):
        collector.add_error("exploration.fractal_threshold doit être dans ]0,1[")
    if not (is_float(anomaly_threshold) and anomaly_threshold > 0):
        collector.add_error("exploration.anomaly_threshold doit être > 0")
    if not (is_int(min_duration) and min_duration > 0):
        collector.add_error("exploration.min_duration doit être > 0")
    if exploration.get("detect_fractal_patterns"):
        print("Fractal motif detection config: OK")

def validate_to_calibrate(tc, collector):
    # Afficher la note sur les seuils théoriques initiaux
    print("\n[NOTE FPS] Seuils théoriques initiaux définis - À ajuster après 5 runs de calibration")
    
    if tc.get("variance_d2S", 0) <= 0:
        collector.add_error("to_calibrate.variance_d2S doit être > 0")
    # Validation du nouveau seuil de fluidité
    fluidity_threshold = tc.get("fluidity_threshold")
    if fluidity_threshold is not None:
        if not (is_float(fluidity_threshold) and 0 < fluidity_threshold < 1):
            collector.add_error("to_calibrate.fluidity_threshold doit être dans ]0,1[")
    if tc.get("stability_ratio", 0) <= 1:
        collector.add_error("to_calibrate.stability_ratio doit être > 1")
    if tc.get("resilience", 0) <= 0:
        collector.add_error("to_calibrate.resilience doit être > 0")
    es = tc.get("entropy_S")
    if not (is_float(es) and 0 < es < 1):
        collector.add_error("to_calibrate.entropy_S doit être dans ]0,1[")
    if tc.get("mean_high_effort", 0) <= 1:
        collector.add_error("to_calibrate.mean_high_effort doit être > 1")
    if tc.get("d_effort_dt", 0) <= 0:
        collector.add_error("to_calibrate.d_effort_dt doit être > 0")
    if tc.get("t_retour", 0) <= 0:
        collector.add_error("to_calibrate.t_retour doit être > 0")
    if tc.get("gamma_n", 0) <= 0:
        collector.add_error("to_calibrate.gamma_n doit être > 0")
    if tc.get("env_n") not in ["gaussienne", "sigmoide"]:
        collector.add_error("to_calibrate.env_n doit être 'gaussienne' ou 'sigmoide'")
    if tc.get("sigma_n", 0) <= 0:
        collector.add_error("to_calibrate.sigma_n doit être > 0")
    if tc.get("cpu_step_ctrl", 0) <= 1:
        collector.add_error("to_calibrate.cpu_step_ctrl doit être > 1")
    if tc.get("max_chaos_events", -1) < 0:
        collector.add_error("to_calibrate.max_chaos_events doit être >= 0")

def validate_validation(validation, collector):
    criteria = validation.get("criteria", [])
    if not (isinstance(criteria, list) and len(criteria) > 0):
        collector.add_error("validation.criteria doit être une liste non vide")
    else:
        for c in criteria:
            if c not in CRITERES_VALIDES:
                collector.add_error(f"validation.criteria contient un critère invalide : {c}")
    if validation.get("alert_sigma", 0) <= 0:
        collector.add_error("validation.alert_sigma doit être > 0")
    if validation.get("batch_size", 0) <= 0:
        collector.add_error("validation.batch_size doit être > 0")
    for boolkey in ["refine_after_runs", "auto_log_refinement"]:
        if boolkey in validation and not is_bool(validation[boolkey]):
            collector.add_error(f"validation.{boolkey} doit être booléen")

def validate_analysis(analysis, collector):
    for boolkey in ["compare_kuramoto", "save_indiv_files", "export_html_report", "visualize_grid"]:
        if boolkey in analysis and not is_bool(analysis[boolkey]):
            collector.add_error(f"analysis.{boolkey} doit être booléen")

def validate_dynamic_parameters(dynamic_parameters, collector):
    for boolkey in ["dynamic_phi", "dynamic_alpha", "dynamic_beta"]:
        if boolkey in dynamic_parameters and not is_bool(dynamic_parameters[boolkey]):
            collector.add_error(f"dynamic_parameters.{boolkey} doit être booléen")

def validate_cross_checks(config, collector):
    # — Cross-section checks selon la checklist —
    N = config.get("system", {}).get("N")
    T = config.get("system", {}).get("T")
    if N is not None and N > 10:
        analysis = config.get("analysis", {})
        if not analysis.get("save_indiv_files", False):
            collector.add_warning("N > 10 : il est recommandé de mettre analysis.save_indiv_files = True")
    if T is not None and T > 1000:
        collector.add_warning("T > 1000 : envisager la compression des logs pour éviter les fichiers massifs.")
    # Vérifier cohérence poids des strates
    strates = config.get("strates", [])
    coupling_type = str(config.get("coupling", {}).get("type", "")).lower()
    for i, s in enumerate(strates):
        w = s.get("w")
        if coupling_type in {"spiral", "ring"}:
            continue  # Les poids seront générés dynamiquement
        if isinstance(w, list) and N is not None and len(w) != N:
            collector.add_error(f"strate[{i}].w doit avoir exactement N={N} éléments")

# -------------------------------------------------------------------------
#  NEW  — VALIDATION DU BLOC COUPLING
# -------------------------------------------------------------------------

def validate_coupling(coupling, collector):
    """Validate optional 'coupling' block (spiral / ring)."""
    if not coupling:
        return  # Block is optional

    ctype = str(coupling.get("type", "")).lower()
    if ctype not in {"spiral", "ring"}:
        collector.add_error("coupling.type doit être 'spiral' ou 'ring' si présent")

    c_val = coupling.get("c")
    if c_val is None or (not is_float(c_val) and not is_int(c_val)) or c_val <= 0:
        collector.add_error("coupling.c doit être un nombre > 0")

    for boolkey in ["closed", "mirror"]:
        if boolkey in coupling and not is_bool(coupling[boolkey]):
            collector.add_error(f"coupling.{boolkey} doit être booléen s'il est présent")

    # closed est forcé à True si type == ring (pas une erreur mais on prévient)
    if ctype == "ring" and coupling.get("closed") is False:
        collector.add_warning("coupling.closed ignoré (forcé à True pour type='ring')")

# 4. — UTILS —
def is_bool(val):
    return isinstance(val, bool)

def is_int(val):
    return isinstance(val, int) and not isinstance(val, bool)

def is_float(val):
    return isinstance(val, float) or isinstance(val, int)

def check_metric(metric):
    """Vérifie qu'une métrique est dans la liste des métriques valides."""
    return metric in METRIQUES_VALIDES

# 5. — NOUVELLES FONCTIONS —

def generate_default_config(N=5, T=100):
    """
    Génère une configuration par défaut pour N strates et T pas de temps.
    Utilise les seuils théoriques initiaux définis dans SEUILS_THEORIQUES_INITIAUX.
    
    NOTE FPS : Cette config est une base de départ pour la phase 1.
    Tous les paramètres sont ajustables et doivent être raffinés selon les runs.
    """
    # Générer les strates avec poids couplés
    strates = []
    for i in range(N):
        # Matrice de poids : diagonale nulle, somme nulle
        w = []
        for j in range(N):
            if i == j:
                w.append(0.0)  # Diagonale nulle
            else:
                w.append(0.1 if j < N-1 else -0.1*(N-2))  # Somme nulle
        
        strate = {
            "A0": 1.0,
            "f0": 1.0 + i*0.1,  # Légère variation de fréquence
            "phi": 0.0,
            "alpha": 0.5,
            "beta": 1.0,
            "k": 2.0,
            "x0": 0.5,
            "w": w
        }
        strates.append(strate)
    
    config = {
        "system": {
            "N": N,
            "T": T,
            "dt": 0.05,
            "seed": 12345,
            "mode": "FPS",
            "logging": {
                "level": "INFO",
                "output": "csv",
                "log_metrics": ["t", "S(t)", "C(t)", "E(t)", "L(t)", "effort(t)", "cpu_step(t)"]
            },
            "input": {
                "baseline": {},
                "perturbations": []
            }
        },
        "strates": strates,
        "dynamic_parameters": {
            "dynamic_phi": False,
            "dynamic_alpha": False,
            "dynamic_beta": False
        },
        "spiral": {
            "phi": 1.618,
            "epsilon": 0.05,
            "omega": 0.1,
            "theta": 0.0
        },
        "regulation": {
            "G_arch": "tanh",
            "lambda": 1.0,
            "dynamic_G": False
        },
        "latence": {
            "gamma_mode": "static",
            "gamma_static_value": 1.0,
            "gamma_dynamic": {"k": 2.0, "t0": T//2},
            "strata_delay": False
        },
        "enveloppe": {
            "env_mode": "static",
            "mu_n": 0.0,
            "sigma_n_static": 0.1,
            "sigma_n_dynamic": {"amp": 0.05, "freq": 1, "offset": 0.1, "T": T}
        },
        "exploration": {
            "metrics": ["S(t)", "C(t)", "effort(t)"],
            "window_sizes": [1, 10, 100],
            "fractal_threshold": 0.8,
            "detect_fractal_patterns": True,
            "detect_anomalies": True,
            "detect_harmonics": True,
            "anomaly_threshold": 3.0,
            "min_duration": 3
        },
        "to_calibrate": SEUILS_THEORIQUES_INITIAUX.copy(),
        "validation": {
            "criteria": list(CRITERES_VALIDES),
            "alert_sigma": 3,
            "batch_size": 5,
            "refine_after_runs": True,
            "auto_log_refinement": True
        },
        "analysis": {
            "compare_kuramoto": True,
            "save_indiv_files": N > 10,
            "export_html_report": True,
            "visualize_grid": True
        },
        "adaptive_windows": {
            "exploration": {
                "target_percent": 0.8,
                "min_absolute": 10,
                "max_percent": 0.95
            },
            "gamma_adaptation": {
                "target_percent": 0.7,
                "min_absolute": 5,
                "max_percent": 0.85
            },
            "G_effectiveness": {
                "target_percent": 0.9,
                "min_absolute": 10,
                "max_percent": 0.98
            },
            "scoring": {
                "immediate": {
                    "target_percent": 0.95,
                    "min_absolute": 5,
                    "max_percent": 0.99
                },
                "recent": {
                    "target_percent": 0.92,
                    "min_absolute": 10,
                    "max_percent": 0.97
                },
                "medium": {
                    "target_percent": 0.88,
                    "min_absolute": 20,
                    "max_percent": 0.95
                }
            },
            "transition_smoothing": {
                "target_percent": 0.98,
                "min_absolute": 10,
                "max_percent": 0.995
            },
            "pattern_detection": {
                "target_percent": 0.9,
                "min_absolute": 10,
                "max_percent": 0.95
            }
        },
        "refinement_factors": {
            "k_reduction": 0.8,
            "alpha_reduction": 0.7,
            "alpha_increase": 1.3,
            "beta_increase": 1.2,
            "epsilon_increase": 1.5,
            "sigma_increase": 1.3,
            "weight_reduction": 0.8
        }
    }
    
    return config

def update_config_threshold(config, criterion, new_value, reason, changelog_path="changelog.txt"):
    """
    Met à jour un seuil dans la configuration et log la modification.
    
    Args:
        config: dictionnaire de configuration
        criterion: nom du critère/seuil à modifier
        new_value: nouvelle valeur
        reason: justification du changement
        changelog_path: chemin du fichier changelog
    
    NOTE FPS : Toute modification doit être justifiée et traçable.
    La plasticité du système passe par cette traçabilité complète.
    """
    # Vérifier que le critère existe
    if criterion not in config.get("to_calibrate", {}):
        print(f"[WARNING] Critère '{criterion}' non trouvé dans to_calibrate")
        return False
    
    # Sauvegarder l'ancienne valeur
    old_value = config["to_calibrate"][criterion]
    
    # Mettre à jour
    config["to_calibrate"][criterion] = new_value
    
    # Logger dans changelog
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = config.get("system", {}).get("run_id", "UNKNOWN")
    seed = config.get("system", {}).get("seed", "UNKNOWN")
    
    log_entry = f"[{timestamp}] | Run {run_id} | {criterion}: {old_value} → {new_value} | Raison: {reason} | seed={seed}\n"
    
    try:
        os.makedirs(os.path.dirname(changelog_path) if os.path.dirname(changelog_path) else ".", exist_ok=True)
        with open(changelog_path, "a") as f:
            f.write(log_entry)
        print(f"[UPDATE] {criterion} modifié : {old_value} → {new_value}")
        return True
    except Exception as e:
        print(f"[ERROR] Impossible de logger dans changelog : {e}")
        return False

def validate_adaptive_windows(config):
    """Valide la configuration des fenêtres adaptatives."""
    errors = []
    warnings = []
    
    adaptive_windows = config.get('adaptive_windows', {})
    
    if not adaptive_windows:
        warnings.append("Section 'adaptive_windows' manquante, utilisation des valeurs par défaut")
        return errors, warnings
    
    # Valider chaque section
    expected_sections = ['exploration', 'gamma_adaptation', 'G_effectiveness', 'scoring', 
                        'transition_smoothing', 'pattern_detection']
    
    for section in expected_sections:
        if section not in adaptive_windows:
            warnings.append(f"Section 'adaptive_windows.{section}' manquante")
            continue
            
        section_config = adaptive_windows[section]
        
        if section == 'scoring':
            # Validation spéciale pour scoring qui contient des sous-sections
            expected_windows = ['immediate', 'recent', 'medium']
            for window in expected_windows:
                if window not in section_config:
                    warnings.append(f"Fenêtre 'adaptive_windows.scoring.{window}' manquante")
                    continue
                    
                window_config = section_config[window]
                if not isinstance(window_config, dict):
                    errors.append(f"'adaptive_windows.scoring.{window}' doit être un dictionnaire")
                    continue
                    
                # Vérifier les paramètres requis
                if 'target_percent' not in window_config:
                    errors.append(f"'adaptive_windows.scoring.{window}.target_percent' requis")
                elif not isinstance(window_config['target_percent'], (int, float)) or window_config['target_percent'] <= 0:
                    errors.append(f"'adaptive_windows.scoring.{window}.target_percent' doit être > 0")
                    
                if 'min_absolute' not in window_config:
                    errors.append(f"'adaptive_windows.scoring.{window}.min_absolute' requis")
                elif not isinstance(window_config['min_absolute'], int) or window_config['min_absolute'] <= 0:
                    errors.append(f"'adaptive_windows.scoring.{window}.min_absolute' doit être un entier > 0")
                    
                # Vérifier max_percent optionnel
                if 'max_percent' in window_config:
                    if not isinstance(window_config['max_percent'], (int, float)) or window_config['max_percent'] <= 0:
                        errors.append(f"'adaptive_windows.scoring.{window}.max_percent' doit être > 0")
        else:
            # Validation pour les autres sections
            if not isinstance(section_config, dict):
                errors.append(f"'adaptive_windows.{section}' doit être un dictionnaire")
                continue
                
            # Vérifier les paramètres requis
            required_params = ['target_percent', 'min_absolute']
            for param in required_params:
                if param not in section_config:
                    errors.append(f"'adaptive_windows.{section}.{param}' requis")
                elif param == 'target_percent':
                    if not isinstance(section_config[param], (int, float)) or section_config[param] <= 0:
                        errors.append(f"'adaptive_windows.{section}.{param}' doit être > 0")
                elif param == 'min_absolute':
                    if not isinstance(section_config[param], int) or section_config[param] <= 0:
                        errors.append(f"'adaptive_windows.{section}.{param}' doit être un entier > 0")
                        
            # Vérifier max_percent optionnel
            if 'max_percent' in section_config:
                if not isinstance(section_config['max_percent'], (int, float)) or section_config['max_percent'] <= 0:
                    errors.append(f"'adaptive_windows.{section}.max_percent' doit être > 0")
                if section_config['max_percent'] <= section_config.get('target_percent', 0):
                    warnings.append(f"'adaptive_windows.{section}.max_percent' devrait être > target_percent")
    
    return errors, warnings

def validate_config(config_path_or_dict):
    """
    Fonction importable pour valider un fichier config ou un dict Python.
    Retourne (erreurs, warnings) : listes de messages.
    """
    # Permet d'accepter soit un path (str), soit déjà le dict (pour tests unitaires)
    if isinstance(config_path_or_dict, str) and os.path.exists(config_path_or_dict):
        try:
            with open(config_path_or_dict, "r") as f:
                config = json.load(f)
        except Exception as e:
            return [f"Erreur lors de la lecture du fichier config : {e}"], []
    elif isinstance(config_path_or_dict, dict):
        config = config_path_or_dict
    else:
        return [f"Config non trouvée ou invalide : {config_path_or_dict}"], []

    collector = ValidationErrorCollector()
    # Valider présence des blocs principaux
    validate_main_blocks(config, collector)

    # Les validations suivantes ne s'exécutent que si les blocs existent
    if "system" in config:
        validate_system(config["system"], collector)
        N = config["system"].get("N")
    else:
        N = None

    coupling_type = str(config.get("coupling", {}).get("type", "")).lower()

    if "strates" in config:
        validate_strates(config["strates"], N, collector, coupling_type=coupling_type)
    if "coupling" in config:
        validate_coupling(config["coupling"], collector)
    if "spiral" in config:
        validate_spiral(config["spiral"], collector)
    if "regulation" in config:
        validate_regulation(config["regulation"], collector)
    if "latence" in config:
        validate_latence(config["latence"], collector)
    if "enveloppe" in config:
        validate_enveloppe(config["enveloppe"], collector)
    if "exploration" in config:
        validate_exploration(config["exploration"], collector)
    if "to_calibrate" in config:
        validate_to_calibrate(config["to_calibrate"], collector)
    if "validation" in config:
        validate_validation(config["validation"], collector)
    if "analysis" in config:
        validate_analysis(config["analysis"], collector)
    if "dynamic_parameters" in config:
        validate_dynamic_parameters(config["dynamic_parameters"], collector)
    if "adaptive_windows" in config:
        adaptive_errors, adaptive_warnings = validate_adaptive_windows(config)
        collector.errors.extend(adaptive_errors)
        collector.warnings.extend(adaptive_warnings)

    # Cross-checks globaux
    validate_cross_checks(config, collector)

    # Retourne les listes d'erreurs et de warnings
    return collector.errors, collector.warnings

# 6. — MAIN (EXEMPLE D'UTILISATION) —
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python validate_config.py <config.json>")
        print("   ou : python validate_config.py --generate N T")
        sys.exit(1)

    # Option pour générer une config par défaut
    if sys.argv[1] == "--generate":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        T = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        config = generate_default_config(N, T)
        with open("config_default.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration par défaut générée : config_default.json (N={N}, T={T})")
        sys.exit(0)

    config_path = sys.argv[1]
    errors, warnings = validate_config(config_path)
    
    # Afficher le rapport
    collector = ValidationErrorCollector()
    collector.errors = errors
    collector.warnings = warnings
    collector.report()

    if errors:
        sys.exit(2)
    else:
        print("\nValidation du config.json : OK")