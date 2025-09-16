import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class FeatureExtractor:
    """
    Builds multi-scale, adaptively normalized features from a run CSV.
    No fixed thresholds; robust to missing columns.
    """

    def __init__(self, window_percent: float = 0.05, min_window: int = 25):
        self.window_percent = window_percent
        self.min_window = min_window

    def _safe_get(self, df: pd.DataFrame, col: str, default_value: float = 0.0) -> np.ndarray:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(method="bfill").fillna(method="ffill").fillna(default_value)
            return series.to_numpy()
        return np.full(len(df), default_value, dtype=float)

    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
        n = len(df)
        if n == 0:
            return np.zeros((0, 1)), {}, 1.0

        # time step estimate
        if "t" in df.columns and len(df) > 1:
            t = pd.to_numeric(df["t"], errors="coerce").fillna(method="bfill").fillna(method="ffill").fillna(0.0).to_numpy()
            dt = float(np.clip(np.median(np.diff(t)), 1e-6, 1e6))
        else:
            t = np.arange(n)
            dt = 1.0

        # window sizes
        w = max(int(n * self.window_percent), self.min_window)
        w = max(5, min(w, max(10, n // 2)))

        # base signals (safe)
        S = self._safe_get(df, "S(t)")
        C = self._safe_get(df, "C(t)", 1.0)
        Eff = self._safe_get(df, "effort(t)")
        Ent = self._safe_get(df, "entropy_S")
        Err = self._safe_get(df, "mean_abs_error")
        r_t = self._safe_get(df, "r_t", 1.0)

        # derived
        dS = np.hstack([[0.0], np.diff(S)]) / max(dt, 1e-9)
        dEff = np.hstack([[0.0], np.diff(Eff)])
        dEnt = np.hstack([[0.0], np.diff(Ent)])

        # rolling stats
        S_roll = pd.Series(S).rolling(w, min_periods=max(3, w // 2)).mean().fillna(method="bfill").fillna(method="ffill").to_numpy()
        S_std = pd.Series(S).rolling(w, min_periods=max(3, w // 2)).std().replace(0, np.nan).fillna(method="bfill").fillna(method="ffill").fillna(1.0).to_numpy()
        varS = pd.Series(S).rolling(w, min_periods=max(3, w // 2)).var().fillna(method="bfill").fillna(method="ffill").fillna(0.0).to_numpy()

        # normalized channels [0,1]
        def norm01(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            if not np.isfinite(x_min) or not np.isfinite(x_max) or np.isclose(x_max - x_min, 0.0):
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min + 1e-12)

        C_n = norm01(C)
        Eff_n = norm01(Eff)
        Err_n_inv = 1.0 - norm01(Err)  # high is good
        calm = 1.0 - norm01(np.abs(dEff))
        novelty = norm01(np.abs((S - S_roll) / (S_std + 1e-12))) + norm01(np.abs(dEnt))
        predictability_lowvar = 1.0 - norm01(varS)

        # feature matrix (stack robust channels)
        # Keep features low-dimensional, well-behaved
        X = np.stack([
            norm01(S),
            norm01(dS),
            C_n,
            Eff_n,
            norm01(Ent),
            norm01(dEnt),
            1.0 - Err_n_inv,  # back to error [0,1]
            norm01(r_t),
            novelty,
            predictability_lowvar,
            calm,
        ], axis=1)

        extras = {
            "t": t,
            "novelty": novelty,
            "predictability_lowvar": predictability_lowvar,
            "satisfaction_proxy": 0.5 * C_n + 0.3 * Err_n_inv + 0.2 * calm,
            "fatigue_proxy": 0.7 * Eff_n + 0.3 * norm01(pd.Series(Eff).rolling(w, min_periods=max(3, w // 2)).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 3 else 0.0, raw=False).fillna(0.0).to_numpy()),
            "C_n": C_n,
            "Err_n_inv": Err_n_inv,
            "calm": calm,
        }

        return X.astype(float), extras, dt


class KMeansSoft:
    """
    Lightweight KMeans with soft assignments via distance softmax.
    Deterministic init by sampling K points on quantiles.
    """

    def __init__(self, K: int = 4, max_iter: int = 50, temperature: float = 1.0, random_state: Optional[int] = 42):
        self.K = K
        self.max_iter = max_iter
        self.temperature = max(1e-6, float(temperature))
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((self.K, X.shape[1]))
        rng = np.random.default_rng(self.random_state)
        # pick quantile-based seeds for stability
        qs = np.linspace(0.05, 0.95, self.K)
        idx = (qs * (n - 1)).astype(int)
        seeds = X[np.clip(idx, 0, n - 1)]
        # small jitter
        seeds = seeds + 1e-3 * rng.standard_normal(seeds.shape)
        return seeds

    @staticmethod
    def _softmax_neg_dist(d2: np.ndarray, temp: float) -> np.ndarray:
        # d2: (n, K) squared distances
        logits = -d2 / max(temp, 1e-9)
        logits -= np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        P = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)
        return P

    def fit(self, X: np.ndarray) -> None:
        if len(X) == 0:
            self.centroids = np.zeros((self.K, X.shape[1]))
            return
        C = self._init_centroids(X)
        for _ in range(self.max_iter):
            # assignments
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            P = self._softmax_neg_dist(d2, self.temperature)
            # update centroids (soft means)
            weights = P.sum(axis=0) + 1e-12
            C_new = (P.T @ X) / weights[:, None]
            if np.allclose(C_new, C, atol=1e-6):
                C = C_new
                break
            C = C_new
        self.centroids = C

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Model not fitted")
        d2 = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return self._softmax_neg_dist(d2, self.temperature)


def compute_inertia_hard(X: np.ndarray, centroids: np.ndarray, P: np.ndarray) -> float:
    """WCSS using hard assignments derived from soft posteriors."""
    if len(X) == 0:
        return 0.0
    labels = np.argmax(P, axis=1)
    d2 = ((X - centroids[labels]) ** 2).sum(axis=1)
    return float(np.sum(d2))


def select_K_via_knee(X: np.ndarray, K_min: int = 2, K_max: int = 8, temperature: float = 0.5, max_iter: int = 50) -> Tuple[int, Dict[int, float]]:
    """Elbow/knee selection on inertia curve. Returns (best_K, inertia_by_K)."""
    Ks = list(range(K_min, max(K_min + 1, K_max + 1)))
    inertia: Dict[int, float] = {}
    vals: List[float] = []
    for K in Ks:
        km = KMeansSoft(K=K, temperature=temperature, max_iter=max_iter)
        km.fit(X)
        P = km.predict_proba(X)
        I = compute_inertia_hard(X, km.centroids, P)
        inertia[K] = I
        vals.append(I)
    # knee by max distance to line between endpoints
    x = np.array(Ks, dtype=float)
    y = np.array(vals, dtype=float)
    if len(x) >= 3 and np.all(np.isfinite(y)):
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        # line: distance formula
        num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
        d = num / den
        idx = int(np.argmax(d))
        best_K = Ks[idx]
    else:
        # fallback: first K where relative improvement < 10%
        best_K = Ks[0]
        for i in range(1, len(Ks)):
            if (vals[i - 1] - vals[i]) / (vals[i - 1] + 1e-12) < 0.10:
                best_K = Ks[i]
                break
    return best_K, inertia


def learn_axis_from_features(X: np.ndarray, P: np.ndarray, init_signal: Optional[np.ndarray] = None, used_states: Optional[set] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """Learn a linear axis from full features X to approximate one state's posterior (chosen by init_signal correlation),
    avoiding states already used. Returns axis in [0,1] and info (weights, bias, chosen_state, r2, corr).
    """
    n = len(X)
    if n == 0 or P.size == 0:
        return np.zeros((n,), dtype=float), {"weights": [], "bias": 0.0, "best_state": -1, "r2": 0.0, "corr": 0.0}
    K = P.shape[1]
    used = used_states or set()

    # rank states by correlation with init_signal (desc by abs corr)
    order = list(range(K))
    if init_signal is not None and np.any(np.isfinite(init_signal)):
        xs = (init_signal - np.nanmean(init_signal)) / (np.nanstd(init_signal) + 1e-12)
        corrs = []
        for k in range(K):
            ys = (P[:, k] - np.mean(P[:, k])) / (np.std(P[:, k]) + 1e-12)
            corrs.append(float(np.nanmean(xs * ys)))
        order = list(np.argsort(-np.abs(np.array(corrs))))

    chosen_k = None
    chosen_corr = 0.0
    for k in order:
        if k in used:
            continue
        chosen_k = int(k)
        if init_signal is not None and np.any(np.isfinite(init_signal)):
            chosen_corr = float(np.corrcoef(init_signal, P[:, k])[0, 1])
        break
    if chosen_k is None:
        # fallback: least used — pick the one with highest variance
        variances = [float(np.var(P[:, k])) for k in range(K)]
        chosen_k = int(np.argmax(variances))
        chosen_corr = 0.0

    # ridge-lite linear regression to predict P[:, chosen_k]
    Xf = np.concatenate([X, np.ones((n, 1))], axis=1)
    y = P[:, chosen_k]
    A = Xf.T @ Xf + 1e-6 * np.eye(Xf.shape[1])
    w = np.linalg.solve(A, Xf.T @ y)
    y_hat = Xf @ w
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    # squash to [0,1]
    a_min, a_max = float(np.nanmin(y_hat)), float(np.nanmax(y_hat))
    axis01 = (y_hat - a_min) / (a_max - a_min + 1e-12)
    info = {
        "weights": [float(x) for x in w[:-1]],
        "bias": float(w[-1]),
        "best_state": int(chosen_k),
        "r2": float(r2),
        "corr": float(chosen_corr),
    }
    return axis01, info


def learn_satisfaction_axis(C_n: np.ndarray, Err_n_inv: np.ndarray, calm: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Learn a linear projection w on [C_n, Err_n_inv, calm] to best explain one state's posterior.
    Returns (axis_values in [0,1], info dict with weights and best_state).
    """
    if len(C_n) == 0 or P.size == 0:
        return np.zeros_like(C_n), {"weights": [0.0, 0.0, 0.0], "best_state": -1, "r2": 0.0}
    Xf = np.stack([C_n, Err_n_inv, calm], axis=1)
    Xf = np.concatenate([Xf, np.ones((len(Xf), 1))], axis=1)  # bias
    best_r2 = -1.0
    best_w = None
    best_k = -1
    for k in range(P.shape[1]):
        y = P[:, k]
        # ridge-lite with tiny L2 for stability
        A = Xf.T @ Xf + 1e-6 * np.eye(Xf.shape[1])
        w = np.linalg.solve(A, Xf.T @ y)
        y_hat = Xf @ w
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
            best_k = k
    if best_w is None:
        return np.zeros(len(C_n)), {"weights": [0.0, 0.0, 0.0], "best_state": -1, "r2": 0.0}
    # compute axis and squash to [0,1]
    axis = (Xf @ best_w)
    # min-max
    a_min, a_max = float(np.nanmin(axis)), float(np.nanmax(axis))
    axis01 = (axis - a_min) / (a_max - a_min + 1e-12)
    info = {"weights": [float(best_w[0]), float(best_w[1]), float(best_w[2])], "bias": float(best_w[3]), "best_state": int(best_k), "r2": float(best_r2)}
    return axis01, info


def orthogonalize_and_rescale(axes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Gram-Schmidt orthogonalization across time, then rescale each axis to [0,1]."""
    names = list(axes.keys())
    T = len(next(iter(axes.values()))) if axes else 0
    if T == 0:
        return axes
    M = len(names)
    A = np.stack([axes[n] for n in names], axis=1).astype(float)  # T x M
    # center
    A -= np.nanmean(A, axis=0, keepdims=True)
    # Gram-Schmidt
    Q = np.zeros_like(A)
    for j in range(M):
        v = A[:, j].copy()
        for i in range(j):
            rij = float(np.dot(Q[:, i], v) / (np.dot(Q[:, i], Q[:, i]) + 1e-12))
            v = v - rij * Q[:, i]
        Q[:, j] = v
    # rescale each column to [0,1]
    out: Dict[str, np.ndarray] = {}
    for j, n in enumerate(names):
        col = Q[:, j]
        cmin, cmax = float(np.nanmin(col)), float(np.nanmax(col))
        if np.isclose(cmax - cmin, 0.0):
            out[n] = np.zeros(T, dtype=float)
        else:
            out[n] = (col - cmin) / (cmax - cmin)
    return out


def interpret_states(P: np.ndarray, axes: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Summarize mean axis values per state and derive a soft label by dominant axis."""
    K = P.shape[1] if P.ndim == 2 and P.size > 0 else 0
    if K == 0:
        return pd.DataFrame(), {}
    means = []
    labels: Dict[int, str] = {}
    axis_names = list(axes.keys())
    for k in range(K):
        wk = P[:, k]
        wk = wk / (np.sum(wk) + 1e-12)
        row = {"state": k}
        for name in axis_names:
            v = axes[name]
            row[name] = float(np.sum(wk * v))
        # pick dominant axis as label
        dom = max(axis_names, key=lambda n: row[n])
        labels[k] = dom
        row["label"] = dom
        means.append(row)
    df = pd.DataFrame(means)
    return df, labels


def smooth_posteriors(P: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Temporal EMA smoothing of posteriors along time axis."""
    if len(P) == 0:
        return P
    Q = np.empty_like(P)
    Q[0] = P[0]
    for i in range(1, len(P)):
        Q[i] = alpha * Q[i - 1] + (1.0 - alpha) * P[i]
        # renormalize
        s = np.sum(Q[i])
        if s > 0:
            Q[i] /= s
    return Q


def posterior_confidence(P: np.ndarray) -> np.ndarray:
    """Confidence = 1 - normalized entropy of posterior."""
    K = P.shape[1] if P.ndim == 2 else 1
    eps = 1e-12
    ent = -np.sum(P * np.log(P + eps), axis=1)
    ent_max = np.log(max(K, 2))
    conf = 1.0 - (ent / (ent_max + eps))
    return np.clip(conf, 0.0, 1.0)


def analyze_log(log_path: str,
                out_dir: str,
                K: int = 4,
                window_percent: float = 0.05,
                min_window: int = 25,
                temperature: float = 0.5,
                smooth_alpha: float = 0.8,
                make_plots: bool = True,
                auto_K: bool = False,
                K_min: int = 2,
                K_max: int = 8,
                memory_path: Optional[str] = None) -> Dict[str, str]:
    """
    Analyze a run CSV and produce:
      - CSV with t, posteriors per-state, confidence, and learned axes
      - Optional PNG plots
    Returns paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(log_path)

    extractor = FeatureExtractor(window_percent=window_percent, min_window=min_window)
    X, extras, dt = extractor.build(df)

    # Auto-select K if requested
    selected_K = K
    inertia_by_K: Dict[int, float] = {}
    if auto_K and len(X) > 0:
        selected_K, inertia_by_K = select_K_via_knee(X, K_min=K_min, K_max=K_max, temperature=temperature)
    model = KMeansSoft(K=selected_K, temperature=temperature)
    if len(X) == 0:
        P = np.zeros((0, selected_K))
    else:
        model.fit(X)
        P_raw = model.predict_proba(X)
        P = smooth_posteriors(P_raw, alpha=smooth_alpha)

    conf = posterior_confidence(P) if len(P) > 0 else np.array([])

    # assemble output dataframe
    out_df = pd.DataFrame({"t": extras.get("t", np.arange(len(P)))})
    for k in range(P.shape[1] if P.ndim == 2 else 0):
        out_df[f"p_state{k}"] = P[:, k] if len(P) > 0 else []
    out_df["confidence"] = conf

    # learned satisfaction from targeted channels (backward compatible)
    sat_axis, sat_info = learn_satisfaction_axis(
        extras.get("C_n", np.zeros(len(out_df))),
        extras.get("Err_n_inv", np.zeros(len(out_df))),
        extras.get("calm", np.zeros(len(out_df))),
        P if len(P) > 0 else np.zeros((len(out_df), 1)),
    )

    # learn curiosity/boredom/fatigue axes from full features, avoid reusing the same state
    used_states: set = set([int(sat_info.get("best_state", -1))]) if sat_info.get("best_state", -1) >= 0 else set()
    cur_axis, cur_info = learn_axis_from_features(X, P if len(P) > 0 else np.zeros((len(out_df), 1)), init_signal=extras.get("novelty"), used_states=used_states)
    used_states.add(int(cur_info.get("best_state", -1)))
    bor_axis, bor_info = learn_axis_from_features(X, P if len(P) > 0 else np.zeros((len(out_df), 1)), init_signal=extras.get("predictability_lowvar"), used_states=used_states)
    used_states.add(int(bor_info.get("best_state", -1)))
    fat_axis, fat_info = learn_axis_from_features(X, P if len(P) > 0 else np.zeros((len(out_df), 1)), init_signal=extras.get("fatigue_proxy"), used_states=used_states)

    # orthogonalize learned axes to reduce collapse
    axes_raw = {
        "curiosity": cur_axis,
        "boredom": bor_axis,
        "satisfaction": sat_axis,
        "fatigue": fat_axis,
    }
    axes_ortho = orthogonalize_and_rescale(axes_raw)

    out_df["curiosity_learned"] = axes_ortho["curiosity"]
    out_df["boredom_learned"] = axes_ortho["boredom"]
    out_df["satisfaction_learned"] = axes_ortho["satisfaction"]
    out_df["fatigue_learned"] = axes_ortho["fatigue"]

    # interpret states by dominant learned axes
    axes_for_interpret = {
        "curiosity": out_df["curiosity_learned"].to_numpy(),
        "boredom": out_df["boredom_learned"].to_numpy(),
        "satisfaction": out_df["satisfaction_learned"].to_numpy(),
        "fatigue": out_df["fatigue_learned"].to_numpy(),
    }
    summary_df, labels = interpret_states(P, axes_for_interpret)

    # write CSV
    base = os.path.splitext(os.path.basename(log_path))[0]
    csv_path = os.path.join(out_dir, f"sentiments_{base}.csv")
    out_df.to_csv(csv_path, index=False)

    # write states summary
    states_csv = os.path.join(out_dir, f"sentiments_states_summary_{base}.csv")
    if len(summary_df) > 0:
        summary_df.to_csv(states_csv, index=False)
    else:
        states_csv = ""

    # persist lightweight memory (centroids, learned weights)
    memory_written = ""
    if memory_path:
        mem = {
            "centroids": model.centroids.tolist() if model.centroids is not None else [],
            "selected_K": int(selected_K),
            "inertia_by_K": inertia_by_K,
            "satisfaction_weights": sat_info,
            "curiosity_model": cur_info,
            "boredom_model": bor_info,
            "fatigue_model": fat_info,
            "axes_pairwise_corr": {},
        }
        # add pairwise correlations for diagnostics
        try:
            names = ["curiosity_learned", "boredom_learned", "satisfaction_learned", "fatigue_learned"]
            corr = np.corrcoef([out_df[n].to_numpy() for n in names])
            mem["axes_pairwise_corr"] = {f"{names[i]}__{names[j]}": float(corr[i, j]) for i in range(4) for j in range(4)}
        except Exception:
            pass
        try:
            with open(memory_path, "w") as f:
                json.dump(mem, f, indent=2)
            memory_written = memory_path
        except Exception:
            memory_written = ""

    plot_paths: Dict[str, str] = {}
    if make_plots and plt is not None and len(out_df) > 0:
        # stacked posteriors
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(out_df["t"], [out_df[f"p_state{k}"] for k in range(P.shape[1])], labels=[f"state{k}" for k in range(P.shape[1])], alpha=0.8)
        ax.plot(out_df["t"], out_df["confidence"], color="k", linewidth=1.0, label="confidence")
        ax.set_title("Latent state posteriors (stacked) + confidence")
        ax.set_xlabel("t")
        ax.set_ylabel("probability")
        ax.legend(loc="upper right", ncol=min(3, max(1, P.shape[1])))
        p1 = os.path.join(out_dir, f"sentiments_posteriors_{base}.png")
        fig.tight_layout()
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        plot_paths["posteriors"] = p1

        # learned axes
        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes_plot = [
            ("curiosity_learned", "Curiosity (learned)"),
            ("boredom_learned", "Boredom (learned)"),
            ("satisfaction_learned", "Satisfaction (learned)"),
            ("fatigue_learned", "Fatigue (learned)"),
        ]
        for (name, title), a in zip(axes_plot, ax2.ravel()):
            a.plot(out_df["t"], out_df[name], label=name)
            a.set_title(title)
            a.grid(True, alpha=0.3)
        ax2[-1, 0].set_xlabel("t")
        ax2[-1, 1].set_xlabel("t")
        p2 = os.path.join(out_dir, f"sentiments_axes_{base}.png")
        fig2.tight_layout()
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        plot_paths["axes"] = p2

    res = {"csv": csv_path}
    if states_csv:
        res["states_csv"] = states_csv
    if memory_written:
        res["memory"] = memory_written
    res.update(plot_paths)
    return res


def main():
    parser = argparse.ArgumentParser(description="Infer adaptive latent states from FPS run logs (non-arbitrary, soft states).")
    parser.add_argument("--log", required=True, help="Path to run CSV in logs/")
    parser.add_argument("--out", default="sentiments_output", help="Output directory")
    parser.add_argument("--K", type=int, default=4, help="Number of latent states")
    parser.add_argument("--window_percent", type=float, default=0.05, help="Rolling window percentage (0..1)")
    parser.add_argument("--min_window", type=int, default=25, help="Minimum rolling window size in steps")
    parser.add_argument("--temperature", type=float, default=0.5, help="Soft assignment temperature")
    parser.add_argument("--smooth_alpha", type=float, default=0.8, help="Temporal smoothing of posteriors (0..1)")
    parser.add_argument("--auto_K", action="store_true", help="Select K automatically via elbow/knee method")
    parser.add_argument("--K_min", type=int, default=2, help="Minimum K for auto selection")
    parser.add_argument("--K_max", type=int, default=8, help="Maximum K for auto selection")
    parser.add_argument("--memory_path", type=str, default="", help="Optional JSON file to persist learned memory")
    parser.add_argument("--no_plots", action="store_true", help="Disable plot generation")
    args = parser.parse_args()

    paths = analyze_log(
        log_path=args.log,
        out_dir=args.out,
        K=args.K,
        window_percent=args.window_percent,
        min_window=args.min_window,
        temperature=args.temperature,
        smooth_alpha=args.smooth_alpha,
        make_plots=not args.no_plots,
        auto_K=args.auto_K,
        K_min=args.K_min,
        K_max=args.K_max,
        memory_path=(args.memory_path or None),
    )

    print(paths)


if __name__ == "__main__":
    main() 