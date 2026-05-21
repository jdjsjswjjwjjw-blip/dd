"""
اكتشاف أنماط سيولة بدون إشراف على نوافذ متداخلة من ميزات يوم-تداول جاهزة.

- يقرأ Parquet (مخرجات prepare_day_trading / refinery).
- يبني نوافذ طولها WINDOW من الصفوف الزمنية المتتابعة.
- يطبّع RobustScaler عبر **جزء التدريب الزمني فقط** (بدون تسرّب مستقبل).
- يدرّب Autoencoder على نفس الجزء؛ التحقق زمني على الذيل.
- يجهّز HDBSCAN على embeddings جزء التدريب؛ توزيع الباقي عبر approximate_predict.
- يضيف liquidity_state و one-hot اختياريًا ويحفظ Parquet + ملخص الأنماط.

متطلبات اختيارية: tensorflow، hdbscan (موصى به للاكتشاف التلقائي لعدد الكلاسترات).

تشغيل:
  python liquidity_pattern_discovery.py --parquet IN.parquet --output OUT.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

# ─── قائمة مرشحة: تُقاطع مع أعمدة الملف الفعلية ─────────────────────────────
DEFAULT_LIQUIDITY_FEATURES: list[str] = [
    # CATBOOST_ADVISOR_FEATURES_DT + سياق بار مفيد
    "cvd",
    "obi",
    "absorption_intensity",
    "cancel_ratio",
    "spoofing_ratio",
    "spoofing_duration",
    "liquidity_trap",
    "micro_atr",
    "volume_burst",
    "inter_event_time",
    "micro_price_rel",
    "bid_wall_strength",
    "ask_wall_strength",
    "distance_to_wall",
    "gap_size",
    "liquidity_density",
    "fisher_signal",
    "anomaly",
    "cvd_momentum",
    "cvd_price_divergence",
    "trend_strength",
    "correction_depth",
    "liquidity_sweep",
    "pdh_rel",
    "pdl_rel",
    "dist_to_pdh",
    "price_position",
    "kyle_lambda",
    "hawkes_intensity",
    "vnet",
    "vwap_z_score",
    "bar_cvd_delta",
    "lob_imbalance",
    "order_flow_imbalance",
    "mbp_depth_accel",
    "mbp_bid_slope_intrabar",
    "mbp_ask_slope_intrabar",
    "mbp_roll_lob_coverage",
    "mbp_bar_coverage",
]


def _ensure_time_order(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def build_windows(
    df: pd.DataFrame,
    features: list[str],
    window: int,
) -> tuple[np.ndarray, list[Any]]:
    """نوافذ متداخلة؛ كل صف = تسطيح (window × n_features)."""
    feat_df = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X_list: list[np.ndarray] = []
    idx_list: list[Any] = []
    for i in range(window, len(feat_df)):
        win = feat_df.iloc[i - window : i].to_numpy()
        X_list.append(win.reshape(-1))
        idx_list.append(df.index[i])
    X = np.stack(X_list, axis=0).astype(np.float32)
    print(f"  Windows: {X.shape} ({X.shape[0]:,} × {X.shape[1]} dim)")
    return X, idx_list


def _build_tf_autoencoder(input_dim: int, bottleneck: int):
    import tensorflow as tf

    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    encoded = tf.keras.layers.Dense(bottleneck, activation="linear", name="bottleneck")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(encoded)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    decoded = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    autoencoder = tf.keras.Model(inp, decoded)
    encoder = tf.keras.Model(inp, encoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return autoencoder, encoder


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    bottleneck: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, Any, dict]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    ae, enc = _build_tf_autoencoder(X_train.shape[1], bottleneck)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
    ]
    hist = ae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=0,
    )
    best = int(np.argmin(hist.history["val_loss"]))
    print(
        f"  AE: best_epoch={best + 1} | "
        f"train_loss={hist.history['loss'][best]:.5f} | "
        f"val_loss={hist.history['val_loss'][best]:.5f}"
    )
    return ae, enc, hist.history


def discover_hdbscan_train_then_predict(
    embeddings: np.ndarray,
    split: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> np.ndarray:
    import hdbscan

    emb_tr = embeddings[:split]
    emb_te = embeddings[split:]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
    )
    clusterer.fit(emb_tr)
    labels_tr = clusterer.labels_.astype(np.int64)
    if len(emb_te) > 0:
        labels_te, _ = hdbscan.approximate_predict(clusterer, emb_te)
        labels_te = np.asarray(labels_te, dtype=np.int64)
        labels = np.concatenate([labels_tr, labels_te])
    else:
        labels = labels_tr
    n_clusters = len(set(labels_tr)) - (1 if -1 in labels_tr else 0)
    noise_tr = float((labels_tr == -1).mean() * 100.0)
    print(f"  HDBSCAN (fit train): ~{n_clusters} clusters on train | train-noise: {noise_tr:.1f}%")
    return labels


def discover_gmm_bic_split(
    embeddings: np.ndarray, split: int, k_min: int, k_max: int, seed: int
) -> np.ndarray:
    """BIC على جزء التدريب فقط، ثم predict على كل النوافذ (بدون إعادة ضبط على المستقبل)."""
    emb_tr = embeddings[:split]
    best_bic = np.inf
    best_k = k_min
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            max_iter=200,
        )
        gmm.fit(emb_tr)
        bic = gmm.bic(emb_tr)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    print(f"  GMM (train BIC): best_k={best_k} BIC={best_bic:.0f}")
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=seed,
        max_iter=200,
    )
    gmm.fit(emb_tr)
    labels_tr = gmm.predict(emb_tr).astype(np.int64)
    if split < len(embeddings):
        labels_te = gmm.predict(embeddings[split:]).astype(np.int64)
        return np.concatenate([labels_tr, labels_te])
    return labels_tr


def _rough_pattern_name(row: dict, cluster_id: int) -> str:
    if cluster_id == -1:
        return "Noise_HDBSCAN"
    cvd_mean = row.get("cvd_mean", 0.0) or 0.0
    obi_mean = row.get("obi_mean", 0.0) or 0.0
    abs_mean = row.get("absorption_intensity_mean", 0.0) or 0.0
    if cvd_mean > 0.3 and obi_mean > 0.2:
        return "Bullish_Pressure_heuristic"
    if cvd_mean < -0.3 and obi_mean < -0.2:
        return "Bearish_Pressure_heuristic"
    if abs_mean > 2.0:
        return "Absorption_heuristic"
    return f"Unknown_{cluster_id}"


def interpret_patterns(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    idx_list: list[Any],
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_lab = df_original.loc[idx_list].copy()
    df_lab["liquidity_state"] = labels

    summary_rows: list[dict] = []
    for cluster_id in sorted(set(labels.tolist())):
        mask = labels == cluster_id
        sub_idx = [idx_list[j] for j in range(len(idx_list)) if mask[j]]
        subset = df_original.loc[sub_idx]
        row: dict[str, Any] = {"cluster": int(cluster_id), "count": int(mask.sum())}
        for feat in features:
            if feat in subset.columns:
                row[f"{feat}_mean"] = float(pd.to_numeric(subset[feat], errors="coerce").mean())
        row["pattern_name"] = _rough_pattern_name(row, int(cluster_id))
        summary_rows.append(row)
        cm = row.get("cvd_mean", 0.0) or 0.0
        om = row.get("obi_mean", 0.0) or 0.0
        print(
            f"  Cluster {cluster_id:4d} ({int(mask.sum()):5d}) | "
            f"CVD={cm:+.2f} OBI={om:+.2f} → {row['pattern_name']}"
        )

    return df_lab, pd.DataFrame(summary_rows)


def add_liquidity_state_to_features(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    idx_list: list[Any],
    one_hot: bool,
) -> pd.DataFrame:
    df = df_features.copy()
    state_col = pd.Series(np.nan, index=df.index, dtype=float)
    for idx, lbl in zip(idx_list, labels):
        state_col.loc[idx] = float(lbl)
    df["liquidity_state"] = state_col

    if not one_hot:
        return df

    known = sorted({int(x) for x in labels if int(x) >= 0})
    for s in known:
        df[f"liq_state_{s}"] = (state_col == float(s)).astype(np.float32).fillna(0.0)
    df["liq_state_noise"] = (state_col == -1.0).astype(np.float32).fillna(0.0)
    print(f"  One-hot: {len(known)} clusters + liq_state_noise")
    return df


def run_liquidity_discovery(
    parquet_path: str,
    output_path: str,
    *,
    window: int,
    bottleneck: int,
    train_frac: float,
    method: str,
    epochs: int,
    batch_size: int,
    seed: int,
    one_hot: bool,
    min_cluster_size: int,
    min_samples: int,
    cluster_eps: float,
    gmm_k_min: int,
    gmm_k_max: int,
    artifact_dir: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Liquidity pattern discovery (unsupervised windows → state id)")
    df = pd.read_parquet(parquet_path)
    df = _ensure_time_order(df)

    avail = [f for f in DEFAULT_LIQUIDITY_FEATURES if f in df.columns]
    missing = [f for f in DEFAULT_LIQUIDITY_FEATURES if f not in df.columns]
    print(f"  Features used: {len(avail)}/{len(DEFAULT_LIQUIDITY_FEATURES)}")
    if missing:
        print(f"  (skipped missing: {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''})")
    if len(avail) < 4:
        raise SystemExit(
            "Too few overlapping feature columns in parquet; check file / column names."
        )

    X, idx_list = build_windows(df, avail, window)
    split = int(len(X) * train_frac)
    split = max(1, min(split, len(X) - 1))

    scaler = RobustScaler()
    scaler.fit(X[:split])
    X_scaled = scaler.transform(X).astype(np.float32)

    try:
        _, enc, _ = train_autoencoder(
            X_scaled[:split],
            X_scaled[split:],
            bottleneck=bottleneck,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
    except ImportError:
        print("  tensorflow not installed; pip install tensorflow", file=sys.stderr)
        raise

    embeddings = enc.predict(X_scaled, batch_size=512, verbose=0)
    print(f"  Embeddings: {embeddings.shape}")

    if method == "hdbscan":
        try:
            labels = discover_hdbscan_train_then_predict(
                embeddings,
                split,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_eps,
            )
        except ImportError:
            print("  hdbscan missing → fallback GMM (train BIC)", file=sys.stderr)
            labels = discover_gmm_bic_split(embeddings, split, gmm_k_min, gmm_k_max, seed)
    else:
        labels = discover_gmm_bic_split(embeddings, split, gmm_k_min, gmm_k_max, seed)

    _, summary = interpret_patterns(df, labels, idx_list, avail)
    df_enriched = add_liquidity_state_to_features(df, labels, idx_list, one_hot=one_hot)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df_enriched.to_parquet(output_path, index=True)
    summary_path = output_path.replace(".parquet", "_patterns.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Saved: {summary_path}")

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "liquidity_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        enc.save(os.path.join(artifact_dir, "liquidity_encoder.keras"))
        meta = {
            "features": avail,
            "window": window,
            "bottleneck": bottleneck,
            "train_frac": train_frac,
            "method": method,
            "split_rows": split,
        }
        with open(os.path.join(artifact_dir, "liquidity_discovery_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Artifacts: {artifact_dir}")

    return df_enriched, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Unsupervised liquidity pattern discovery on bar features")
    p.add_argument("--parquet", required=True, help="Input Parquet (bar-level features)")
    p.add_argument("--output", required=True, help="Output Parquet with liquidity_state")
    p.add_argument("--window", type=int, default=12, help="Bars per window (e.g. 12 × 5m = 1h)")
    p.add_argument("--bottleneck", type=int, default=12)
    p.add_argument("--train-frac", type=float, default=0.8, help="Chronological fraction for scaler+AE+HDBSCAN-fit")
    p.add_argument("--method", choices=("hdbscan", "gmm"), default="hdbscan")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--one-hot", action="store_true", help="Add liq_state_* columns for CatBoost")
    p.add_argument("--min-cluster-size", type=int, default=30)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--cluster-eps", type=float, default=0.3)
    p.add_argument("--gmm-k-min", type=int, default=4)
    p.add_argument("--gmm-k-max", type=int, default=15)
    p.add_argument("--artifacts", default=None, help="Directory to save scaler + encoder + meta.json")
    args = p.parse_args()

    run_liquidity_discovery(
        args.parquet,
        args.output,
        window=args.window,
        bottleneck=args.bottleneck,
        train_frac=args.train_frac,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        one_hot=args.one_hot,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_eps=args.cluster_eps,
        gmm_k_min=args.gmm_k_min,
        gmm_k_max=args.gmm_k_max,
        artifact_dir=args.artifacts,
    )


if __name__ == "__main__":
    main()
