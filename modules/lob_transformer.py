"""
lob_transformer.py - LOB depth encoder for late-fusion path
"""

from __future__ import annotations

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("  ⚠️  TensorFlow not installed — LOBTransformer unavailable")


VISUAL_EMB_DIM = 8
DEFAULT_TIME_STEPS = 50
DEFAULT_PRICE_LEVELS = 20
DEFAULT_CHANNELS = 3


def _load_keras_model_allow_lambda(path: str, *, compile: bool = False, custom_objects: dict | None = None):
    """
    Keras 3 blocks Lambda deserialization by default (safe_mode=True).
    Our trusted local checkpoints contain Lambda slicing layers.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow unavailable")
    kwargs = {"compile": compile}
    if custom_objects:
        kwargs["custom_objects"] = custom_objects
    try:
        return tf.keras.models.load_model(path, safe_mode=False, **kwargs)
    except TypeError:
        # Older TF/Keras builds may not expose safe_mode.
        cfg = getattr(tf.keras, "config", None)
        if cfg is not None and hasattr(cfg, "enable_unsafe_deserialization"):
            try:
                cfg.enable_unsafe_deserialization()
            except Exception:
                pass
        return tf.keras.models.load_model(path, **kwargs)


def _infer_depth_view(arr: np.ndarray) -> np.ndarray:
    """
    Accepts LOB tensors shaped (N,T,P,C) or (N,T,P) and returns (N,T,P)
    depth slices for the transformer path.
    """
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 4:
        # Channel-0 is the depth map in the existing QuantSystem tensor format.
        return x[..., 0]
    if x.ndim == 3:
        # Single tensor with channels: (T,P,C) -> (T,P)
        if x.shape[-1] <= 4 and x.shape[0] >= 4 and x.shape[1] >= 4:
            return x[..., 0]
        # Batch without channels: (N,T,P)
        return x
    raise ValueError(f"Unexpected LOB tensor shape: {x.shape}")


def _prepare_lob_model_input(x: np.ndarray, T: int, P: int, C: int) -> np.ndarray:
    """Ensure float32 batch (N, T, P, C) for the LOB encoder."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ..., np.newaxis]
    elif arr.ndim == 3:
        d0, d1, d2 = arr.shape
        if d0 == int(T) and d1 == int(P) and d2 <= int(C):
            arr = arr[np.newaxis, ...]
        else:
            arr = arr[..., np.newaxis]
    elif arr.ndim != 4:
        raise ValueError(f"Unexpected LOB tensor rank: {arr.shape}")

    if arr.shape[1] != int(T) or arr.shape[2] != int(P):
        raise ValueError(
            f"LOB tensor shape {arr.shape} incompatible with encoder (T={T}, P={P})"
        )
    c_r = arr.shape[-1]
    if c_r < int(C):
        pad = np.zeros((arr.shape[0], int(T), int(P), int(C) - c_r), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=-1)
    elif c_r > int(C):
        arr = arr[..., : int(C)]
    return arr


class LOBTransformer:
    """
    Late-fusion depth encoder:
      Input  : (batch, time_steps, levels[, channels])
      Output : (batch, emb_dim) depth embeddings
    """

    def __init__(
        self,
        time_steps: int = DEFAULT_TIME_STEPS,
        price_levels: int = DEFAULT_PRICE_LEVELS,
        channels: int = DEFAULT_CHANNELS,
        emb_dim: int = VISUAL_EMB_DIM,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.10,
        brain_file: str = "outputs/lob_transformer_v19.keras",
    ):
        self.T = int(time_steps)
        self.P = int(price_levels)
        self.C = int(channels)
        self.emb_dim = int(emb_dim)
        self.d_model = int(d_model)
        self.nhead = int(max(1, nhead))
        self.num_layers = int(max(1, num_layers))
        self.ff_dim = int(max(16, ff_dim))
        self.dropout = float(max(0.0, min(dropout, 0.8)))
        self.brain_file = brain_file

        self.model = None           # embedding model
        self._train_model = None    # auxiliary head model
        self._fitted = False

        if not TF_AVAILABLE:
            return

        if os.path.exists(brain_file):
            try:
                loaded = _load_keras_model_allow_lambda(brain_file, compile=False)
                layer_names = {str(getattr(layer, "name", "") or "") for layer in loaded.layers}
                if "multi_lob_fuse" not in layer_names:
                    print(
                        "[LOBTransformer] legacy checkpoint (no multi-channel fuse) — rebuilding encoder"
                    )
                    self.model = self._build_embedding_model()
                    self._fitted = False
                else:
                    self.model = loaded
                    self._fitted = True
                    print(f"[LOBTransformer] loaded: {brain_file}")
            except Exception as exc:
                print(f"[LOBTransformer] broken checkpoint ({exc}) — rebuilding")
                self.model = self._build_embedding_model()
        else:
            print("[LOBTransformer] building depth transformer...")
            self.model = self._build_embedding_model()

    def _transformer_block(self, x, block_idx: int):
        attn = layers.MultiHeadAttention(
            num_heads=self.nhead,
            key_dim=max(4, self.d_model // self.nhead),
            dropout=self.dropout,
            name=f"tfm_mha_{block_idx}",
        )(x, x)
        x = layers.Add(name=f"tfm_add_attn_{block_idx}")([x, attn])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"tfm_ln_attn_{block_idx}")(x)

        ff = layers.Dense(self.ff_dim, activation="gelu", name=f"tfm_ff1_{block_idx}")(x)
        ff = layers.Dropout(self.dropout, name=f"tfm_ff_drop_{block_idx}")(ff)
        ff = layers.Dense(self.d_model, activation=None, name=f"tfm_ff2_{block_idx}")(ff)
        x = layers.Add(name=f"tfm_add_ff_{block_idx}")([x, ff])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"tfm_ln_ff_{block_idx}")(x)
        return x

    def _build_embedding_model(self) -> "tf.keras.Model":
        inp = layers.Input(shape=(self.T, self.P, self.C), name="lob_tensor")
        depth = layers.Lambda(lambda t: t[..., 0], name="depth_channel")(inp)
        flow1 = layers.Lambda(lambda t: t[..., 1], name="flow_channel_1")(inp)
        flow2 = layers.Lambda(lambda t: t[..., 2], name="flow_channel_2")(inp)

        def _branch(z, prefix: str):
            z = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln")(z)
            hdim = int(max(8, self.d_model // 2))
            return layers.TimeDistributed(
                layers.Dense(hdim, activation="gelu"),
                name=f"{prefix}_proj",
            )(z)

        h0 = _branch(depth, "depth")
        h1 = _branch(flow1, "flow1")
        h2 = _branch(flow2, "flow2")
        fused = layers.Add(name="multi_lob_fuse")([h0, h1, h2])
        x = layers.TimeDistributed(
            layers.Dense(self.d_model, activation="gelu"),
            name="depth_input_proj",
        )(fused)

        for block_idx in range(self.num_layers):
            x = self._transformer_block(x, block_idx)

        x = layers.GlobalAveragePooling1D(name="depth_pool")(x)
        x = layers.Dropout(self.dropout, name="depth_dropout")(x)
        emb = layers.Dense(self.emb_dim, activation=None, name="depth_embedding")(x)
        model = Model(inp, emb, name="LOBTransformerDepthEncoder")
        return model

    def _build_train_model(self) -> "tf.keras.Model | None":
        if not TF_AVAILABLE or self.model is None:
            return None
        inp = self.model.input
        emb = self.model.output
        aux = layers.Dense(1, activation="tanh", name="aux_out")(emb)
        train_model = Model(inp, aux, name="LOBTransformerAux")
        train_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",
        )
        return train_model

    def fit_auxiliary(
        self,
        X_tensors: np.ndarray,
        y_targets: np.ndarray,
        epochs: int = 20,
        batch: int = 64,
        output_dir: str = "outputs",
    ) -> None:
        if not TF_AVAILABLE or self.model is None:
            return

        X = _prepare_lob_model_input(np.asarray(X_tensors, dtype=np.float32), self.T, self.P, self.C)
        y = np.asarray(y_targets, dtype=np.float32).reshape(-1, 1)
        if len(X) < 32:
            return

        os.makedirs(output_dir, exist_ok=True)
        self._train_model = self._build_train_model()
        if self._train_model is None:
            return

        split = int(max(1, len(X) * 0.8))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            ),
        ]
        if len(X_val) > 0:
            self._train_model.fit(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                epochs=int(max(1, epochs)),
                batch_size=int(max(8, batch)),
                callbacks=callbacks,
                verbose=0,
            )
        else:
            self._train_model.fit(
                X_tr,
                y_tr,
                epochs=int(max(1, epochs)),
                batch_size=int(max(8, batch)),
                verbose=0,
            )

        # Persist embedding model only.
        self.model.save(self.brain_file)
        self._fitted = True

    def get_embeddings(self, tensor: np.ndarray) -> np.ndarray:
        arr = np.asarray(tensor, dtype=np.float32)
        single = arr.ndim == 2 or (
            arr.ndim == 3
            and arr.shape[0] == int(self.T)
            and arr.shape[1] == int(self.P)
        )
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        try:
            x4 = _prepare_lob_model_input(arr, self.T, self.P, self.C)
        except ValueError:
            # fallback: depth-only view for odd legacy shapes
            x3 = _infer_depth_view(arr)
            if x3.ndim == 2:
                x3 = x3[np.newaxis, ...]
            x4 = _prepare_lob_model_input(x3, self.T, self.P, self.C)

        if not TF_AVAILABLE or self.model is None or not self._fitted:
            out = np.zeros((x4.shape[0], self.emb_dim), dtype=np.float32)
            return out[0] if single else out

        n = int(x4.shape[0])
        if n <= 0:
            out = np.zeros((0, self.emb_dim), dtype=np.float32)
            return out[0] if single else out
        bs = min(256, max(32, n))
        parts = []
        for i in range(0, n, bs):
            parts.append(self.model.predict(x4[i : i + bs], verbose=0))
        emb = np.concatenate(parts, axis=0)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] < self.emb_dim:
            pad = np.zeros((emb.shape[0], self.emb_dim - emb.shape[1]), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=1)
        emb = emb[:, : self.emb_dim]
        return emb[0] if single else emb

    def save(self, output_dir: str = "outputs") -> None:
        if self.model is None:
            return
        path = os.path.join(output_dir, "lob_transformer_v19.keras")
        self.model.save(path)
        print(f"  ✅ LOBTransformer: {path}")

    def load(self, output_dir: str = "outputs") -> bool:
        if not TF_AVAILABLE:
            return False
        path = os.path.join(output_dir, "lob_transformer_v19.keras")
        if not os.path.exists(path):
            return False
        try:
            self.model = _load_keras_model_allow_lambda(path, compile=False)
            self._fitted = True
            return True
        except Exception as exc:
            print(f"  ⚠️ LOBTransformer load failed: {exc}")
            return False
