"""
modules/deep_lob/system_integration.py
───────────────────────────────────────
System-level integration helpers that prepare the existing system to receive
the Hierarchical LOB Transformer without modifying core production code.

3 integration points:

  ① Stage 2 replacement: provide HierarchicalLOBTransformer as drop-in for
     DeepLOB CNN في train_v19.py Stage 2.

  ② Bridge enhancement: provide multi-task outputs to IntegrationBridge for
     adaptive TP/SL decisions.

  ③ Pattern library extension: discovered patterns automatically added to
     regime_alpha_library.

All integrations are OPT-IN — system works without them (backward compat 100%).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DeepLOBConfig
from .hierarchical_model import HierarchicalLOBTransformer
from .integration_adapter import DeepLOBCNNAdapter, BridgeAdapter, compute_dynamic_tp_sl


# ════════════════════════════════════════════════════════════════════════════
# Integration Point ①: Stage 2 replacement
# ════════════════════════════════════════════════════════════════════════════


def get_visual_embedder(
    use_transformer: bool = False,
    transformer_checkpoint: str = "outputs/hierarchical_lob_transformer.pt",
    channels: int = 7,
    config: Optional[DeepLOBConfig] = None,
):
    """Factory returning either DeepLOB CNN or Transformer adapter.

    Drop-in replacement function in train_v19.py Stage 2:

        # Before (Stage 2 hardcoded):
        from modules.deeplob_cnn import DeepLOBCNN
        embedder = DeepLOBCNN(channels=7)

        # After (opt-in transformer):
        from modules.deep_lob.system_integration import get_visual_embedder
        embedder = get_visual_embedder(
            use_transformer=os.environ.get('USE_TRANSFORMER', 'false') == 'true',
        )

    Either path provides .predict(lob_tensor) → (N, 8) embedding.
    """
    if use_transformer:
        return DeepLOBCNNAdapter(
            channels=channels,
            brain_file=transformer_checkpoint,
            config=config,
        )
    # fallback: existing CNN
    from modules.deeplob_cnn import DeepLOBCNN
    return DeepLOBCNN(channels=channels)


# ════════════════════════════════════════════════════════════════════════════
# Integration Point ②: Bridge enhancement
# ════════════════════════════════════════════════════════════════════════════


class EnhancedBridgeDecision:
    """Wrapper that adds transformer-derived dynamic TP/SL to bridge decisions.

    Usage in paper_v19.py (when --use-transformer is set):

        from modules.deep_lob.system_integration import EnhancedBridgeDecision

        enhancer = EnhancedBridgeDecision(transformer_model, bridge, regime_tp_sl)
        decision = enhancer.evaluate(bar, dl_proba)
        # decision includes: action, tp_mult_dynamic, sl_mult_dynamic, reason, ...
    """

    def __init__(
        self,
        transformer_model: HierarchicalLOBTransformer,
        bridge,
        regime_tp_sl: dict[str, tuple[float, float]],
    ):
        self.adapter = BridgeAdapter(transformer_model)
        self.bridge = bridge
        self.regime_tp_sl = regime_tp_sl

    def evaluate(self, bar, dl_proba: np.ndarray | None = None) -> dict[str, Any]:
        """Bridge decision + dynamic TP/SL from transformer.

        Parameters
        ----------
        bar : BarLOB
        dl_proba : optional manual DL proba (else computed from transformer)
        """
        # Get full transformer predictions
        preds = self.adapter.predict_with_targets(bar)

        # Use transformer-derived proba if not provided
        if dl_proba is None:
            dl_proba = preds["dl_proba"]

        # Bridge decision (regime-aware via existing logic)
        # Convert BarLOB to row dict for bridge.evaluate_row
        row_dict = {
            **bar.context,
            "regime_label": bar.context.get("regime_label", "trending"),
        }
        bridge_decision = self.bridge.evaluate_row(row_dict, dl_proba)

        # Determine regime + dynamic TP/SL
        regime = row_dict.get("regime_label", "trending")
        base_tp, base_sl = self.regime_tp_sl.get(regime, (2.0, 1.0))
        dyn_tp, dyn_sl = compute_dynamic_tp_sl(
            preds, base_tp_mult=base_tp, base_sl_mult=base_sl,
        )

        return {
            "action": bridge_decision.action.name,
            "bridge_confidence": bridge_decision.confidence,
            "bridge_reason": bridge_decision.reason,
            "tp_mult_dynamic": dyn_tp,
            "sl_mult_dynamic": dyn_sl,
            "transformer_predictions": preds,
            "regime": regime,
        }


# ════════════════════════════════════════════════════════════════════════════
# Integration Point ③: Pattern library extension
# ════════════════════════════════════════════════════════════════════════════


def extend_alpha_library_with_discoveries(
    library_path: str,
    discovered_patterns: list,
    min_score: float = 1.0,
    backup: bool = True,
) -> dict[str, int]:
    """Extend RegimeAlphaLibrary JSON with newly discovered patterns from
    the Transformer.

    Workflow:
        1. Run training + pattern discovery
        2. Filter discoveries by min_score
        3. Convert to alpha format
        4. Append to existing library (with backup)

    Parameters
    ----------
    library_path : path to existing alphas_library.json
    discovered_patterns : list of DiscoveredPattern objects
    min_score : minimum quality score to include

    Returns
    -------
    dict {n_added: int, n_skipped: int, regime_counts: dict}
    """
    from modules.regime_alpha_library import RegimeAlphaLibrary

    # Load existing
    if os.path.exists(library_path):
        library = RegimeAlphaLibrary.load(library_path)
    else:
        library = RegimeAlphaLibrary()

    # Backup
    if backup and os.path.exists(library_path):
        backup_path = library_path + ".bak"
        import shutil
        shutil.copy2(library_path, backup_path)

    n_added = 0
    n_skipped = 0
    regime_counts: dict[str, int] = {}

    for pattern in discovered_patterns:
        if pattern.score < min_score:
            n_skipped += 1
            continue

        # Try to infer regime from metadata or default
        regime = pattern.metadata.get("regime", "trending")

        # Convert to alpha schema
        alpha = {
            "name": f"transformer_{pattern.pattern_id}",
            "method": pattern.method,
            "direction": 1 if pattern.metadata.get("mean_outcome", 0) > 0 else -1,
            "horizon": 6,  # default; could be inferred
            "discovered_by": "hierarchical_lob_transformer",
            "stats": {
                "wr": float(pattern.metadata.get("win_rate", 0.5)),
                "sharpe": float(pattern.metadata.get("sharpe", 0.0)),
                "n_examples": int(pattern.n_examples),
                "score": float(pattern.score),
            },
            "embedding_centroid": (
                pattern.centroid_embedding.tolist()
                if pattern.centroid_embedding is not None else None
            ),
        }

        library.add_alpha(regime, alpha)
        n_added += 1
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    # Save updated library
    library.save(library_path)

    return {
        "n_added": n_added,
        "n_skipped": n_skipped,
        "regime_counts": regime_counts,
        "library_total": library.total_alphas(),
        "library_path": str(library_path),
    }


# ════════════════════════════════════════════════════════════════════════════
# Health check: verify system is ready to receive transformer
# ════════════════════════════════════════════════════════════════════════════


def system_readiness_check() -> dict[str, Any]:
    """Verify the existing system has all necessary hooks to receive the
    Transformer.

    Checks:
        - modules.deeplob_cnn exists (Stage 2 replacement target)
        - modules.integration_bridge exists (Bridge enhancement target)
        - modules.regime_alpha_library exists (Pattern extension target)
        - paper_v19, live_predictor exist (Bridge usage)
        - PyTorch available

    Returns
    -------
    dict with checks + recommendations
    """
    checks: dict[str, dict] = {}

    # PyTorch
    try:
        import torch
        checks["pytorch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    except ImportError:
        checks["pytorch"] = {"available": False}

    # Stage 2 hook
    try:
        from modules.deeplob_cnn import DeepLOBCNN  # noqa
        checks["stage2_hook"] = {"available": True, "module": "modules.deeplob_cnn"}
    except ImportError as e:
        checks["stage2_hook"] = {"available": False, "error": str(e)}

    # Bridge hook
    try:
        from modules.integration_bridge import IntegrationBridge  # noqa
        checks["bridge_hook"] = {"available": True, "module": "modules.integration_bridge"}
    except ImportError as e:
        checks["bridge_hook"] = {"available": False, "error": str(e)}

    # Regime library hook
    try:
        from modules.regime_alpha_library import RegimeAlphaLibrary  # noqa
        checks["regime_library_hook"] = {"available": True}
    except ImportError as e:
        checks["regime_library_hook"] = {"available": False, "error": str(e)}

    # Paper/Live scripts
    root = Path(__file__).parent.parent.parent
    checks["paper_v19"] = {"available": (root / "paper_v19.py").exists()}
    checks["live_predictor"] = {"available": (root / "live_predictor.py").exists()}

    all_ready = all(c.get("available", False) for c in checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "recommendations": _readiness_recommendations(checks),
    }


def _readiness_recommendations(checks: dict) -> list[str]:
    recs = []
    if not checks.get("pytorch", {}).get("available"):
        recs.append("Install PyTorch: pip install torch")
    if not checks.get("pytorch", {}).get("cuda_available"):
        recs.append("Consider GPU: training will be 10-50× faster")
    if not checks.get("stage2_hook", {}).get("available"):
        recs.append("modules.deeplob_cnn missing — Stage 2 replacement unavailable")
    if not checks.get("bridge_hook", {}).get("available"):
        recs.append("modules.integration_bridge missing — Bridge enhancement unavailable")
    if not checks.get("regime_library_hook", {}).get("available"):
        recs.append("modules.regime_alpha_library missing — Pattern extension unavailable")
    if not recs:
        recs.append("✅ System ready to receive Hierarchical LOB Transformer")
    return recs
