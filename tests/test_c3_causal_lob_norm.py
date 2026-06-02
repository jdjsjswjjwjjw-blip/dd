"""C3 fix — causal trailing-window LOB tensor normalization (anti-leakage).

Locks the property that _normalize_lob_tensor_nonflat normalizes each bar i
using ONLY bars [i-lookback, i-1] (strictly past) — never the dataset-wide
(N×T) statistics it used before, which leaked the future into every bar.

The decisive test is TRUNCATE-INVARIANCE: a purely causal transform gives the
SAME value for bar i whether or not any bars after i exist. The previous
global-mean/std version FAILED this (truncating changed the stats). These tests
are a property of the function, independent of the data, so they prove the
guarantee for any real tensor.

quant-rigor-guard RULE 1 (no dataset-wide stats) + RULE 5 (DeepLOB prior-window
z-score). quant-exec-kit §4 causal_zscore_normalize.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prepare_day_trading import _normalize_lob_tensor_nonflat as _norm


def _regime_shift_tensor(n=600, t=50, p=20, c=9, seed=0):
    """Synthetic (N,T,P,C) with a sharp regime jump at the midpoint — if the
    normalization ever used global/future stats, that jump would contaminate
    earlier bars and break truncate-invariance."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, t, p, c).astype(np.float32)
    x[n // 2:] = x[n // 2:] * 5.0 + 20.0
    return x


class TestTruncateInvariance:
    """Bar i's normalized value must not change when future bars are removed."""

    def test_prefix_matches_full_run(self):
        x = _regime_shift_tensor(seed=1)
        full = _norm(x.copy())
        for k in (200, 300, 450, 599):
            trunc = _norm(x[:k].copy())
            assert np.allclose(full[:k], trunc, atol=1e-5, equal_nan=True), (
                f"LEAK: bars[0:{k}] changed when bars>={k} were removed"
            )

    def test_expanding_mode_also_causal(self):
        # lookback_bars=0 → expanding (all-prior) window; must also be causal.
        x = _regime_shift_tensor(seed=2)
        full = _norm(x.copy(), lookback_bars=0)
        for k in (250, 500):
            trunc = _norm(x[:k].copy(), lookback_bars=0)
            assert np.allclose(full[:k], trunc, atol=1e-5, equal_nan=True)


class TestFuturePerturbation:
    def test_future_change_leaves_past_untouched(self):
        x = _regime_shift_tensor(seed=3)
        base = _norm(x.copy())
        rng = np.random.RandomState(99)
        pert = x.copy()
        pert[400:] = (rng.randn(*pert[400:].shape) * 99 + 999).astype(np.float32)
        base_pert = _norm(pert)
        assert np.allclose(base[:400], base_pert[:400], atol=1e-5, equal_nan=True), (
            "LEAK: perturbing future bars changed past-bar normalization"
        )


class TestTeeth:
    """Prove the truncate test actually discriminates — a GLOBAL normalizer
    (the old C3 bug) must FAIL it, so a passing causal version is meaningful."""

    @staticmethod
    def _global_norm(x, clip=6.0, eps=1e-12):
        out = x.astype(np.float64).copy()
        n, t, p, c = out.shape
        for cc in range(c):
            for pp in range(p):
                sl = out[:, :, pp, cc].reshape(-1)
                nz = sl[np.abs(sl) > eps]
                if len(nz) < 10:
                    continue
                mu, sd = float(nz.mean()), float(nz.std())
                if sd > 1e-8:
                    out[:, :, pp, cc] = np.clip((out[:, :, pp, cc] - mu) / sd, -clip, clip)
        return out

    def test_global_norm_fails_truncate_invariance(self):
        x = _regime_shift_tensor(seed=4)
        full = self._global_norm(x)
        trunc = self._global_norm(x[:250])
        assert not np.allclose(full[:250], trunc, atol=1e-5), (
            "global normalizer should leak (change on truncation) — if it doesn't, "
            "the truncate test has no teeth"
        )


class TestContract:
    def test_shape_and_dtype_preserved(self):
        x = _regime_shift_tensor(seed=5)
        out = _norm(x.copy())
        assert out.shape == x.shape
        assert out.dtype == np.float32

    def test_warmup_bar_left_raw(self):
        # First bar has no prior history → must be left raw (not normalized).
        x = _regime_shift_tensor(seed=6)
        out = _norm(x.copy())
        assert np.array_equal(out[0], x[0]), "bar 0 has no past → should stay raw"

    def test_non_4d_passthrough(self):
        x = np.random.RandomState(7).randn(100, 9).astype(np.float32)
        out = _norm(x.copy())
        assert np.array_equal(out, x)  # non-(N,T,P,C) returned untouched
