#!/usr/bin/env python3
"""D4 — LOB tensor ↔ MBP-10 fidelity validator (quality gate before training).

WHAT THIS IS (and is NOT)
═════════════════════════
QUANTSYSTEM_WORKFLOW D4 asks: "validate the MBO→book reconstruction against
MBP-10". A code audit (this commit's study) found there is NO MBO→book
reconstruction in the pipeline:
  • build_rolling_lob_tensors_from_mbp READS MBP-10 directly (real depth).
  • build_rolling_lob_tensors_mbo_only emits ZEROS for the depth channels.
  • enrich_mbo_with_core_microstructure rebuilds SCALARS, not a book.
So the relevant RULE 5 check on the CURRENT code is fidelity of the ENCODING:
does the from_mbp tensor faithfully carry the source MBP-10 book, or does the
9-channel encoding silently corrupt it (swap bid↔ask, reverse levels, mangle
sizes)? "A wrong book = a silently corrupted tensor."

🚩 TODO(depth track): when a real MBO→book reconstruction is built (for
   iceberg / order-level depth), THAT reconstruction must be validated against
   MBP-10 snapshots here too — the true D4. It does not exist yet.

THE ENCODING (modules: build_rolling_lob_tensors_from_mbp, _compose snapshot)
════════════════════════════════════════════════════════════════════════════
P = 20 levels = [bid9..bid0  (deepest→best),  ask0..ask9  (best→deepest)].
  ch3 bid_depth_log : log1p(bid_sz[::-1]) on levels 0..9,  ZERO on 10..19
  ch4 ask_depth_log : log1p(ask_sz)        on levels 10..19, ZERO on 0..9
Decode: bid_sz = expm1(ch3[0:L])[::-1] ; ask_sz = expm1(ch4[L:2L]).

USAGE
═════
    from tools.diagnostics.validate_lob_vs_mbp import validate_lob_tensor_vs_mbp
    rep = validate_lob_tensor_vs_mbp(df_mbo, df_mbp, df_bars, freq="5min")
    assert rep["verdict"] == "PASS", rep

Run this on YOUR real MBO+MBP before any depth training (data lives on your
machine; the unit test exercises it on the mock MBP-10 fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def validate_lob_tensor_vs_mbp(
    df_mbo: pd.DataFrame,
    df_mbp: pd.DataFrame,
    df_bars: pd.DataFrame,
    *,
    freq: str = "5min",
    levels: int = 10,
    lookback_bars: int = 50,
    zero_tol: float = 1e-5,
    range_tol: float = 1e-3,
    sample: int | None = None,
) -> dict:
    """Decode the from_mbp tensor's depth channels and check fidelity to MBP-10.

    For each (sampled) bar, at the frontier lag (the bar's own snapshot):
      • STRUCTURE / no-swap: ch3 ≈ 0 on the ask half, ch4 ≈ 0 on the bid half.
      • VALUE / no-corruption: the decoded per-level bid_sz / ask_sz lie within
        [min, max] (± range_tol) of that bar's MBP-10 snapshots — the tensor is
        an imbalance-weighted blend of them, so it must stay inside their range.
        (For a bar with a single MBP snapshot this is an exact match.)
    A bid↔ask swap or a level reversal breaks one of these and is reported.

    Returns a dict: verdict PASS/FAIL, n_bars_checked, swap_detected,
    max_zero_leak, max_range_violation, and the first failing bar (if any).
    """
    import prepare_day_trading as P

    tensors, tensor_ts, _ = P.build_rolling_lob_tensors_from_mbp(
        df_mbo, df_mbp, df_bars, freq=freq, lookback_bars=lookback_bars,
        levels=levels, normalize=False,        # raw log1p — decodable
    )
    return check_lob_tensor_vs_mbp(
        tensors, tensor_ts, df_mbp, freq=freq, levels=levels,
        zero_tol=zero_tol, range_tol=range_tol, sample=sample,
    )


def check_lob_tensor_vs_mbp(
    tensors: np.ndarray,
    tensor_ts: np.ndarray,
    df_mbp: pd.DataFrame,
    *,
    freq: str = "5min",
    levels: int = 10,
    zero_tol: float = 1e-5,
    range_tol: float = 1e-3,
    sample: int | None = None,
) -> dict:
    """The fidelity check on an ALREADY-BUILT raw tensor (separated so a test can
    feed a deliberately corrupted tensor and confirm the corruption is caught)."""
    n_bars, T, Pdim, C = tensors.shape

    mbp = df_mbp.copy()
    mbp["ts_event"] = pd.to_datetime(mbp["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    mbp = mbp.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)
    mbp_ts = mbp["ts_event"].to_numpy("datetime64[ns]")
    bar_ns = pd.Timedelta(freq).to_timedelta64()

    def _col(frame, c):
        return (pd.to_numeric(frame[c], errors="coerce").fillna(0.0).to_numpy(np.float64)
                if c in frame.columns else np.zeros(len(frame)))

    bid_sz = np.vstack([_col(mbp, f"bid_sz_{i:02d}") for i in range(levels)]).T  # (M, L)
    ask_sz = np.vstack([_col(mbp, f"ask_sz_{i:02d}") for i in range(levels)]).T

    idx = np.arange(n_bars)
    if sample and sample < n_bars:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(n_bars, sample, replace=False))

    max_zero_leak = 0.0
    max_range_viol = 0.0
    swap_detected = False
    n_checked = 0
    first_fail = None

    for bi in idx:
        t0 = np.asarray(tensor_ts[bi], dtype="datetime64[ns]")
        t1 = t0 + bar_ns
        lo = int(np.searchsorted(mbp_ts, t0, side="left"))
        hi = int(np.searchsorted(mbp_ts, t1, side="left"))
        if hi <= lo:
            continue                            # no MBP snapshot in this bar → skip
        n_checked += 1
        snap = tensors[bi, -1]                   # (P, C) — the bar's frontier snapshot

        # STRUCTURE: zero-regions (a swap pushes bid data into ch4's bid half etc.)
        zero_leak = max(
            float(np.abs(snap[levels:, 3]).max()),   # ch3 must be 0 on ask half
            float(np.abs(snap[:levels, 4]).max()),   # ch4 must be 0 on bid half
        )
        max_zero_leak = max(max_zero_leak, zero_leak)
        if zero_leak > zero_tol:
            swap_detected = True

        # VALUE: decoded depth must lie within the bar's MBP snapshot range
        bid_dec = np.expm1(snap[:levels, 3].astype(np.float64))[::-1]   # bid_sz best..deepest
        ask_dec = np.expm1(snap[levels:, 4].astype(np.float64))         # ask_sz best..deepest
        bmin, bmax = bid_sz[lo:hi].min(0), bid_sz[lo:hi].max(0)
        amin, amax = ask_sz[lo:hi].min(0), ask_sz[lo:hi].max(0)
        viol = max(
            float(np.maximum(bmin - bid_dec, bid_dec - bmax).max()),
            float(np.maximum(amin - ask_dec, ask_dec - amax).max()),
        )
        max_range_viol = max(max_range_viol, viol)

        if (zero_leak > zero_tol or viol > range_tol) and first_fail is None:
            first_fail = {
                "bar": int(bi), "zero_leak": zero_leak, "range_violation": viol,
                "bid_decoded": bid_dec.round(3).tolist(),
                "bid_mbp_range": [bmin.round(3).tolist(), bmax.round(3).tolist()],
            }

    ok = (max_zero_leak <= zero_tol) and (max_range_viol <= range_tol) and (n_checked > 0)
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_bars_checked": n_checked,
        "swap_detected": swap_detected,
        "max_zero_leak": max_zero_leak,
        "max_range_violation": max_range_viol,
        "first_fail": first_fail,
    }


def assert_mbo_only_has_zero_depth(tensors: np.ndarray, *, tol: float = 1e-6) -> None:
    """D4 (ب): the MBO-only fallback must emit ZERO depth channels (ch0/3/4/7) —
    it reconstructs no book, so it must not present fake depth. Complements C4."""
    for ch in (0, 3, 4, 7):
        m = float(np.abs(tensors[..., ch]).max()) if tensors.size else 0.0
        if m > tol:
            raise AssertionError(
                f"mbo_only depth channel ch{ch} is non-zero (max={m}); the "
                f"trade-flow fallback must not fabricate book depth."
            )
