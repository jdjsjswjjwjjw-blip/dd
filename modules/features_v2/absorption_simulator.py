"""Phase 1.12 — Market-realistic absorption ground-truth simulator.

THE PROBLEM THIS SOLVES
═══════════════════════
The Absorption Intensity Index (modules/microstructure.py) measures whether
a large passive participant is SOAKING UP aggressive flow — heavy signed
volume (CVD) while price stays pinned:

    AII  =  |Δcvd over window|  /  |Δprice over window|   (median-scaled)

The production refinery z-scores this and fires `absorb_z > 1.0`
(prepare_day_trading.py:2380). But nothing PROVES that flag actually fires
during absorption and stays quiet otherwise. A naive "high volume" signal
would also fire during a trending breakout — which is the OPPOSITE of
absorption (price is moving freely, nobody is absorbing).

This simulator plants four regime types with ground-truth labels and runs
the production AII path (via AbsorptionDetector) against them.

THE CRITICAL CONTROL — VOLUME-MATCHED TREND
═══════════════════════════════════════════
The headline confounder is CONFOUND_TREND: it has the SAME one-sided CVD
trajectory as an absorption episode (identical volume, identical side) — it
differs ONLY in that price MOVES with the flow instead of staying pinned.

If AII separates ABSORPTION from CONFOUND_TREND, it is provably measuring
PRICE SUPPRESSION, not just volume. That is the non-circular guarantee: a
detector that merely fired on volume would score both identically and fail
this test. (This is the absorption analogue of the iceberg simulator's
"straddle the threshold" ratio design.)

ANTI-CIRCULARITY GUARANTEE
══════════════════════════
  - Absorption price-excursion is sampled from a range that STRADDLES the
    boundary toward trend territory — weakly-pinned absorption SHOULD be
    harder to detect; that's a real recall measurement, not a bug.
  - CONFOUND_TREND is volume-matched (same |Δcvd|), isolating price.
  - CONFOUND_THIN (small volume, big gaps) and BASELINE (balanced two-sided
    chop) provide the low-AII reference the z-score normalises against.

The ground truth is the planting log (segment spans + labels), not the AII
rule. That is what makes this non-circular.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


# Ground-truth label codes
GT_ABSORPTION = "absorption"
GT_CONFOUND_TREND = "trend"           # volume-matched, price moves → NOT absorption
GT_CONFOUND_THIN = "thin"             # small volume, big price gaps → NOT absorption
GT_BASELINE = "baseline"              # balanced two-sided chop (z-score reference)


@dataclass(frozen=True)
class AbsorptionSimConfig:
    # Absorption episode distribution (deliberately straddles the boundary)
    n_trades_range: tuple[int, int] = (80, 160)        # >= aii_window so AII settles
    trade_size_range: tuple[float, float] = (2.0, 12.0)
    # Price excursion (in ticks) over an absorption episode — SMALL = strongly
    # pinned (high AII); the upper end blends toward trend (anti-circular).
    absorption_excursion_ticks: tuple[float, float] = (0.5, 6.0)
    # A trend confounder moves this many ticks (volume-matched, price flows).
    trend_excursion_ticks: tuple[float, float] = (20.0, 60.0)
    # Thin-liquidity confounder: LOW volume (small sizes) but big price gaps.
    # Length must exceed aii_window so the harness can score it; what makes it
    # "thin" is the small size + large excursion (low volume per price move),
    # not a short burst.
    thin_n_trades_range: tuple[int, int] = (70, 120)
    thin_size_range: tuple[float, float] = (0.3, 2.5)
    thin_excursion_ticks: tuple[float, float] = (15.0, 45.0)
    # Baseline quiet filler length between events (z-score reference window).
    baseline_n_trades_range: tuple[int, int] = (90, 140)
    baseline_excursion_ticks: tuple[float, float] = (4.0, 12.0)
    # Confounder mix (per absorption episode, how many of each to inject)
    confounder_ratio: float = 1.0
    base_inter_trade_ms: float = 100.0
    tick_size: float = 0.0001
    seed: int = 0


@dataclass
class PlantedSegment:
    """One planted regime segment with its ground-truth label + span."""
    gt_label: str
    start_idx: int            # global trade index (inclusive)
    end_idx: int              # global trade index (exclusive)
    side: str                 # dominant aggressor side ('A' buy / 'B' sell)
    planted_volume: float     # total |signed volume| in the segment
    planted_excursion: float  # price excursion in price units
    # planted_ratio = volume / excursion — the absorption "strength" that AII
    # should track. High ratio = strongly-pinned (true absorption).
    planted_ratio: float = 0.0


class AbsorptionGroundTruthSimulator:
    """Generates a trade stream with planted absorption episodes + confounders,
    plus the ground-truth segment log to measure detection against."""

    def __init__(self, config: AbsorptionSimConfig | None = None) -> None:
        self.config = config or AbsorptionSimConfig()
        self._rng = np.random.RandomState(self.config.seed)

    # ── public: build a labeled trade frame ────────────────────────────
    def generate(
        self,
        *,
        n_absorptions: int = 40,
        base_price: float = 1.2500,
        start: str = "2025-04-01 08:00:00",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (trades_df, ground_truth_df).

        trades_df columns: ts_event, price, size, side  (AbsorptionDetector input)
        ground_truth_df: one row per planted segment with gt_label + span.
        """
        cfg = self.config
        n_conf = int(round(n_absorptions * cfg.confounder_ratio))
        plan: list[str] = (
            [GT_ABSORPTION] * n_absorptions
            + [GT_CONFOUND_TREND] * (n_conf // 2)
            + [GT_CONFOUND_THIN] * (n_conf - n_conf // 2)
        )
        self._rng.shuffle(plan)

        rows: list[dict] = []
        segments: list[PlantedSegment] = []
        t = pd.Timestamp(start, tz="UTC")
        price = base_price

        # Always open with a baseline so the z-score has a reference.
        t, price = self._emit_baseline(rows, segments, t, price)
        for kind in plan:
            side = "A" if self._rng.rand() < 0.5 else "B"
            if kind == GT_ABSORPTION:
                t, price = self._emit_absorption(rows, segments, t, price, side)
            elif kind == GT_CONFOUND_TREND:
                t, price = self._emit_trend(rows, segments, t, price, side)
            else:
                t, price = self._emit_thin(rows, segments, t, price, side)
            # quiet baseline between events (re-anchors the z reference)
            t, price = self._emit_baseline(rows, segments, t, price)

        trades = pd.DataFrame(rows)
        gt_df = pd.DataFrame([{
            "gt_label": s.gt_label, "start_idx": s.start_idx, "end_idx": s.end_idx,
            "side": s.side, "planted_volume": s.planted_volume,
            "planted_excursion": s.planted_excursion, "planted_ratio": s.planted_ratio,
        } for s in segments])
        return trades, gt_df

    # ── segment emitters ────────────────────────────────────────────────
    def _emit_absorption(self, rows, segments, t, price, side):
        """Heavy one-sided flow, price PINNED in a tight band. The passive
        side absorbs. High volume / tiny excursion → high AII."""
        cfg = self.config
        n = int(self._rng.randint(*cfg.n_trades_range))
        excursion = float(self._rng.uniform(*cfg.absorption_excursion_ticks)) * cfg.tick_size
        return self._emit_one_sided(
            rows, segments, t, price, side, n, excursion, GT_ABSORPTION,
        )

    def _emit_trend(self, rows, segments, t, price, side):
        """VOLUME-MATCHED control: same one-sided flow as absorption, but
        price MOVES with it (a breakout). Big volume AND big excursion → low
        AII. If the detector fires here it cannot tell absorption from
        momentum (precision loss)."""
        cfg = self.config
        n = int(self._rng.randint(*cfg.n_trades_range))
        excursion = float(self._rng.uniform(*cfg.trend_excursion_ticks)) * cfg.tick_size
        return self._emit_one_sided(
            rows, segments, t, price, side, n, excursion, GT_CONFOUND_TREND,
        )

    def _emit_thin(self, rows, segments, t, price, side):
        """Thin liquidity: few trades, big price gaps. Small volume / big
        excursion → very low AII. Tests the detector doesn't fire on
        gappy low-volume moves."""
        cfg = self.config
        n = int(self._rng.randint(*cfg.thin_n_trades_range))
        excursion = float(self._rng.uniform(*cfg.thin_excursion_ticks)) * cfg.tick_size
        return self._emit_one_sided(
            rows, segments, t, price, side, n, excursion, GT_CONFOUND_THIN,
            size_range=cfg.thin_size_range,
        )

    def _emit_baseline(self, rows, segments, t, price):
        """Balanced two-sided chop — net CVD ~ 0, moderate price walk. This
        is the quiet reference the rolling z-score normalises against."""
        cfg = self.config
        n = int(self._rng.randint(*cfg.baseline_n_trades_range))
        excursion = float(self._rng.uniform(*cfg.baseline_excursion_ticks)) * cfg.tick_size
        start_idx = len(rows)
        p0 = price
        # random walk within +/- excursion, alternating-ish sides (balanced)
        for k in range(n):
            side = "A" if self._rng.rand() < 0.5 else "B"
            sz = float(self._rng.uniform(*cfg.trade_size_range))
            # price wanders but mean-reverts to stay in band
            price = p0 + (self._rng.rand() - 0.5) * 2.0 * excursion
            price = round(price, 6)
            t = t + pd.Timedelta(milliseconds=cfg.base_inter_trade_ms)
            rows.append(self._row(t, price, sz, side))
        end_idx = len(rows)
        segments.append(PlantedSegment(
            GT_BASELINE, start_idx, end_idx, "mixed",
            planted_volume=0.0, planted_excursion=excursion, planted_ratio=0.0,
        ))
        return t, round(price, 6)

    def _emit_one_sided(self, rows, segments, t, price, side, n, excursion, label,
                        *, size_range: tuple[float, float] | None = None):
        """Shared primitive: n same-side trades, price moving monotonically
        by `excursion` total over the segment (bounded random walk toward the
        target). Direction of price move follows the aggressor side."""
        cfg = self.config
        size_range = size_range or cfg.trade_size_range
        start_idx = len(rows)
        p0 = price
        # buy aggressor (side 'A') pushes price up; sell ('B') pushes down
        direction = 1.0 if side == "A" else -1.0
        total_vol = 0.0
        for k in range(n):
            sz = float(self._rng.uniform(*size_range))
            total_vol += sz
            frac = (k + 1) / n
            # monotone drift to the target excursion + small jitter (<= 0.4 tick)
            jitter = (self._rng.rand() - 0.5) * 0.8 * cfg.tick_size
            price = p0 + direction * excursion * frac + jitter
            price = round(price, 6)
            t = t + pd.Timedelta(milliseconds=cfg.base_inter_trade_ms)
            rows.append(self._row(t, price, sz, side))
        end_idx = len(rows)
        eff_excursion = max(abs(price - p0), cfg.tick_size * 0.5)
        segments.append(PlantedSegment(
            label, start_idx, end_idx, side,
            planted_volume=total_vol, planted_excursion=eff_excursion,
            planted_ratio=total_vol / eff_excursion,
        ))
        return t, round(price, 6)

    def _row(self, ts, price, size, side):
        return {"ts_event": ts, "price": float(price),
                "size": float(size), "side": side}


# ── evaluation harness ──────────────────────────────────────────────────────
def _segment_scores(
    detected: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    aii_window: int = 50,
    score_pct: float = 90.0,
) -> pd.DataFrame:
    """For each planted segment, summarise the detector's per-trade output
    over its SETTLED portion (trades at least `aii_window` deep into the
    segment, so the full AII rolling window lies inside the segment — this
    removes boundary contamination from the preceding segment).

    Returns the ground-truth frame with added columns:
        peak_z       : high-percentile absorb_z in the settled portion
        fire_frac    : fraction of settled trades with absorb_z > threshold
        any_fired    : did `fired` trip anywhere settled (kept for diagnostics)
        median_aii   : median AII over the settled portion

    Why fire_frac, not any_fired: over a ~70-trade settled window, a single
    z>1 spike happens by chance even in quiet baseline (P(any z>1) → 1 as the
    window grows). Calibration on the simulator showed baseline any_fired=81%
    but baseline fire_frac median=0.04 vs absorption fire_frac median=0.38 —
    the FRACTION is the robust discriminator, so the segment verdict uses it.
    """
    z = detected["absorb_z"].to_numpy()
    aii = detected["absorption_intensity"].to_numpy()
    fired = detected["fired"].to_numpy()

    peaks, fired_any, fire_frac, med_aii = [], [], [], []
    for _, g in ground_truth.iterrows():
        lo = int(g["start_idx"]) + aii_window
        hi = int(g["end_idx"])
        if hi <= lo:                      # segment shorter than the window
            peaks.append(np.nan); fired_any.append(False)
            fire_frac.append(np.nan); med_aii.append(np.nan)
            continue
        zs = z[lo:hi]; fs = fired[lo:hi]
        peaks.append(float(np.percentile(zs, score_pct)) if len(zs) else np.nan)
        fired_any.append(bool(fs.any()))
        fire_frac.append(float(fs.mean()) if len(fs) else np.nan)
        med_aii.append(float(np.median(aii[lo:hi])) if len(zs) else np.nan)

    out = ground_truth.copy()
    out["peak_z"] = peaks
    out["any_fired"] = fired_any
    out["fire_frac"] = fire_frac
    out["median_aii"] = med_aii
    return out


def evaluate_detector(
    detected: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    aii_window: int = 50,
    min_fire_frac: float = 0.15,
) -> dict:
    """Segment-level precision / recall / F1 for absorption detection.

    A non-baseline segment is "detected" if at least `min_fire_frac` of its
    settled trades fire (absorb_z > z_threshold). ABSORPTION detected = TP;
    CONFOUND_* detected = FP; ABSORPTION not detected = FN. BASELINE is
    excluded from P/R (it is the z-score reference) but its fire rate is
    reported as a sanity check.

    The 0.15 default was calibrated on the simulator: absorption segments
    fire on a median 38% of their settled trades, baseline on 4%, trend on
    ~0%. A 15% floor sits cleanly in the gap, rejecting noise spikes while
    catching genuine sustained absorption.
    """
    scored = _segment_scores(detected, ground_truth, aii_window=aii_window)

    def _detected(mask):
        ff = scored.loc[mask, "fire_frac"]
        return (ff >= min_fire_frac).fillna(False)

    is_abs = scored["gt_label"] == GT_ABSORPTION
    is_conf = scored["gt_label"].isin([GT_CONFOUND_TREND, GT_CONFOUND_THIN])
    is_base = scored["gt_label"] == GT_BASELINE

    det_abs = _detected(is_abs)
    det_conf = _detected(is_conf)

    tp = int(det_abs.sum())
    fn = int((~det_abs).sum())
    fp = int(det_conf.sum())

    n_abs = int(is_abs.sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_abs, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    base_ff = scored.loc[is_base, "fire_frac"]
    base_fire_rate = float((base_ff >= min_fire_frac).fillna(False).mean()) if is_base.any() else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_absorptions": n_abs,
        "n_confounders": int(is_conf.sum()),
        "baseline_fire_rate": base_fire_rate,
        "min_fire_frac": min_fire_frac,
    }


def absorption_vs_trend_separation(
    detected: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    aii_window: int = 50,
) -> dict:
    """THE headline non-circularity metric. Compares the AII distribution on
    ABSORPTION segments vs the VOLUME-MATCHED CONFOUND_TREND segments. Since
    both carry identical one-sided volume, a high separation proves AII is
    measuring price-suppression, not volume.

    Returns median AII for each regime + the Mann-Whitney AUC (probability a
    random absorption trade out-scores a random trend trade; 0.5 = no
    separation, 1.0 = perfect).
    """
    scored = _segment_scores(detected, ground_truth, aii_window=aii_window)
    aii = detected["absorption_intensity"].to_numpy()

    def _pool(label):
        vals = []
        for _, g in scored[scored["gt_label"] == label].iterrows():
            lo = int(g["start_idx"]) + aii_window
            hi = int(g["end_idx"])
            if hi > lo:
                vals.append(aii[lo:hi])
        return np.concatenate(vals) if vals else np.array([])

    abs_vals = _pool(GT_ABSORPTION)
    trend_vals = _pool(GT_CONFOUND_TREND)
    thin_vals = _pool(GT_CONFOUND_THIN)
    base_vals = _pool(GT_BASELINE)

    def _auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(pos, neg, alternative="greater")
        return float(u / (len(pos) * len(neg)))

    return {
        "median_aii_absorption": float(np.median(abs_vals)) if len(abs_vals) else float("nan"),
        "median_aii_trend": float(np.median(trend_vals)) if len(trend_vals) else float("nan"),
        "median_aii_thin": float(np.median(thin_vals)) if len(thin_vals) else float("nan"),
        "median_aii_baseline": float(np.median(base_vals)) if len(base_vals) else float("nan"),
        "auc_absorption_vs_trend": _auc(abs_vals, trend_vals),
        "auc_absorption_vs_thin": _auc(abs_vals, thin_vals),
        "auc_absorption_vs_baseline": _auc(abs_vals, base_vals),
        "n_absorption_trades": int(len(abs_vals)),
        "n_trend_trades": int(len(trend_vals)),
    }


def recall_by_ratio_bucket(
    detected: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    aii_window: int = 50,
    min_fire_frac: float = 0.15,
    buckets: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Sensitivity curve: recall of ABSORPTION segments grouped by their
    planted volume/excursion ratio. Recall should RISE with the ratio —
    strongly-pinned absorption (high ratio) is easier to detect than
    weakly-pinned (low ratio, blending toward trend)."""
    scored = _segment_scores(detected, ground_truth, aii_window=aii_window)
    gt_abs = scored[scored["gt_label"] == GT_ABSORPTION].copy()
    gt_abs["detected"] = (gt_abs["fire_frac"] >= min_fire_frac).fillna(False)
    if buckets is None:
        if len(gt_abs) == 0:
            return pd.DataFrame(columns=["ratio_lo", "ratio_hi", "n", "recall"])
        qs = np.quantile(gt_abs["planted_ratio"], [0.0, 0.25, 0.5, 0.75, 1.0])
        buckets = tuple(float(q) for q in np.unique(qs))

    edges = list(buckets)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gt_abs["planted_ratio"] >= lo) & (gt_abs["planted_ratio"] <= hi)
        n = int(m.sum())
        rec = float(gt_abs.loc[m, "detected"].mean()) if n else float("nan")
        rows.append({"ratio_lo": lo, "ratio_hi": hi, "n": n, "recall": rec})
    return pd.DataFrame(rows)
