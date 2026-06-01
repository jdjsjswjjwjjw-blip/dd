"""Phase 1.9 — Market-realistic iceberg ground-truth simulator.

THE PROBLEM THIS SOLVES
═══════════════════════
The iceberg detector (modules/features_v2/iceberg.py) uses a RULE:
    executed >= 2.5 x max_displayed  AND  refill within 8s  AND  no clearout.

The existing tests (test_phase1_7_iceberg.py) plant icebergs that match
that rule exactly, then check the detector finds them. That validates the
CODE but not the METHODOLOGY — it's tautological (plant-what-you-detect).

This simulator does the opposite: it plants icebergs with parameters drawn
from a realistic DISTRIBUTION that deliberately straddles and exceeds the
detector's thresholds, AND injects confounders (legitimate refills,
visible sweeps, spoofing) that look iceberg-ish but are NOT. Running the
detector against this ground truth yields real precision/recall — and a
threshold-sensitivity curve that answers "is 2.5x / 8s the right cutoff,
or is the rule brittle?".

ANTI-CIRCULARITY GUARANTEE
══════════════════════════
  - Hidden:visible ratio is sampled from [1.5x, 10x] — NOT fixed at 2.5x.
    Icebergs with ratio 1.6x SHOULD be missed by a 2.5x rule; that's a
    real recall measurement, not a bug.
  - Refill latency is sampled from [0.5s, 15s] — straddles the 8s cutoff.
  - Confounders are planted with the SAME surface statistics (refills,
    large executions) but are labeled NOT-iceberg. A detector that fires
    on them loses precision — exactly what we want to measure.

The ground truth is the planting log, not the rule. That is what makes
this non-circular.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd


# ── Realistic parameter distributions (from market-microstructure lit) ─────
# Korajczyk-Murphy (2019) + Hautsch-Huang (2012): institutional iceberg
# hidden:visible ratios cluster 2-8x but the tail reaches 10x+; refill
# latency is sub-second for HFT-managed icebergs, seconds for manual.
@dataclass(frozen=True)
class IcebergSimConfig:
    # Iceberg planting distribution (deliberately wider than the detector rule)
    hidden_visible_ratio_range: tuple[float, float] = (1.5, 10.0)
    refill_latency_s_range: tuple[float, float] = (0.5, 15.0)
    visible_size_range: tuple[int, int] = (5, 50)
    n_refills_range: tuple[int, int] = (2, 8)
    # Confounder mix (per planted iceberg, how many confounders to inject)
    confounder_ratio: float = 1.0      # 1.0 = equal icebergs and confounders
    # Base市场 tick spacing
    base_inter_event_ms: float = 50.0
    tick_size: float = 0.0001
    seed: int = 0


# Ground-truth label codes for the planting log
GT_ICEBERG = "iceberg"
GT_CONFOUND_LEGIT_REFILL = "legit_refill"     # market-maker re-quoting
GT_CONFOUND_VISIBLE_SWEEP = "visible_sweep"   # large trade, nothing hidden
GT_CONFOUND_SPOOF = "spoof"                   # rapid add/cancel, no execution


@dataclass
class PlantedEvent:
    """One planted event with its ground-truth label."""
    price_level: float
    side: str                 # 'B' or 'A'
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    gt_label: str             # GT_* code
    true_hidden_volume: float # actual hidden size (0 for non-icebergs)
    # Provenance — the realistic params used, for the sensitivity curve
    planted_ratio: float = 0.0
    planted_refill_s: float = 0.0


class IcebergGroundTruthSimulator:
    """Generates an MBO tick stream with planted icebergs + confounders,
    plus the ground-truth planting log to measure detection against."""

    def __init__(self, config: IcebergSimConfig | None = None) -> None:
        self.config = config or IcebergSimConfig()
        self._rng = np.random.RandomState(self.config.seed)

    # ── public: build a labeled MBO frame ──────────────────────────────
    def generate(
        self,
        *,
        n_icebergs: int = 50,
        base_price: float = 1.2500,
        start: str = "2025-04-01 08:00:00",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (mbo_df, ground_truth_df).

        mbo_df has the columns the detector expects:
            ts_event, price, side, displayed_size, executed_size, action
        ground_truth_df has one row per planted event with gt_label.
        """
        cfg = self.config
        rows: list[dict] = []
        gt: list[PlantedEvent] = []
        t = pd.Timestamp(start, tz="UTC")

        n_confounders = int(round(n_icebergs * cfg.confounder_ratio))
        # Interleave icebergs + confounders in time
        plan: list[str] = (
            [GT_ICEBERG] * n_icebergs
            + [GT_CONFOUND_LEGIT_REFILL] * (n_confounders // 3)
            + [GT_CONFOUND_VISIBLE_SWEEP] * (n_confounders // 3)
            + [GT_CONFOUND_SPOOF] * (n_confounders - 2 * (n_confounders // 3))
        )
        self._rng.shuffle(plan)

        price = base_price
        for kind in plan:
            # Drift the price a little between events so levels are distinct
            price = round(price + self._rng.randn() * cfg.tick_size * 3, 5)
            side = 'B' if self._rng.rand() < 0.5 else 'A'
            if kind == GT_ICEBERG:
                evt_rows, evt = self._plant_iceberg(t, price, side)
            elif kind == GT_CONFOUND_LEGIT_REFILL:
                evt_rows, evt = self._plant_legit_refill(t, price, side)
            elif kind == GT_CONFOUND_VISIBLE_SWEEP:
                evt_rows, evt = self._plant_visible_sweep(t, price, side)
            else:
                evt_rows, evt = self._plant_spoof(t, price, side)
            rows.extend(evt_rows)
            gt.append(evt)
            # advance time past this event + a gap
            t = evt.end_ts + pd.Timedelta(milliseconds=cfg.base_inter_event_ms * 5)

        mbo = pd.DataFrame(rows).sort_values("ts_event").reset_index(drop=True)
        gt_df = pd.DataFrame([{
            "price_level": e.price_level, "side": e.side,
            "start_ts": e.start_ts, "end_ts": e.end_ts,
            "gt_label": e.gt_label, "true_hidden_volume": e.true_hidden_volume,
            "planted_ratio": e.planted_ratio, "planted_refill_s": e.planted_refill_s,
        } for e in gt])
        return mbo, gt_df

    # ── planting primitives ────────────────────────────────────────────
    def _plant_iceberg(self, t0, price, side):
        """A real iceberg: visible tip refilled repeatedly while a large
        hidden quantity executes. Params sampled from realistic ranges that
        STRADDLE the detector's thresholds (anti-circular)."""
        cfg = self.config
        visible = int(self._rng.randint(*cfg.visible_size_range))
        ratio = float(self._rng.uniform(*cfg.hidden_visible_ratio_range))
        refill_s = float(self._rng.uniform(*cfg.refill_latency_s_range))
        n_refills = int(self._rng.randint(*cfg.n_refills_range))
        hidden_total = float(visible * ratio)
        per_fill = hidden_total / max(n_refills, 1)

        rows = []
        t = t0
        # initial ADD (visible tip)
        rows.append(self._row(t, price, side, displayed=visible, executed=0, action='ADD'))
        for k in range(n_refills):
            t = t + pd.Timedelta(seconds=refill_s * 0.5)
            # FILL consumes the visible tip
            rows.append(self._row(t, price, side, displayed=0, executed=per_fill, action='FILL'))
            t = t + pd.Timedelta(seconds=refill_s)
            # REFILL restores the tip (the iceberg signature)
            disp = visible if k < n_refills - 1 else max(1, visible // 2)  # partial last → no clearout
            rows.append(self._row(t, price, side, displayed=disp, executed=0, action='REFILL'))
        evt = PlantedEvent(price, side, t0, t, GT_ICEBERG,
                           true_hidden_volume=hidden_total,
                           planted_ratio=ratio, planted_refill_s=refill_s)
        return rows, evt

    def _plant_legit_refill(self, t0, price, side):
        """Confounder: a market maker re-quoting after small fills. Has
        REFILL actions, but executed volume stays BELOW the visible size
        (no hidden inventory). A 2.5x rule should NOT fire — if it does,
        that's a precision loss."""
        cfg = self.config
        visible = int(self._rng.randint(*cfg.visible_size_range))
        rows = []
        t = t0
        rows.append(self._row(t, price, side, displayed=visible, executed=0, action='ADD'))
        # small fill (< visible), then a benign re-quote
        for _ in range(2):
            t = t + pd.Timedelta(seconds=1.0)
            rows.append(self._row(t, price, side, displayed=visible // 2, executed=visible * 0.3, action='FILL'))
            t = t + pd.Timedelta(seconds=2.0)
            rows.append(self._row(t, price, side, displayed=visible, executed=0, action='REFILL'))
        evt = PlantedEvent(price, side, t0, t, GT_CONFOUND_LEGIT_REFILL, true_hidden_volume=0.0)
        return rows, evt

    def _plant_visible_sweep(self, t0, price, side):
        """Confounder: a large aggressive trade that sweeps the visible
        book — big executed volume, but NO refill (nothing was hidden).
        The detector requires a refill, so this should NOT fire; tests the
        no-refill branch against a realistic large-trade event."""
        cfg = self.config
        visible = int(self._rng.randint(*cfg.visible_size_range))
        rows = []
        t = t0
        rows.append(self._row(t, price, side, displayed=visible, executed=0, action='ADD'))
        t = t + pd.Timedelta(seconds=0.5)
        # one big fill, book emptied, no refill
        rows.append(self._row(t, price, side, displayed=0, executed=visible * 4.0, action='FILL'))
        evt = PlantedEvent(price, side, t0, t, GT_CONFOUND_VISIBLE_SWEEP, true_hidden_volume=0.0)
        return rows, evt

    def _plant_spoof(self, t0, price, side):
        """Confounder: rapid add/cancel with NO execution (spoofing). Lots
        of book churn, zero executed volume. Must NOT register as iceberg."""
        cfg = self.config
        visible = int(self._rng.randint(*cfg.visible_size_range))
        rows = []
        t = t0
        for _ in range(4):
            rows.append(self._row(t, price, side, displayed=visible, executed=0, action='ADD'))
            t = t + pd.Timedelta(milliseconds=200)
            rows.append(self._row(t, price, side, displayed=0, executed=0, action='CANCEL'))
            t = t + pd.Timedelta(milliseconds=200)
        evt = PlantedEvent(price, side, t0, t, GT_CONFOUND_SPOOF, true_hidden_volume=0.0)
        return rows, evt

    def _row(self, ts, price, side, *, displayed, executed, action):
        return {
            "ts_event": ts, "price": float(price), "side": side,
            "displayed_size": float(displayed), "executed_size": float(executed),
            "action": action,
        }


# ── precision / recall harness ─────────────────────────────────────────────
def evaluate_detector(
    detected_events: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    price_tol: float = 1e-6,
) -> dict:
    """Match detected iceberg events to the ground-truth planting log by
    (price_level, time overlap) and compute precision / recall / F1.

    A detected event is a TRUE POSITIVE iff it overlaps a GT_ICEBERG event
    at the same price. Matching a confounder is a FALSE POSITIVE. A
    GT_ICEBERG with no matching detection is a FALSE NEGATIVE.
    """
    gt_ice = ground_truth[ground_truth["gt_label"] == GT_ICEBERG].copy()
    n_gt_ice = len(gt_ice)

    if detected_events is None or len(detected_events) == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": n_gt_ice,
            "n_detected": 0, "n_gt_icebergs": n_gt_ice,
        }

    det = detected_events.copy()
    det["ts_event"] = pd.to_datetime(det["ts_event"], utc=True)
    matched_gt = set()
    tp = 0
    fp = 0
    for _, d in det.iterrows():
        dp = float(d["price_level"])
        dts = d["ts_event"]
        # find a GT_ICEBERG at the same price whose window contains dts
        hit = None
        for gi, g in gt_ice.iterrows():
            if gi in matched_gt:
                continue
            if abs(float(g["price_level"]) - dp) <= price_tol and (
                g["start_ts"] <= dts <= g["end_ts"] + pd.Timedelta(seconds=1)
            ):
                hit = gi
                break
        if hit is not None:
            tp += 1
            matched_gt.add(hit)
        else:
            fp += 1

    fn = n_gt_ice - len(matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt_ice, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_detected": len(det), "n_gt_icebergs": n_gt_ice,
    }


def recall_by_ratio_bucket(
    detected_events: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    buckets: tuple[float, ...] = (1.5, 2.5, 4.0, 6.0, 10.0),
) -> pd.DataFrame:
    """The threshold-sensitivity curve: recall of GT_ICEBERG events grouped
    by their planted hidden:visible ratio. Reveals whether the detector's
    2.5x rule is well-calibrated — recall should be near-zero below 2.5x
    and near-one above it IF the rule matches reality."""
    gt_ice = ground_truth[ground_truth["gt_label"] == GT_ICEBERG].copy()
    det = detected_events.copy() if detected_events is not None and len(detected_events) else pd.DataFrame(columns=["price_level", "ts_event"])
    if len(det):
        det["ts_event"] = pd.to_datetime(det["ts_event"], utc=True)

    def _is_detected(g) -> bool:
        if len(det) == 0:
            return False
        m = (np.abs(det["price_level"].to_numpy() - float(g["price_level"])) <= 1e-6)
        if not m.any():
            return False
        sub = det[m]
        return bool(((sub["ts_event"] >= g["start_ts"]) &
                     (sub["ts_event"] <= g["end_ts"] + pd.Timedelta(seconds=1))).any())

    gt_ice["detected"] = gt_ice.apply(_is_detected, axis=1)
    edges = list(buckets)
    out_rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gt_ice["planted_ratio"] >= lo) & (gt_ice["planted_ratio"] < hi)
        n = int(m.sum())
        rec = float(gt_ice.loc[m, "detected"].mean()) if n else float("nan")
        out_rows.append({"ratio_lo": lo, "ratio_hi": hi, "n": n, "recall": rec})
    return pd.DataFrame(out_rows)
