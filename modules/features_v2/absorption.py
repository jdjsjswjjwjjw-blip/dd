"""Phase 1.12 — Absorption detector wrapper over the production AII engine.

WHAT "ABSORPTION" MEANS
═══════════════════════
A large passive participant sits on the book and SOAKS UP aggressive flow
without letting price move. The microstructure signature: cumulative volume
delta (CVD) moves a lot while PRICE stays pinned. The Absorption Intensity
Index (AII) captures exactly this:

    AII  =  |Δcvd over window|  /  |Δprice over window|   (median-scaled)

High AII = lots of signed volume, little price movement = absorption.
Low  AII = price moving freely with (or without) volume = NOT absorption.

WHY THIS WRAPPER EXISTS
═══════════════════════
The production refinery (prepare_day_trading.py) computes AII per trade via
modules.microstructure.AbsorptionIntensityEngine, then z-scores it
(EVENT_ZSCORE_WINDOW=100) and fires the discrete signal `absorb_z > 1.0`
(prepare_day_trading.py:2380). That logic is scattered across the streaming
loop. This wrapper COMPOSES the exact same primitives into one stateless,
testable entry point so the ground-truth simulator
(absorption_simulator.py) can measure the REAL production behaviour — not a
re-implementation that might drift from it.

It deliberately does NOT re-derive the AII formula: it imports the same
AbsorptionIntensityEngine the refinery uses, and reproduces the same
rolling z-score the refinery applies. If production changes, this wrapper's
measurement changes with it.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Side conventions — mirror prepare_day_trading.py exactly. In MBO data the
# `side` field is the RESTING order's side; a trade lifting the ASK is a buy.
BUY_SIDES = {"A", "ASK", "BUY", "BOT"}
SELL_SIDES = {"B", "BID", "S", "SELL"}


@dataclass(frozen=True)
class AbsorptionConfig:
    """Detector knobs — defaults mirror the production pipeline.

    aii_window      : AbsorptionIntensityEngine rolling window. The engine's
                      own default is 50 (modules/microstructure.py:155).
    z_window        : rolling window for the z-score. Matches
                      EVENT_ZSCORE_WINDOW=100 (prepare_day_trading.py:87).
    z_min_periods   : matches EVENT_ZSCORE_MIN_PERIODS=20 (line 88).
    z_threshold     : fire when absorb_z exceeds this. Production uses 1.0
                      (prepare_day_trading.py:2380, `absorb_z > 1.0`).
    min_price_move  : floor on |Δprice| to avoid divide-by-zero (engine arg).
    """
    aii_window: int = 50
    z_window: int = 100
    z_min_periods: int = 20
    z_threshold: float = 1.0
    min_price_move: float = 1e-8


class AbsorptionDetector:
    """Stateless wrapper: feed a trade stream, get back the per-trade AII,
    its rolling z-score, and the production fire flag."""

    def __init__(self, config: AbsorptionConfig | None = None) -> None:
        self.config = config or AbsorptionConfig()

    def compute(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Run the production AII path over a trade stream.

        Expected columns:
            ts_event  (datetime64 UTC)
            price     (float)
            size      (float — trade size)
            side      ('A'/'ASK'/... = buy aggressor, 'B'/'BID'/... = sell)

        Returns a frame aligned 1:1 with `trades` (sorted by ts_event):
            ts_event, price, cvd, absorption_intensity, absorb_z, fired
        """
        required = ("ts_event", "price", "size", "side")
        for c in required:
            if c not in trades.columns:
                raise ValueError(f"absorption detector: column missing: {c!r}")

        if len(trades) == 0:
            return pd.DataFrame(columns=(
                "ts_event", "price", "cvd",
                "absorption_intensity", "absorb_z", "fired",
            ))

        # Import the SAME engine the refinery uses — no re-implementation.
        from modules.microstructure import AbsorptionIntensityEngine

        df = trades.sort_values("ts_event").reset_index(drop=True)
        engine = AbsorptionIntensityEngine(
            window=self.config.aii_window,
            min_price_move=self.config.min_price_move,
        )

        prices = pd.to_numeric(df["price"], errors="coerce").to_numpy(np.float64)
        sizes = pd.to_numeric(df["size"], errors="coerce").to_numpy(np.float64)
        sides = df["side"].astype(str).str.upper().to_numpy()

        n = len(df)
        cvd = 0.0
        cvd_arr = np.zeros(n, dtype=np.float64)
        aii_arr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sd = sides[i]
            if sd in BUY_SIDES:
                cvd += sizes[i]
            elif sd in SELL_SIDES:
                cvd -= sizes[i]
            cvd_arr[i] = cvd
            aii_arr[i] = float(engine.update(prices[i], cvd))

        # Rolling z-score — reproduce _zscore() from prepare_day_trading.py.
        aii_s = pd.Series(aii_arr)
        roll = aii_s.rolling(self.config.z_window, min_periods=self.config.z_min_periods)
        absorb_z = ((aii_s - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0).to_numpy()
        fired = absorb_z > self.config.z_threshold

        return pd.DataFrame({
            "ts_event": df["ts_event"].to_numpy(),
            "price": prices,
            "cvd": cvd_arr,
            "absorption_intensity": aii_arr,
            "absorb_z": absorb_z,
            "fired": fired,
        })
