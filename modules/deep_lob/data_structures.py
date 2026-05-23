"""
modules/deep_lob/data_structures.py
────────────────────────────────────
Type-safe data structures للـ LOB representations.

تستخدم frozen dataclasses + Enum للـ immutability + performance.
Conversion helpers from/to numpy arrays + pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════════


class OrderSide(IntEnum):
    """Order side. IntEnum للـ direct array indexing."""
    BUY = 0
    SELL = 1

    @classmethod
    def from_string(cls, s: str) -> "OrderSide":
        s = s.upper().strip()
        if s in ("B", "BUY", "BID"):
            return cls.BUY
        if s in ("S", "SELL", "ASK"):
            return cls.SELL
        raise ValueError(f"Unknown side: {s!r}")


class OrderType(IntEnum):
    """Order action type."""
    TRADE = 0
    ADD = 1
    CANCEL = 2
    MODIFY = 3

    @classmethod
    def from_string(cls, s: str) -> "OrderType":
        s = s.upper().strip()
        mapping = {
            "T": cls.TRADE, "TRADE": cls.TRADE, "F": cls.TRADE, "FILL": cls.TRADE,
            "A": cls.ADD, "ADD": cls.ADD, "N": cls.ADD, "NEW": cls.ADD,
            "C": cls.CANCEL, "CANCEL": cls.CANCEL, "X": cls.CANCEL,
            "M": cls.MODIFY, "MODIFY": cls.MODIFY, "U": cls.MODIFY,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown order type: {s!r}")


# ════════════════════════════════════════════════════════════════════════════
# Order
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Order:
    """Single order/event في الـ LOB.

    Immutable for hash-ability + cache efficiency.

    Fields
    ──────
    timestamp_ns   : int    — nanosecond timestamp (UTC epoch)
    side           : OrderSide
    type           : OrderType
    price          : float  — quote price
    size           : float  — quantity
    venue          : int    — venue ID (default 0)
    order_id       : int    — optional order ID for tracking
    """

    timestamp_ns: int
    side: OrderSide
    type: OrderType
    price: float
    size: float
    venue: int = 0
    order_id: int = -1

    def to_array(self, mid_price: float, bar_start_ns: int, tick_size: float = 0.0001) -> np.ndarray:
        """Convert to feature array suitable للـ embedding.

        Returns
        -------
        np.ndarray shape (7,):
            [side, type, log_size, price_distance_ticks, time_offset_ms, venue, has_id]
        """
        log_size = float(np.log1p(self.size))
        price_dist_ticks = (self.price - mid_price) / max(tick_size, 1e-9)
        time_offset_ms = (self.timestamp_ns - bar_start_ns) / 1e6
        has_id = 1.0 if self.order_id >= 0 else 0.0

        return np.array([
            float(self.side),
            float(self.type),
            log_size,
            price_dist_ticks,
            time_offset_ms,
            float(self.venue),
            has_id,
        ], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# OrderBatch (vectorized representation)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class OrderBatch:
    """Batch of orders كـ vectorized arrays (للـ efficiency).

    Stores arrays directly (not list of Order objects) للـ PyTorch consumption.

    Shape conventions:
        N = number of orders في الـ batch
        sides[i], types[i], prices[i], sizes[i], timestamps[i] = order i
    """

    sides: np.ndarray         # (N,) int8
    types: np.ndarray         # (N,) int8
    prices: np.ndarray        # (N,) float32
    sizes: np.ndarray         # (N,) float32
    timestamps_ns: np.ndarray # (N,) int64
    venues: np.ndarray        # (N,) int8
    order_ids: np.ndarray     # (N,) int64

    @property
    def n(self) -> int:
        return self.sides.shape[0]

    def __len__(self) -> int:
        return self.n

    def __post_init__(self):
        n = self.sides.shape[0]
        for name, arr in [
            ("types", self.types), ("prices", self.prices),
            ("sizes", self.sizes), ("timestamps_ns", self.timestamps_ns),
            ("venues", self.venues), ("order_ids", self.order_ids),
        ]:
            if arr.shape[0] != n:
                raise ValueError(
                    f"OrderBatch field '{name}' has length {arr.shape[0]}, "
                    f"expected {n} (matching sides)"
                )

    @classmethod
    def empty(cls) -> "OrderBatch":
        """Empty batch (للـ edge cases)."""
        return cls(
            sides=np.zeros(0, dtype=np.int8),
            types=np.zeros(0, dtype=np.int8),
            prices=np.zeros(0, dtype=np.float32),
            sizes=np.zeros(0, dtype=np.float32),
            timestamps_ns=np.zeros(0, dtype=np.int64),
            venues=np.zeros(0, dtype=np.int8),
            order_ids=np.zeros(0, dtype=np.int64),
        )

    @classmethod
    def from_orders(cls, orders: list[Order]) -> "OrderBatch":
        """Construct from list of Order objects (one-time conversion)."""
        if not orders:
            return cls.empty()
        return cls(
            sides=np.array([int(o.side) for o in orders], dtype=np.int8),
            types=np.array([int(o.type) for o in orders], dtype=np.int8),
            prices=np.array([o.price for o in orders], dtype=np.float32),
            sizes=np.array([o.size for o in orders], dtype=np.float32),
            timestamps_ns=np.array([o.timestamp_ns for o in orders], dtype=np.int64),
            venues=np.array([o.venue for o in orders], dtype=np.int8),
            order_ids=np.array([o.order_id for o in orders], dtype=np.int64),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "OrderBatch":
        """Construct from DataFrame مع columns standard MBO.

        Expected columns:
            ts_event, side, action, price, size, [venue, order_id]
        """
        required = {"ts_event", "side", "action", "price", "size"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"OrderBatch.from_dataframe missing: {missing}")

        # Convert timestamps to ns
        ts = df["ts_event"]
        if pd.api.types.is_datetime64_any_dtype(ts):
            ts_ns = ts.astype("int64").to_numpy()
        else:
            ts_ns = pd.to_datetime(ts).astype("int64").to_numpy()

        sides = np.array([
            int(OrderSide.from_string(str(s))) for s in df["side"].values
        ], dtype=np.int8)
        types = np.array([
            int(OrderType.from_string(str(t))) for t in df["action"].values
        ], dtype=np.int8)

        n = len(df)
        return cls(
            sides=sides,
            types=types,
            prices=df["price"].astype(np.float32).values,
            sizes=df["size"].astype(np.float32).values,
            timestamps_ns=ts_ns,
            venues=df.get("venue", pd.Series(np.zeros(n, dtype=np.int8))).astype(np.int8).values,
            order_ids=df.get("order_id", pd.Series(np.full(n, -1, dtype=np.int64))).astype(np.int64).values,
        )

    def to_feature_matrix(
        self,
        mid_price: float,
        bar_start_ns: int,
        tick_size: float = 0.0001,
        max_orders: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion to feature matrix.

        Returns
        -------
        features : (N, 7) float32 — [side, type, log_size, price_dist, time_off, venue, has_id]
        mask     : (N,) bool      — valid positions (False for padding if max_orders specified)
        """
        n = self.n
        if n == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=bool)

        log_sizes = np.log1p(self.sizes)
        price_dists = (self.prices - mid_price) / max(tick_size, 1e-9)
        time_offsets_ms = (self.timestamps_ns - bar_start_ns).astype(np.float32) / 1e6
        has_ids = (self.order_ids >= 0).astype(np.float32)

        features = np.stack([
            self.sides.astype(np.float32),
            self.types.astype(np.float32),
            log_sizes.astype(np.float32),
            price_dists.astype(np.float32),
            time_offsets_ms,
            self.venues.astype(np.float32),
            has_ids,
        ], axis=1)

        mask = np.ones(n, dtype=bool)

        if max_orders is not None and n != max_orders:
            if n > max_orders:
                # truncate (keep most recent)
                features = features[-max_orders:]
                mask = mask[-max_orders:]
            else:
                # pad
                pad_n = max_orders - n
                features = np.vstack([features, np.zeros((pad_n, 7), dtype=np.float32)])
                mask = np.concatenate([mask, np.zeros(pad_n, dtype=bool)])

        return features, mask

    def filter(
        self,
        time_range_ns: tuple[int, int] | None = None,
        side: OrderSide | None = None,
        type: OrderType | None = None,
    ) -> "OrderBatch":
        """Filter to subset matching criteria."""
        mask = np.ones(self.n, dtype=bool)
        if time_range_ns is not None:
            mask &= (self.timestamps_ns >= time_range_ns[0]) & \
                    (self.timestamps_ns < time_range_ns[1])
        if side is not None:
            mask &= self.sides == int(side)
        if type is not None:
            mask &= self.types == int(type)

        return OrderBatch(
            sides=self.sides[mask],
            types=self.types[mask],
            prices=self.prices[mask],
            sizes=self.sizes[mask],
            timestamps_ns=self.timestamps_ns[mask],
            venues=self.venues[mask],
            order_ids=self.order_ids[mask],
        )


# ════════════════════════════════════════════════════════════════════════════
# BarLOB (single bar = orders + snapshot)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class BarLOB:
    """Single bar's worth of LOB data.

    Combines:
        - orders: stream of orders during the bar
        - snapshot: traditional LOB snapshot at bar end (for backward compat)
        - context: 138 features من النظام الحالي (regime, simulators, bell pairs, etc.)
        - metadata: timestamps, mid_price, etc.
    """

    bar_start_ns: int
    bar_end_ns: int
    mid_price: float
    tick_size: float = 0.0001

    orders: OrderBatch = field(default_factory=OrderBatch.empty)
    snapshot: np.ndarray | None = None        # optional: (P, C) classic LOB snapshot
    context: dict[str, float] = field(default_factory=dict)  # 138 features

    @property
    def n_orders(self) -> int:
        return self.orders.n

    @property
    def duration_ms(self) -> float:
        return (self.bar_end_ns - self.bar_start_ns) / 1e6

    def to_features(
        self,
        max_orders: int = 500,
    ) -> dict[str, np.ndarray]:
        """Convert to feature arrays ready للـ PyTorch.

        Returns
        -------
        dict with keys:
            'order_features': (max_orders, 7) float32
            'order_mask':     (max_orders,) bool
            'snapshot':       (P, C) float32 إن وُجد
            'context':        (n_context,) float32
        """
        order_features, order_mask = self.orders.to_feature_matrix(
            mid_price=self.mid_price,
            bar_start_ns=self.bar_start_ns,
            tick_size=self.tick_size,
            max_orders=max_orders,
        )

        result = {
            "order_features": order_features,
            "order_mask": order_mask,
            "n_orders_actual": np.array(min(self.n_orders, max_orders), dtype=np.int32),
        }

        if self.snapshot is not None:
            result["snapshot"] = self.snapshot.astype(np.float32)

        if self.context:
            ctx_values = np.array(list(self.context.values()), dtype=np.float32)
            ctx_values = np.where(np.isfinite(ctx_values), ctx_values, 0.0)
            result["context"] = ctx_values

        return result

    def summary(self) -> dict[str, Any]:
        """Diagnostic summary."""
        return {
            "bar_start_ns": int(self.bar_start_ns),
            "duration_ms": float(self.duration_ms),
            "mid_price": float(self.mid_price),
            "n_orders": int(self.n_orders),
            "n_buys": int((self.orders.sides == int(OrderSide.BUY)).sum()),
            "n_sells": int((self.orders.sides == int(OrderSide.SELL)).sum()),
            "n_trades": int((self.orders.types == int(OrderType.TRADE)).sum()),
            "n_cancels": int((self.orders.types == int(OrderType.CANCEL)).sum()),
            "total_size": float(self.orders.sizes.sum()),
            "has_snapshot": self.snapshot is not None,
            "context_dim": len(self.context),
        }
