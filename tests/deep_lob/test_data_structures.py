"""Unit tests for data structures."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestOrderSide(unittest.TestCase):
    def test_from_string(self):
        from modules.deep_lob import OrderSide
        self.assertEqual(OrderSide.from_string("B"), OrderSide.BUY)
        self.assertEqual(OrderSide.from_string("buy"), OrderSide.BUY)
        self.assertEqual(OrderSide.from_string("S"), OrderSide.SELL)
        self.assertEqual(OrderSide.from_string("ask"), OrderSide.SELL)
        with self.assertRaises(ValueError):
            OrderSide.from_string("invalid")


class TestOrderType(unittest.TestCase):
    def test_from_string(self):
        from modules.deep_lob import OrderType
        self.assertEqual(OrderType.from_string("T"), OrderType.TRADE)
        self.assertEqual(OrderType.from_string("Add"), OrderType.ADD)
        self.assertEqual(OrderType.from_string("CANCEL"), OrderType.CANCEL)
        with self.assertRaises(ValueError):
            OrderType.from_string("XYZ")


class TestOrder(unittest.TestCase):
    def test_creation(self):
        from modules.deep_lob import Order, OrderSide, OrderType
        o = Order(
            timestamp_ns=1_000_000_000,
            side=OrderSide.BUY,
            type=OrderType.TRADE,
            price=1.2345,
            size=100,
        )
        self.assertEqual(o.side, OrderSide.BUY)
        self.assertEqual(o.size, 100)

    def test_to_array_shape(self):
        from modules.deep_lob import Order, OrderSide, OrderType
        o = Order(0, OrderSide.BUY, OrderType.TRADE, 1.2345, 100)
        arr = o.to_array(mid_price=1.2345, bar_start_ns=0)
        self.assertEqual(arr.shape, (7,))
        self.assertEqual(arr.dtype, np.float32)


class TestOrderBatch(unittest.TestCase):
    def test_empty(self):
        from modules.deep_lob import OrderBatch
        b = OrderBatch.empty()
        self.assertEqual(b.n, 0)
        self.assertEqual(len(b), 0)

    def test_from_orders(self):
        from modules.deep_lob import Order, OrderBatch, OrderSide, OrderType
        orders = [
            Order(i * 1000, OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                  OrderType.TRADE, 1.0 + i * 0.0001, 100 + i)
            for i in range(10)
        ]
        b = OrderBatch.from_orders(orders)
        self.assertEqual(b.n, 10)
        self.assertEqual(b.sides.shape, (10,))
        self.assertEqual(b.prices.shape, (10,))

    def test_from_dataframe(self):
        from modules.deep_lob import OrderBatch
        df = pd.DataFrame({
            "ts_event": pd.date_range("2024-01-01", periods=5, freq="1ms", tz="UTC"),
            "side": ["B", "S", "B", "B", "S"],
            "action": ["T", "A", "C", "T", "T"],
            "price": [1.2345, 1.2346, 1.2345, 1.2347, 1.2346],
            "size": [100, 200, 50, 300, 150],
        })
        b = OrderBatch.from_dataframe(df)
        self.assertEqual(b.n, 5)

    def test_to_feature_matrix_shape(self):
        from modules.deep_lob import Order, OrderBatch, OrderSide, OrderType
        orders = [
            Order(i * 1000, OrderSide.BUY, OrderType.TRADE, 1.2345, 100)
            for i in range(15)
        ]
        b = OrderBatch.from_orders(orders)
        feat, mask = b.to_feature_matrix(mid_price=1.2345, bar_start_ns=0)
        self.assertEqual(feat.shape, (15, 7))
        self.assertEqual(mask.shape, (15,))
        self.assertTrue(mask.all())

    def test_to_feature_matrix_padding(self):
        from modules.deep_lob import Order, OrderBatch, OrderSide, OrderType
        orders = [
            Order(i * 1000, OrderSide.BUY, OrderType.TRADE, 1.2345, 100)
            for i in range(5)
        ]
        b = OrderBatch.from_orders(orders)
        feat, mask = b.to_feature_matrix(mid_price=1.2345, bar_start_ns=0, max_orders=10)
        self.assertEqual(feat.shape, (10, 7))
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(int(mask.sum()), 5)

    def test_to_feature_matrix_truncation(self):
        from modules.deep_lob import Order, OrderBatch, OrderSide, OrderType
        orders = [
            Order(i * 1000, OrderSide.BUY, OrderType.TRADE, 1.2345, 100)
            for i in range(20)
        ]
        b = OrderBatch.from_orders(orders)
        feat, mask = b.to_feature_matrix(mid_price=1.2345, bar_start_ns=0, max_orders=10)
        self.assertEqual(feat.shape, (10, 7))
        self.assertTrue(mask.all())

    def test_filter_by_side(self):
        from modules.deep_lob import Order, OrderBatch, OrderSide, OrderType
        orders = [
            Order(i * 1000, OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                  OrderType.TRADE, 1.0, 100)
            for i in range(10)
        ]
        b = OrderBatch.from_orders(orders)
        b_buys = b.filter(side=OrderSide.BUY)
        self.assertEqual(b_buys.n, 5)


class TestBarLOB(unittest.TestCase):
    def test_creation(self):
        from modules.deep_lob import BarLOB, OrderBatch
        bar = BarLOB(
            bar_start_ns=0,
            bar_end_ns=60_000_000_000,  # 1 min
            mid_price=1.2345,
            orders=OrderBatch.empty(),
        )
        self.assertEqual(bar.n_orders, 0)
        self.assertAlmostEqual(bar.duration_ms, 60_000.0)

    def test_to_features(self):
        from modules.deep_lob import BarLOB, Order, OrderBatch, OrderSide, OrderType
        orders = [Order(i*1000, OrderSide.BUY, OrderType.TRADE, 1.2345, 100) for i in range(5)]
        bar = BarLOB(
            bar_start_ns=0, bar_end_ns=60_000_000_000, mid_price=1.2345,
            orders=OrderBatch.from_orders(orders),
            context={"feat1": 0.5, "feat2": 1.2},
        )
        feats = bar.to_features(max_orders=10)
        self.assertIn("order_features", feats)
        self.assertIn("order_mask", feats)
        self.assertIn("context", feats)
        self.assertEqual(feats["order_features"].shape, (10, 7))
        self.assertEqual(feats["context"].shape, (2,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
