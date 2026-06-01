"""Phase 1.4/1.5 feature builders — engineered after the Q2 IC audit.

This package houses feature builders introduced post-cleanup. They are
authored as standalone modules so each carries its own causality tests
and can be invoked independently of the main refinery (useful for
unit-testing on synthetic data + per-feature IC re-audits).

Currently:
    iceberg — Korajczyk-Murphy-style iceberg detector for MBO tick streams.
"""
