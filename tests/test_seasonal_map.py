"""Tests for modules.seasonal_map."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import pandas as pd

from modules.seasonal_map import add_seasonal_features, SEASONAL_FEATURE_COLS


class TestSeasonalMap(unittest.TestCase):

    def _bars(self, start='2024-01-01 00:00:00', n=24*7, freq='1h'):
        ts = pd.date_range(start, periods=n, freq=freq, tz='UTC').tz_localize(None)
        return pd.DataFrame({'ts_event': ts})

    def test_all_columns_added(self):
        df = self._bars()
        out = add_seasonal_features(df)
        for c in SEASONAL_FEATURE_COLS:
            self.assertIn(c, out.columns, f"missing column: {c}")
        self.assertEqual(len(out), len(df))

    def test_dow_cyclical_continuity(self):
        df = self._bars()
        out = add_seasonal_features(df)
        # sin² + cos² = 1
        check = out['dow_sin'].astype(float) ** 2 + out['dow_cos'].astype(float) ** 2
        self.assertTrue(np.allclose(check, 1.0, atol=1e-5))

    def test_is_monday_friday(self):
        # 2024-01-01 is a Monday
        df = self._bars(start='2024-01-01 12:00:00', n=7, freq='1D')
        out = add_seasonal_features(df)
        # Monday=1, Tue=0, Wed=0, Thu=0, Fri=1, Sat=0, Sun=0
        self.assertEqual(out['is_monday'].tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(out['is_friday'].tolist(), [0, 0, 0, 0, 1, 0, 0])

    def test_session_phase(self):
        # Hourly bars across one day
        df = self._bars(start='2024-01-01 00:00:00', n=24, freq='1h')
        out = add_seasonal_features(df)
        phases = out['session_phase'].tolist()
        # 00:00-06:00 = outside (3)
        self.assertTrue(all(p == 3 for p in phases[0:7]))
        # 07:00 = London opening (0)
        self.assertEqual(phases[7], 0)
        # 13:00 = NY opening (0)
        self.assertEqual(phases[13], 0)
        # 22:00+ = outside again
        self.assertEqual(phases[22], 3)

    def test_time_since_london_open(self):
        df = self._bars(start='2024-01-01 07:00:00', n=10, freq='1h')
        out = add_seasonal_features(df)
        # 07:00=0min, 08:00=60min, ..., 15:00=480min, 16:00=0 (closed)
        self.assertEqual(out['time_since_london_open_min'].iloc[0], 0)
        self.assertEqual(out['time_since_london_open_min'].iloc[1], 60)
        self.assertEqual(out['time_since_london_open_min'].iloc[8], 480)
        self.assertEqual(out['time_since_london_open_min'].iloc[9], 0)  # 16:00 closed

    def test_month_end(self):
        # 2024-01-30, 2024-01-31 should be month-end
        ts = pd.to_datetime([
            '2024-01-15', '2024-01-29', '2024-01-30', '2024-01-31', '2024-02-01'
        ])
        df = pd.DataFrame({'ts_event': ts})
        out = add_seasonal_features(df)
        self.assertEqual(out['is_month_end'].tolist(), [0, 0, 1, 1, 0])

    def test_quarter_end(self):
        # 2024-03-30, 2024-06-30, 2024-09-30, 2024-12-30 = quarter ends
        # 2024-01-30 is month-end but NOT quarter-end
        ts = pd.to_datetime([
            '2024-01-30', '2024-03-29', '2024-06-30', '2024-09-30', '2024-12-30',
        ])
        df = pd.DataFrame({'ts_event': ts})
        out = add_seasonal_features(df)
        self.assertEqual(out['is_quarter_end'].tolist(), [0, 1, 1, 1, 1])

    def test_year_end(self):
        # year-end = last 5 days of December (days_to_eom <= 4)
        # 2024-12-27 has days_to_eom=4 → year-end; 2024-12-26 has dte=5 → not
        ts = pd.to_datetime([
            '2024-12-01', '2024-12-26', '2024-12-27', '2024-12-31', '2025-01-01',
        ])
        df = pd.DataFrame({'ts_event': ts})
        out = add_seasonal_features(df)
        self.assertEqual(out['is_year_end'].tolist(), [0, 0, 1, 1, 0])

    def test_idempotent(self):
        """Calling twice should not error and should not overwrite existing cols."""
        df = self._bars()
        out1 = add_seasonal_features(df)
        out1['dow_sin'] = 99.0  # sentinel
        out2 = add_seasonal_features(out1)
        # The sentinel must survive (idempotent skip)
        self.assertTrue((out2['dow_sin'] == 99.0).all())

    def test_no_lookahead(self):
        """Inject a future bar; past rows should not change."""
        df_short = self._bars(n=24)
        out_short = add_seasonal_features(df_short.copy())
        df_long = self._bars(n=48)
        out_long = add_seasonal_features(df_long.copy())
        # First 24 rows should be identical
        for col in SEASONAL_FEATURE_COLS:
            a = out_short[col].to_numpy()
            b = out_long[col].iloc[:24].to_numpy()
            np.testing.assert_array_equal(a, b, err_msg=f"look-ahead in {col}")


if __name__ == '__main__':
    unittest.main()
