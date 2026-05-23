"""
modules/seasonal_map.py
───────────────────────
Seasonal Map — Causal time/calendar features for QuantSystem V19.

Fills gap #3 of the architecture review: a module that captures temporal
patterns (time-of-day phase, day-of-week, month-end flows, etc.) WITHOUT
look-ahead. Designed to complement `session_features.py` (which provides
basic session flags) and feed into:
  - prepare_day_trading.py (bar-level features)
  - prepare_training_data.py refinery
  - modules/deep_lob/context_encoder.py (138 context features)

All features are derived from `ts_event` alone — pure causal — and produce
deterministic outputs (no randomness, no global aggregates).

API:
    from modules.seasonal_map import add_seasonal_features
    df = add_seasonal_features(df, ts_col='ts_event')

Adds 21 columns (all float32 / int8):
    Time-of-day phase (5):
      session_phase, time_since_london_open_min, time_to_london_close_min,
      time_since_ny_open_min, time_to_ny_close_min
    Day-of-week cyclical (4):
      dow_sin, dow_cos, is_monday, is_friday
    Month/quarter (7):
      dom, dom_sin, dom_cos, is_month_end, is_month_start,
      is_quarter_end, is_year_end
    Week-of-year (3):
      woy_sin, woy_cos, is_first_week_of_year
    Compound (2):
      is_dst_transition_week, is_event_window  (placeholder, hook for calendar DB)
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# Session windows in UTC hours (matches modules/session_features.py)
LONDON_OPEN_HOUR = 7   # 07:00 UTC
LONDON_CLOSE_HOUR = 16  # 16:00 UTC
NY_OPEN_HOUR = 13      # 13:00 UTC
NY_CLOSE_HOUR = 22     # 22:00 UTC


def _coerce_ts(ts_series: pd.Series) -> pd.DatetimeIndex:
    """Convert to naive UTC DatetimeIndex; safe for any input form."""
    if isinstance(ts_series.dtype, pd.DatetimeTZDtype):
        ts = pd.to_datetime(ts_series, errors='coerce').dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        ts = pd.to_datetime(ts_series, errors='coerce', utc=True).dt.tz_localize(None)
    return pd.DatetimeIndex(ts.values)


def _minutes_in_window(hour: np.ndarray, minute: np.ndarray,
                       open_hour: int, close_hour: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (minutes_since_open, minutes_to_close, in_window_mask).

    For times outside the session, since/to_close are 0 and mask is False.
    Mask is the unambiguous indicator (since=0 is valid at session open).
    """
    cur_min = hour.astype(np.int32) * 60 + minute.astype(np.int32)
    open_min = open_hour * 60
    close_min = close_hour * 60
    in_window = (cur_min >= open_min) & (cur_min < close_min)
    since = np.where(in_window, cur_min - open_min, 0).astype(np.int32)
    to_close = np.where(in_window, close_min - cur_min, 0).astype(np.int32)
    return since, to_close, in_window


def add_seasonal_features(
    df: pd.DataFrame,
    ts_col: str = 'ts_event',
    *,
    include_calendar_events: bool = False,
) -> pd.DataFrame:
    """Add seasonal/temporal features to df. All causal, no look-ahead.

    Parameters
    ----------
    df : DataFrame with at least the ts_col timestamp column.
    ts_col : column name containing timestamps (default 'ts_event').
    include_calendar_events : if True, marks placeholder is_event_window=0.
        Set True to allow downstream code to populate from a calendar DB.

    Returns
    -------
    df with ~21 new columns. Idempotent: skips columns that already exist.
    """
    out = df.copy()

    if ts_col not in out.columns:
        # Fallback to index if it's a DatetimeIndex
        if isinstance(out.index, pd.DatetimeIndex):
            ts = out.index
        else:
            raise ValueError(
                f"add_seasonal_features: '{ts_col}' not in df.columns and "
                f"index is not DatetimeIndex"
            )
    else:
        ts = _coerce_ts(out[ts_col])

    # ── Time-of-day phase ────────────────────────────────────────────────
    hour = ts.hour.to_numpy()
    minute = ts.minute.to_numpy()

    london_since, london_to_close, in_london = _minutes_in_window(
        hour, minute, LONDON_OPEN_HOUR, LONDON_CLOSE_HOUR,
    )
    ny_since, ny_to_close, in_ny = _minutes_in_window(
        hour, minute, NY_OPEN_HOUR, NY_CLOSE_HOUR,
    )

    if 'time_since_london_open_min' not in out.columns:
        out['time_since_london_open_min'] = london_since.astype(np.int32)
    if 'time_to_london_close_min' not in out.columns:
        out['time_to_london_close_min'] = london_to_close.astype(np.int32)
    if 'time_since_ny_open_min' not in out.columns:
        out['time_since_ny_open_min'] = ny_since.astype(np.int32)
    if 'time_to_ny_close_min' not in out.columns:
        out['time_to_ny_close_min'] = ny_to_close.astype(np.int32)

    # session_phase: 0=opening (first 30min any active session),
    #                1=middle, 2=closing (last 30min), 3=outside any session
    london_phase = np.where(
        london_since < 30, 0,
        np.where(london_to_close < 30, 2, 1),
    )
    ny_phase = np.where(
        ny_since < 30, 0,
        np.where(ny_to_close < 30, 2, 1),
    )
    # NY phase overrides London when in overlap (NY is dominant in 13:00-16:00)
    phase = np.full(len(out), 3, dtype=np.int8)  # 3 = outside
    london_only = in_london & ~in_ny
    phase[london_only] = london_phase[london_only].astype(np.int8)
    phase[in_ny] = ny_phase[in_ny].astype(np.int8)
    if 'session_phase' not in out.columns:
        out['session_phase'] = phase

    # ── Day-of-week cyclical ─────────────────────────────────────────────
    dow = ts.dayofweek.to_numpy()  # 0=Mon ... 6=Sun
    if 'dow_sin' not in out.columns:
        out['dow_sin'] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    if 'dow_cos' not in out.columns:
        out['dow_cos'] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    if 'is_monday' not in out.columns:
        out['is_monday'] = (dow == 0).astype(np.int8)
    if 'is_friday' not in out.columns:
        out['is_friday'] = (dow == 4).astype(np.int8)

    # ── Day-of-month + month-end / quarter-end / year-end ────────────────
    dom = ts.day.to_numpy()
    month = ts.month.to_numpy()
    days_in_month = ts.days_in_month.to_numpy()
    days_to_eom = (days_in_month - dom).astype(np.int32)

    if 'dom' not in out.columns:
        out['dom'] = dom.astype(np.int8)
    if 'dom_sin' not in out.columns:
        out['dom_sin'] = np.sin(2 * np.pi * (dom - 1) / 31.0).astype(np.float32)
    if 'dom_cos' not in out.columns:
        out['dom_cos'] = np.cos(2 * np.pi * (dom - 1) / 31.0).astype(np.float32)
    if 'is_month_end' not in out.columns:
        # last 2 calendar days of the month (e.g. 30 and 31 for January)
        out['is_month_end'] = (days_to_eom <= 1).astype(np.int8)
    if 'is_month_start' not in out.columns:
        out['is_month_start'] = (dom <= 2).astype(np.int8)
    if 'is_quarter_end' not in out.columns:
        # last 5 days of Mar/Jun/Sep/Dec
        is_quarter_month = np.isin(month, (3, 6, 9, 12))
        out['is_quarter_end'] = (is_quarter_month & (days_to_eom <= 4)).astype(np.int8)
    if 'is_year_end' not in out.columns:
        out['is_year_end'] = ((month == 12) & (days_to_eom <= 4)).astype(np.int8)

    # ── Week-of-year ─────────────────────────────────────────────────────
    woy = ts.isocalendar().week.to_numpy()
    if 'woy_sin' not in out.columns:
        out['woy_sin'] = np.sin(2 * np.pi * woy / 52.0).astype(np.float32)
    if 'woy_cos' not in out.columns:
        out['woy_cos'] = np.cos(2 * np.pi * woy / 52.0).astype(np.float32)
    if 'is_first_week_of_year' not in out.columns:
        out['is_first_week_of_year'] = (woy == 1).astype(np.int8)

    # ── DST transition (Mar/Nov 2nd Sunday US, last Sun Mar/Oct EU) ──────
    # Approximate: flag entire week containing transition
    is_dst_week = (
        ((month == 3) & (dom >= 8) & (dom <= 14))   # US spring forward
        | ((month == 11) & (dom >= 1) & (dom <= 7))  # US fall back
        | ((month == 3) & (dom >= 25))               # EU spring (last week Mar)
        | ((month == 10) & (dom >= 25))              # EU fall (last week Oct)
    )
    if 'is_dst_transition_week' not in out.columns:
        out['is_dst_transition_week'] = is_dst_week.astype(np.int8)

    # ── Placeholder for calendar event DB (FOMC, NFP, CPI) ───────────────
    if 'is_event_window' not in out.columns:
        out['is_event_window'] = np.zeros(len(out), dtype=np.int8)
    # If include_calendar_events=True and a DB is wired up later, replace zeros.

    return out


SEASONAL_FEATURE_COLS = [
    'session_phase',
    'time_since_london_open_min', 'time_to_london_close_min',
    'time_since_ny_open_min', 'time_to_ny_close_min',
    'dow_sin', 'dow_cos', 'is_monday', 'is_friday',
    'dom', 'dom_sin', 'dom_cos',
    'is_month_end', 'is_month_start', 'is_quarter_end', 'is_year_end',
    'woy_sin', 'woy_cos', 'is_first_week_of_year',
    'is_dst_transition_week', 'is_event_window',
]


__all__ = ['add_seasonal_features', 'SEASONAL_FEATURE_COLS']
