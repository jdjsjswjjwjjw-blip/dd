"""
prepare_day_trading.py — Day Trading Refinery for QuantSystem V19
═══════════════════════════════════════════════════════════════════

## تصميم هجين (تكات + شموع) — مسار واحد

**المرحلة 1 — تجميع غني (هذا الملف = «جسر السيولة»، بدون mbo_aggregator منفصل):**
  - من MBO/MBP الخام → شمعة OHLCV + ميزات ميكروستراكتشر مُجمَّعة (CVD، امتصاص، تدفق، …).
  - بناء LOB tensor لكل شمعة (50 × 20 × 3): البعد الأول = آخر 50 شمعة تاريخية؛ مع MBP لقطة ذروة imbalance لكل شمعة.
  - محاذاة سطح CatBoost الـ31 مع prepare_training_data؛ لا نكرّر raw__ هنا لأن المصفاة لا تحجّم أعمدة الـ advisor قبل الحفظ (النسخ كانت مطابقة للعمود الأساسي وتهدر مساحة).
  - vwap_dist داخل الشمعة + VWAP متداول سببي ~ساعة تقويمية (`vwap_roll_1h` / `vwap_dist_1h_roll`) لتأكيد اتجاه الشمع الأدق (مثلاً 5m مع freq=5min أو 1min مع freq=1min).
  - CLI: أي `--freq` صالح في pandas (مثل `1min`)؛ `--slim-output` يقلّل الملفات (بدون preflight parquet وبدون LOB npy)؛ `--sanitize-max-bar-return` يصفّي شمعات بقفزات إغلاق شاذة.
  - تنظيف مدخلات التيك (افتراضيًا مفعّل): فرز زمني، إسقاط صفوف بلا `ts_event`/سعر صالح، إزالة الصفوف المكررة تمامًا، وإزالة تكرار المفتاح عند توفر (`sequence`/`order_id`/… مع `symbol`)؛ على MBP: إسقاط صفوف **BBO متقاطع** (`ask_px_00` < `bid_px_00`). عطّل بـ `--no-sanitize-input-ticks`.
  - **امتصاص هجين:** على مستوى الشمعة، ``absorption_intensity`` = قمة امتصاص التيك داخل الشمعة (``max`` عند التجميع). أعمدة ``dist_to_london_*_atr`` + ``london_sess_*`` تعطي سياق لندن **سببيًا** (بدون لجوء لقمة/قاع يوم لندن الكامل قبل انتهاء الجلسة).
  - **Hawkes / Kyle / VNET هجين:** ``hawkes_intensity`` (max) + ``hawkes_intrabar_sum``؛ ``kyle_lambda`` (max) + ``kyle_lambda_intrabar_mean``؛ ``vnet`` (sum تراكيم الشمعة) + ``vnet_intrabar_last`` (مستوى آخر تيك).

**المرحلة 2 — تدريب موحّد (train_v19.py):**
  - CatBoost/XGB على الميزات الجدولية (+ soft_label أو bias).
  - DeepLOB CNN على tensors المُجمَّعة.
  - MetaLearner (LSTM/TCN) على تسلسلات الشموع بمدخلات: إحصاء + احتمالات الأشجار + embeddings.

**المرحلة 3 — حي (مستقبلاً):**
  - كل إغلاق شمعة: تجميع تيكات → نفس الميزات → نفس النماذج المحفوظة.

المخرج: نفس شكل المصفاة الكاملة بقدر الإمكان
        → train_v19.py يشتغل بدون تعديل على مسار الحدث.
"""

from __future__ import annotations

import argparse
import os
import sys

# يضمن أن جذر المشروع (الذي يحوي modules/ و regime_config.py) على sys.path
# حتى عند تشغيل الملف كسكربت بمسار كامل: `python /path/to/v20/prepare_day_trading.py`.
# مع `python -m v20.prepare_day_trading` هذا no-op لأن cwd جذر المشروع أصلاً.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_candidate, 'modules')) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break

if __name__ == '__main__':
    print('[prepare_day_trading] loading heavy imports (numpy, pandas, TF hooks)...', flush=True)

import glob
import json
import hashlib
import numpy as np
import pandas as pd
try:
    from pykalman import KalmanFilter  # type: ignore
    _KALMAN_OK = True
except Exception:
    KalmanFilter = None  # type: ignore[assignment]
    _KALMAN_OK = False

from modules.context_features import compute_daily_weekly_levels
from modules.tick_intrabar_slices import enrich_bars_with_intrabar
from modules.intrabar_mbp_microstructure import enrich_bars_with_intrabar_mbp
from modules.manifest_v19 import write_manifest

# ─── Regime Configuration (Section 9) ────────────────────────────────────────
try:
    from regime_config import (
        REGIME_TP_SL,
        REGIME_MAX_BARS,
        REGIME_EVENT_THRESHOLD,
        EVENT_ZSCORE_WINDOW,
        EVENT_ZSCORE_MIN_PERIODS,
        EVENT_SCORE_WEIGHTS,
        EVENT_LOB_COVERAGE_CUT,
        EVENT_MBO_COVERAGE_CUT,
        DAYTRADE_EVENT_SCORE_TIER_LABELS,
        EVENT_LABEL_SCORE_STRONG_MIN,
        EVENT_LABEL_SCORE_MID_MIN,
        EVENT_LABEL_TIER_STRONG,
        EVENT_LABEL_TIER_MID,
        EVENT_LABEL_TIER_WEAK,
    )
except ImportError:
    # Fallback إذا لم يُنشأ regime_config.py بعد
    REGIME_TP_SL = {'trending': (2.0, 1.0), 'ranging': (0.8, 0.6), 'volatile': (1.5, 1.5)}
    REGIME_MAX_BARS = {'trending': 12, 'ranging': 6, 'volatile': 3}
    REGIME_EVENT_THRESHOLD = {'trending': 0.60, 'ranging': 0.60, 'volatile': 0.75}
    EVENT_ZSCORE_WINDOW = 100
    EVENT_ZSCORE_MIN_PERIODS = 20
    EVENT_LOB_COVERAGE_CUT = 0.50
    EVENT_MBO_COVERAGE_CUT = 0.50
    EVENT_SCORE_WEIGHTS = {
        'hawkes_z_above_1': 0.26, 'absorb_z_above_1': 0.26,
        'kyle_z_above_05': 0.165, 'cvd_align_above_06': 0.165,
        'mbp_roll_cov_above_cut': 0.075,
        'mbo_tick_cov_above_cut': 0.075,
    }
    DAYTRADE_EVENT_SCORE_TIER_LABELS = True
    EVENT_LABEL_SCORE_STRONG_MIN = 0.70
    EVENT_LABEL_SCORE_MID_MIN = 0.50
    EVENT_LABEL_TIER_STRONG = (2.0, 1.0, 24)
    EVENT_LABEL_TIER_MID = (1.5, 1.0, 12)
    EVENT_LABEL_TIER_WEAK = (1.0, 1.0, 6)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ─── Session Windows (UTC) ────────────────────────────────────────────────────
SESSION_PROFILES: dict[str, dict[str, tuple[str, str]]] = {
    'daytrade_default': {
        'asia': ('00:00', '07:00'),
        'london': ('07:00', '12:00'),
        'overlap': ('12:00', '16:00'),
        'london_ny': ('13:30', '16:00'),
        'ny': ('13:30', '20:00'),
    },
    'stage1_like': {
        'asia': ('00:00', '08:00'),
        'london': ('08:00', '13:00'),
        'overlap': ('13:00', '16:00'),
        'london_ny': ('13:30', '16:00'),
        'ny': ('13:30', '20:00'),
    },
}
SESSIONS: dict[str, tuple[str, str]] = dict(SESSION_PROFILES['daytrade_default'])

# إطار الشمعة الافتراضي لمسار اليوم (pandas offset؛ يُوحَّد مع CLI وسطح بايثون)
DAY_TRADE_DEFAULT_BAR_FREQ: str = '1min'

# ─── Day-trade: سطح الميزات المُصدَّر فقط (بدون مشتقات إضافية في الـ parquet) ─
ORIGINAL_FEATURES: tuple[str, ...] = (
    # Core liquidity & flow (من التيكات)
    'cvd', 'session_cvd', 'absorption_intensity', 'cancel_ratio',
    'spoofing_ratio', 'spoofing_duration', 'liquidity_trap',
    'kyle_lambda', 'hawkes_intensity', 'vnet',
    'hawkes_intrabar_sum', 'kyle_lambda_intrabar_mean', 'vnet_intrabar_last',
    'micro_atr', 'volume_burst', 'inter_event_time',
    'fisher_signal', 'anomaly',
    'cvd_momentum', 'cvd_price_divergence',
    'trend_strength', 'correction_depth', 'liquidity_sweep',
    # تقريب امتصاص جانبي (يُضاف أحيانًا خارج التجميع؛ غير موجود يُتخطّى)
    'buy_absorption_approx', 'sell_absorption_approx',
    # Order book (عمق)
    'obi', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    'micro_price', 'current_vwap', 'vwap_z_score',
    # Levels
    'pdh', 'pdl', 'dist_to_pdh', 'price_position',
    'london_sess_high', 'london_sess_low',
    'dist_to_london_high_atr', 'dist_to_london_low_atr',
    # Tick count & coverage
    'tick_count', 'mbp_bar_coverage', 'mbo_bar_coverage',
    'mbp_roll_lob_coverage', 'liquidity_gaps',
    # Session context (وسوم؛ يمكن استبعادها من مدخلات النموذج لاحقًا)
    'is_london', 'is_overlap', 'is_ny',
    # ── Volatility / Technical (computed in add_day_trading_features
    # but previously DROPPED at save by this whitelist) ──
    'atr_14', 'is_session_break',
    'bar_range', 'body_ratio',
    'rsi_14', 'macd_hist',
    'vwap_dist', 'vwap_dist_1h_roll', 'lob_imbalance',
    'return_6b', 'return_1h', 'return_4h',
    'volume_ratio_6b', 'volume_ratio_1h', 'volume_ratio_4h',
    'cvd_slope_6b', 'cvd_slope_1h', 'cvd_slope_4h',
    # ── Seasonal Map (21 features from modules/seasonal_map.py — were
    # being computed and DROPPED at save) ──
    'session_phase',
    'time_since_london_open_min', 'time_to_london_close_min',
    'time_since_ny_open_min', 'time_to_ny_close_min',
    'dow_sin', 'dow_cos', 'is_monday', 'is_friday',
    'dom', 'dom_sin', 'dom_cos',
    'is_month_end', 'is_month_start',
    'is_quarter_end', 'is_year_end',
    'woy_sin', 'woy_cos', 'is_first_week_of_year',
    'is_dst_transition_week', 'is_event_window',
    # ── Price Cycle Structural (12 features from
    # modules/price_cycle/feature_pipeline.py — same issue) ──
    'cycle_structure_score', 'cycle_bars_since_swing',
    'cycle_trend_maturity', 'cycle_momentum_decay',
    'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
    'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
    'cycle_position', 'cycle_hurst',
    'cycle_fractal_dim', 'cycle_mtf_alignment',
)

# Alias للمانفيست والعقود — نفس ORIGINAL_FEATURES فقط
DAY_TRADING_FEATURES: list[str] = list(ORIGINAL_FEATURES)

# أعمدة تسمية / ميتاداتا تُحفظ مع parquet ولا تُعدّ من ضمن «فيتشرز النموذج» أعلاه
DAY_TRADE_PARQUET_METADATA_COLS: tuple[str, ...] = (
    'ts_event', 'label_end_ts', 'bias_label', 'signal_quality', 'forward_return',
    'event_flag', 'train_event_flag', 'soft_label', 'label_confidence', 'soft_sample_weight',
    'is_event', 'event_score', 'event_direction', 'path_outcome', 'trade_duration',
    'neutral_reason', 'kalman_direction', 'regime_label', 'regime_cluster',
    'dataset_slice', 'is_train_slice', 'is_holdout_slice', 'is_purged_slice',
    'price', 'open', 'high', 'low', 'close', 'volume',
    'effective_horizon', 'label_horizon_steps', 'fwd_ret_clean',
    'adverse_path_flag', 'bias_label_detail', 'timeout_move_exceeded_band',
    'label_dynamic_threshold', 'soft_label_long', 'soft_label_short',
    'mc_sample_weight', 'label_stability',
)

# ─── تعبئة فراغات advisor على الشمعة بأسماء مطابقة للأساس (بدون micro_price_rel / pdh_rel) ──
CATBOOST_ADVISOR_FEATURES_DT = [
    'cvd', 'obi', 'absorption_intensity', 'cancel_ratio',
    'spoofing_ratio', 'spoofing_duration', 'liquidity_trap',
    'micro_atr', 'volume_burst', 'inter_event_time',
    'micro_price', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    'fisher_signal', 'anomaly',
    'cvd_momentum', 'cvd_price_divergence',
    'trend_strength', 'correction_depth', 'liquidity_sweep',
    'pdh', 'pdl', 'dist_to_pdh', 'price_position',
    'dist_to_london_high_atr', 'dist_to_london_low_atr',
    'kyle_lambda', 'hawkes_intensity', 'vnet',
    'hawkes_intrabar_sum', 'kyle_lambda_intrabar_mean', 'vnet_intrabar_last',
    'vwap_z_score',
]

TRADE_ACTIONS = {'T', 'F', 'TRADE', 'EXECUTE', 'E', '0'}
# Aggressor side convention for this feed:
# - A / ASK / BUY -> buy-initiated (lifting ask)
# - B / BID / SELL -> sell-initiated (hitting bid)
BUY_SIDES = {'A', 'ASK', 'BUY', 'BOT'}
SELL_SIDES = {'B', 'BID', 'S', 'SELL'}
CORE_MBO_REQUIRED_COLS = [
    'cvd',
    'session_cvd',
    'absorption_intensity',
    'cancel_ratio',
    'micro_atr',
    'volume_burst',
    'inter_event_time',
    'fisher_signal',
    'anomaly',
    'cvd_momentum',
    'cvd_price_divergence',
    'trend_strength',
    'correction_depth',
    'liquidity_sweep',
    'kyle_lambda',
    'hawkes_intensity',
    'vnet',
    'current_vwap',
    'vwap_z_score',
]


def finalize_daytrade_parquet_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    يقيّد ملف day_trading_features.parquet إلى ORIGINAL_FEATURES + أعمدة التسميات/الميتادات.
    أي عمود مشتط خارج هذه القائمة يُسقَط من التصدير (قد يبقى في preflight فقط).

    Audit guard (Issue #24): emit a loud warning when an expected ORIGINAL_FEATURE
    is missing from the input. Previously such columns were silently filled with
    zeros — if e.g. add_seasonal_features raised, the model would train on
    21 zero columns without anyone noticing.
    """
    out = df.copy()
    missing_before_fill: list[str] = []
    for col in ORIGINAL_FEATURES:
        if col not in out.columns:
            missing_before_fill.append(col)
            out[col] = np.float64(0.0)
    if missing_before_fill:
        n = len(missing_before_fill)
        sample = missing_before_fill[:8]
        print(
            f"  ⚠️  finalize: {n} ORIGINAL_FEATURES missing — filling with zeros.\n"
            f"     Examples: {sample}\n"
            f"     Likely cause: an upstream feature builder raised silently. "
            f"Check the pipeline log for warnings (seasonal_map/price_cycle/...)."
        )
    feat_ordered = [c for c in ORIGINAL_FEATURES if c in out.columns]
    meta_ordered = [c for c in DAY_TRADE_PARQUET_METADATA_COLS if c in out.columns and c not in set(feat_ordered)]
    return out[feat_ordered + meta_ordered].copy()


def inspect_daytrade_parquet_schema(path: str) -> None:
    """
    ملخص بنية parquet المتوقعة لمسار اليوم: عدد الصفوف والأعمدة، وتغطية ORIGINAL_FEATURES، دون تحميل كامل للجدول إن أمكن.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    print(f"📄 Day-trade parquet schema — {path}")

    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        md = pf.metadata
        names = list(pf.schema_arrow.names)
        print(f"   rows={md.num_rows:,} | n_columns={len(names)} | row_groups={md.num_row_groups}")
        feat_here = [c for c in ORIGINAL_FEATURES if c in names]
        meta_here = [c for c in DAY_TRADE_PARQUET_METADATA_COLS if c in names]
        print(f"   ORIGINAL_FEATURES: {len(feat_here)}/{len(ORIGINAL_FEATURES)} أعمدة موجودة")
        print(f"   ميتاداتا التسمية: {len(meta_here)}/{len(DAY_TRADE_PARQUET_METADATA_COLS)} أعمدة موجودة")
        known = set(ORIGINAL_FEATURES) | set(DAY_TRADE_PARQUET_METADATA_COLS)
        extra = [c for c in names if c not in known]
        if extra:
            print(f"   أعمدة خارج عقد تصدير اليوم: {len(extra)} (مثال أول 12: {extra[:12]}{' …' if len(extra) > 12 else ''})")
        missing_f = [c for c in ORIGINAL_FEATURES if c not in names]
        if missing_f:
            head = missing_f[:40]
            print(f"   ⚠️ أعمدة فيض ناقصة ({len(missing_f)}): {head}{' …' if len(missing_f) > 40 else ''}")
        for key in ('ts_event', 'forward_return', 'absorption_intensity', 'hawkes_intensity', 'bias_label'):
            print(f"   [{key}] {'✓' if key in names else '✗ ناقص'}")
        return
    except Exception as exc:
        print(f"   (تخطّي pyarrow metadata: {exc})")

    df = pd.read_parquet(path)
    names = list(df.columns)
    print(f"   rows={len(df):,} | n_columns={len(names)} (تم التحميل بالكامل)")
    feat_here = [c for c in ORIGINAL_FEATURES if c in names]
    print(f"   ORIGINAL_FEATURES: {len(feat_here)}/{len(ORIGINAL_FEATURES)} أعمدة موجودة")
    known = set(ORIGINAL_FEATURES) | set(DAY_TRADE_PARQUET_METADATA_COLS)
    extra = [c for c in names if c not in known]
    if extra:
        print(f"   أعمدة خارج عقد تصدير اليوم: {len(extra)} (مثال أول 12: {extra[:12]}{' …' if len(extra) > 12 else ''})")
    for key in ('ts_event', 'forward_return', 'absorption_intensity', 'hawkes_intensity', 'bias_label'):
        print(f"   [{key}] {'✓' if key in names else '✗ ناقص'}")


def _resolve_mbo_size_column(df_mbo: pd.DataFrame) -> str:
    for candidate in ('size', 'qty', 'volume'):
        if candidate in df_mbo.columns:
            return candidate
    raise KeyError("MBO data must include one of size/qty/volume")


def _enrich_mbo_run_loop(
    price_arr: np.ndarray,
    size_arr: np.ndarray,
    action_arr: np.ndarray,
    side_arr: np.ndarray,
    oid_arr: np.ndarray,
    ts_ns_arr: np.ndarray,
    ts_pd_arr,  # pd.DatetimeIndex or np.ndarray[datetime64]
    day_idx_arr: np.ndarray,
    min_price_move: float,
    volatility: float,
) -> dict:
    """Pure loop over MBO ticks for one chunk (one day in parallel mode، الكامل في sequential).

    تُنشئ مَحركات جديدة، تجري التحديث لكل tick، وتُعيد dict من numpy arrays
    بالطول نفسه. لا تعتمد على pandas — لكي تُستدعى من worker processes أو inline.
    """
    from modules.microstructure import FastMicrostructureEngine, AbsorptionIntensityEngine, CancelRatioEngine
    from modules.micro_volatility import MicroVolatilityEngine
    from modules.market_research_features import KylesLambdaEngine, HawkesIntensityEngine, VNETEngine
    from modules.context_features import MomentumContextEngine, LiquiditySweepDetector
    from modules.session_features import SessionVWAPEngine
    from modules.fisher_alpha import FastFisherAlpha
    from modules.fim_anomaly import FastFIMDetector

    micro = FastMicrostructureEngine()
    absorb = AbsorptionIntensityEngine(min_price_move=max(float(min_price_move), 1e-8))
    cancel = CancelRatioEngine()
    mv = MicroVolatilityEngine()
    kyle = KylesLambdaEngine(window=50)
    hawkes = HawkesIntensityEngine()
    vnet = VNETEngine()
    fisher = FastFisherAlpha()
    fim = FastFIMDetector()
    momentum = MomentumContextEngine()
    sweep = LiquiditySweepDetector(
        sweep_threshold=max(0.05, min(float(volatility) * 2.0, 0.30)),
    )
    vwap = SessionVWAPEngine()

    n = len(price_arr)
    cvd_arr = np.zeros(n, dtype=np.float64)
    sess_cvd_arr = np.zeros(n, dtype=np.float64)
    absorb_arr = np.zeros(n, dtype=np.float64)
    cancel_arr = np.zeros(n, dtype=np.float64)
    micro_atr_arr = np.zeros(n, dtype=np.float64)
    vol_burst_arr = np.zeros(n, dtype=np.float64)
    iet_arr = np.zeros(n, dtype=np.float64)
    fisher_arr = np.zeros(n, dtype=np.float64)
    anomaly_arr = np.zeros(n, dtype=np.float64)
    cvd_mom_arr = np.zeros(n, dtype=np.float64)
    cvd_div_arr = np.zeros(n, dtype=np.float64)
    trend_arr = np.zeros(n, dtype=np.float64)
    corr_arr = np.zeros(n, dtype=np.float64)
    sweep_arr = np.zeros(n, dtype=np.float64)
    kyle_arr = np.zeros(n, dtype=np.float64)
    hawkes_arr = np.zeros(n, dtype=np.float64)
    vnet_arr = np.zeros(n, dtype=np.float64)
    vwap_arr = np.zeros(n, dtype=np.float64)
    vwap_z_arr = np.zeros(n, dtype=np.float64)
    spoof_arr = np.zeros(n, dtype=np.float64)

    cvd = 0.0
    last_cancel = 0.0
    last_absorb = 0.0
    last_fisher = 0.0
    last_anomaly = 0.0
    last_cvd_mom = 0.0
    last_cvd_div = 0.0
    last_trend = 0.0
    last_corr = 0.0
    last_kyle = 0.0
    last_vnet = 0.0
    last_vwap = 0.0
    last_vwap_z = 0.0
    last_sess_cvd = 0.0
    last_day_idx = -1

    for i in range(n):
        px = price_arr[i]
        sz = size_arr[i]
        act = action_arr[i]
        sd = side_arr[i]
        oid = oid_arr[i]
        day_idx = day_idx_arr[i]
        if last_day_idx == -1:
            last_day_idx = day_idx
        elif day_idx != last_day_idx:
            # Keep CVD session-local instead of carrying net month bias across days.
            cvd = 0.0
            last_sess_cvd = 0.0
            last_day_idx = day_idx
        ts_ns = int(ts_ns_arr[i])

        spoof_arr[i] = float(micro.process_mbo_tick(act, oid, sd, sz, px, ts_ns))
        cr = float(cancel.process_tick(act, oid, sz))
        if cr != 0.0:
            last_cancel = cr
        cancel_arr[i] = last_cancel
        mu_atr, vol_burst, iet = mv.process_tick(act, px, sz, ts_ns)
        micro_atr_arr[i] = float(mu_atr)
        vol_burst_arr[i] = float(vol_burst)
        iet_arr[i] = float(iet)
        hawkes_arr[i] = float(hawkes.update(ts_ns, act))
        sweep_arr[i] = float(sweep.update(px))

        is_trade = act in TRADE_ACTIONS
        if is_trade:
            is_buy = sd in BUY_SIDES
            is_sell = sd in SELL_SIDES
            if is_buy:
                cvd += sz
            elif is_sell:
                cvd -= sz

            last_absorb = float(absorb.update(px, cvd))
            last_fisher = float(fisher.update_and_get_signal(px, cvd))
            last_anomaly = float(fim.detect_stop_hunts(px))
            last_cvd_mom, last_cvd_div, last_trend, last_corr = momentum.update(px, cvd)
            last_kyle = float(kyle.update(px, sz))
            if is_buy:
                vnet_side = 'B'
            elif is_sell:
                vnet_side = 'A'
            else:
                vnet_side = sd
            last_vnet = float(vnet.update(px, sz, vnet_side))
            last_vwap, last_vwap_z, _, last_sess_cvd = vwap.update(ts_pd_arr[i], px, float(sz), bool(is_buy))

        cvd_arr[i] = cvd
        sess_cvd_arr[i] = float(last_sess_cvd)
        absorb_arr[i] = float(last_absorb)
        fisher_arr[i] = float(last_fisher)
        anomaly_arr[i] = float(last_anomaly)
        cvd_mom_arr[i] = float(last_cvd_mom)
        cvd_div_arr[i] = float(last_cvd_div)
        trend_arr[i] = float(last_trend)
        corr_arr[i] = float(last_corr)
        kyle_arr[i] = float(last_kyle)
        vnet_arr[i] = float(last_vnet)
        vwap_arr[i] = float(last_vwap)
        vwap_z_arr[i] = float(last_vwap_z)

    return {
        'cvd': cvd_arr,
        'session_cvd': sess_cvd_arr,
        'absorption_intensity': absorb_arr,
        'cancel_ratio': cancel_arr,
        'micro_atr': micro_atr_arr,
        'volume_burst': vol_burst_arr,
        'inter_event_time': iet_arr,
        'fisher_signal': fisher_arr,
        'anomaly': anomaly_arr,
        'cvd_momentum': cvd_mom_arr,
        'cvd_price_divergence': cvd_div_arr,
        'trend_strength': trend_arr,
        'correction_depth': corr_arr,
        'liquidity_sweep': sweep_arr,
        'kyle_lambda': kyle_arr,
        'hawkes_intensity': hawkes_arr,
        'vnet': vnet_arr,
        'current_vwap': vwap_arr,
        'vwap_z_score': vwap_z_arr,
        'spoofing_ratio': spoof_arr,
    }


def _enrich_mbo_chunk_worker(args: tuple) -> dict:
    """Pool.map adapter: يفك tuple ويستدعي _enrich_mbo_run_loop."""
    return _enrich_mbo_run_loop(*args)


def enrich_mbo_with_core_microstructure(
    df_mbo: pd.DataFrame,
    *,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Reconstruct core tick-level microstructure features when raw MBO lacks them.
    Uses the same engine families as stage-1 refinery to avoid losing signal quality.

    n_workers: عدد العمليات المتوازية. None = auto (cpu_count - 1)، 1 = sequential.
               override عبر env DD_N_WORKERS. الانقسام بالـ day (آمن: cvd/session_cvd
               يتم reset في حدود اليوم في النسخة sequential أيضاً).
    """
    missing = [c for c in CORE_MBO_REQUIRED_COLS if c not in df_mbo.columns]
    if not missing:
        return df_mbo

    print(
        "  🧠 Core microstructure reconstruction: "
        f"{len(missing)} missing columns -> rebuilding from raw ticks",
    )

    from modules.auto_calibrator import AutoCalibrator

    out = df_mbo.copy()
    out['ts_event'] = pd.to_datetime(out['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    out = out.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    if out.empty:
        return out

    size_col = _resolve_mbo_size_column(out)
    if size_col != 'size':
        out['size'] = pd.to_numeric(out[size_col], errors='coerce').fillna(0.0).astype(np.float64)
    else:
        out['size'] = pd.to_numeric(out['size'], errors='coerce').fillna(0.0).astype(np.float64)
    out['price'] = pd.to_numeric(out['price'], errors='coerce').fillna(0.0).astype(np.float64)
    action_s = out['action'].astype(str).str.upper() if 'action' in out.columns else pd.Series('T', index=out.index)
    side_s = out['side'].astype(str).str.upper() if 'side' in out.columns else pd.Series('', index=out.index)
    order_ids = out['order_id'] if 'order_id' in out.columns else pd.Series([None] * len(out), index=out.index)

    calibrator = AutoCalibrator(n_ticks=2000).fit(out, price_col='price', size_col='size', action_col='action')
    min_price_move = float(calibrator.min_price_move)
    volatility = float(calibrator.volatility)

    # ── Cache numpy arrays + precompute ts/day vectorized
    _price_arr = out['price'].to_numpy(dtype=np.float64)
    _size_arr = out['size'].to_numpy(dtype=np.float64)
    _action_arr = np.asarray([s.strip().upper() for s in action_s.astype(str).to_numpy()], dtype=object)
    _side_arr = np.asarray([s.strip().upper() for s in side_s.astype(str).to_numpy()], dtype=object)
    _oid_arr = order_ids.to_numpy() if hasattr(order_ids, 'to_numpy') else np.asarray(order_ids)
    _ts_dt = pd.to_datetime(out['ts_event'].to_numpy())
    _ts_ns_arr = _ts_dt.astype('datetime64[ns]').astype(np.int64)
    _day_idx_arr = (_ts_ns_arr // (86_400 * 1_000_000_000)).astype(np.int64)
    _ts_pd_arr = pd.DatetimeIndex(_ts_dt)

    # ── قرار التوازي: env override > arg > auto
    env_workers = os.environ.get('DD_N_WORKERS', '').strip()
    if env_workers:
        try:
            n_workers = max(1, int(env_workers))
        except ValueError:
            pass
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    unique_days = np.unique(_day_idx_arr)
    n_days = int(len(unique_days))
    effective_workers = min(n_workers, n_days) if n_days > 0 else 1

    if effective_workers <= 1 or n_days <= 1:
        # ── Sequential path (نفس السلوك القديم)
        print(f"  ⏳ enrich loop: sequential (n_ticks={len(out):,}, days={n_days})")
        rebuilt_cols = _enrich_mbo_run_loop(
            _price_arr, _size_arr, _action_arr, _side_arr, _oid_arr,
            _ts_ns_arr, _ts_pd_arr, _day_idx_arr,
            min_price_move, volatility,
        )
    else:
        # ── Parallel path: نقسم على الأيام، نعالج كل يوم في worker process
        import multiprocessing as mp
        print(
            f"  ⚡ enrich loop: parallel (workers={effective_workers}, "
            f"n_ticks={len(out):,}, days={n_days})"
        )
        # Build chunk args per day (preserve order)
        chunk_args: list[tuple] = []
        for d in unique_days:
            mask = (_day_idx_arr == d)
            chunk_args.append((
                _price_arr[mask],
                _size_arr[mask],
                _action_arr[mask],
                _side_arr[mask],
                _oid_arr[mask],
                _ts_ns_arr[mask],
                _ts_pd_arr[mask],  # DatetimeIndex slicing keeps as DatetimeIndex
                _day_idx_arr[mask],
                min_price_move,
                volatility,
            ))
        # fork on Linux inherits modules — no re-import overhead
        ctx = mp.get_context('fork')
        with ctx.Pool(effective_workers) as pool:
            chunk_results = pool.map(_enrich_mbo_chunk_worker, chunk_args)
        # Concatenate per-column في ترتيب الأيام (which matches original sequential)
        rebuilt_cols = {}
        for col in chunk_results[0].keys():
            rebuilt_cols[col] = np.concatenate([r[col] for r in chunk_results])

    for col, arr in rebuilt_cols.items():
        if col not in df_mbo.columns:
            out[col] = pd.Series(arr, index=out.index).astype(np.float64)

    print(
        "  ✅ Core microstructure reconstructed "
        f"(rows={len(out):,} | rebuilt_cols={sum(1 for c in rebuilt_cols if c not in df_mbo.columns)})",
    )
    return out


def _bar_period_seconds(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return max(float(td.total_seconds()), 1e-6)


def _bars_for_target_minutes(freq: str, target_minutes: int) -> int:
    """كم شمعة تغطي تقريباً target_minutes دقيقة لهذا الشمع الأساس (يدعم 5min و 1h وغيره عبر pandas)."""
    bar_minutes = _bar_period_seconds(freq) / 60.0
    return max(1, int(round(float(target_minutes) / max(bar_minutes, 1e-9))))


def _sanitize_bars_extreme_close_returns(df: pd.DataFrame, max_abs_pct_ret: float | None) -> pd.DataFrame:
    """إزالة الشمعات التي يكون فيها إغلاق الشمعة منحنياً جداً مقارنة بالافتتاح (عادةً أخطاء تصفير/قفزة نادراً في البيانات الخام)."""
    if max_abs_pct_ret is None or df.empty or 'open' not in df.columns or 'close' not in df.columns:
        return df
    mx = float(max_abs_pct_ret)
    if mx <= 0 or mx >= 1.0:
        return df
    op = pd.to_numeric(df['open'], errors='coerce').astype(float)
    cl = pd.to_numeric(df['close'], errors='coerce').astype(float)
    base = op.where(op.abs() > 1e-12, np.nan)
    ret = (cl - op) / base
    bad = ret.abs() > mx
    n = int(bad.sum())
    if n:
        print(f"  🧹 sanitize bars: dropped {n:,} rows with |close−open|/|open| > {mx:.6g}")
        df = df.loc[~bad.fillna(False)].copy()
    return df


def add_rolling_vwap_dist_features(
    df: pd.DataFrame,
    *,
    window_bars: int,
    anchor_col: str,
    dist_col: str,
    atr_col: str = 'atr_14',
) -> pd.DataFrame:
    """
    VWAP سببي على typical price × volume لآخر window_bars شمعة انتهت قبل الشمعة الحالية.

    - لا يدخل سعر/حجم الشمعة الحالية في المرجع (shift(1) ثم rolling).
    - المسافة الطبيعية: (close − VWAP المرجعي) / (atr + eps) لتكون مقارنة على مقياس تقلب الشمعة.

    مثال: freq=5min و window_bars من _bars_for_target_minutes(freq, 60) → نحو ساعة تقويمية من الشموع السابقة.
    """
    out = df.copy()
    win = max(int(window_bars), 1)
    h = pd.to_numeric(out['high'], errors='coerce').astype(np.float64)
    l = pd.to_numeric(out['low'], errors='coerce').astype(np.float64)
    c = pd.to_numeric(out['close'], errors='coerce').astype(np.float64)
    v = pd.to_numeric(out['volume'], errors='coerce').astype(np.float64).clip(lower=0.0)
    typical = (h + l + c) / 3.0
    pv = typical * v
    pv_sum = pv.shift(1).rolling(win, min_periods=win).sum()
    v_sum = v.shift(1).rolling(win, min_periods=win).sum()
    anchor = pv_sum / v_sum.clip(lower=1e-12)
    out[anchor_col] = anchor.astype(np.float64)
    atr = pd.to_numeric(out.get(atr_col, 0.0), errors='coerce').astype(np.float64).clip(lower=1e-12)
    out[dist_col] = ((c - anchor) / (atr + 1e-8)).astype(np.float64)
    return out


def _mbp_required_columns(levels: int = 10) -> list[str]:
    cols = ['ts_event']
    for prefix in ('bid_px', 'ask_px', 'bid_sz', 'ask_sz'):
        cols.extend([f'{prefix}_{i:02d}' for i in range(int(levels))])
    return cols


_MBP_OPTIONAL_META_COLS: tuple[str, ...] = (
    'symbol',
    'instrument_id',
    'sequence',
    'seq',
    'sequence_id',
    'order_id',
)


def _mbp_columns_for_load(levels: int = 10) -> list[str]:
    """أعمدة MBP للتحميل: مستويات الدفتر + حقول تعريف اختيارية إن وُجدت في الملف."""
    return list(dict.fromkeys(_mbp_required_columns(levels=levels) + list(_MBP_OPTIONAL_META_COLS)))


def _read_mbp_table(path: str, *, levels: int = 10) -> pd.DataFrame:
    """Load only MBP columns needed by DayTrade to avoid OOM on large CSVs."""
    wanted = _mbp_columns_for_load(levels=levels)

    def _existing_csv_cols(file_path: str) -> list[str]:
        header = pd.read_csv(file_path, nrows=0, compression='infer')
        return [c for c in wanted if c in set(header.columns)]

    def _existing_parquet_cols(file_path: str) -> list[str]:
        try:
            import pyarrow.parquet as pq  # type: ignore
            names = set(pq.ParquetFile(file_path).schema.names)
        except Exception:
            names = set(pd.read_parquet(file_path).columns)
        return [c for c in wanted if c in names]

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.csv', '.gz', '.zst'):
            cols = _existing_csv_cols(path)
            if 'ts_event' not in cols:
                raise ValueError(f"❌ MBP file missing ts_event: {path}")
            return pd.read_csv(path, usecols=cols, low_memory=False, compression='infer')
        if ext in ('.parquet', '.pq', '.snappy'):
            cols = _existing_parquet_cols(path)
            if 'ts_event' not in cols:
                raise ValueError(f"❌ MBP parquet missing ts_event: {path}")
            return pd.read_parquet(path, columns=cols)
        raise FileNotFoundError(f"❌ صيغة ملف MBP غير مدعومة: {path}")

    files = sorted(glob.glob(os.path.join(path, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"❌ لا توجد ملفات parquet في: {path}")
    chunks = []
    for file_path in files:
        cols = _existing_parquet_cols(file_path)
        if 'ts_event' not in cols:
            continue
        chunks.append(pd.read_parquet(file_path, columns=cols))
    if not chunks:
        raise FileNotFoundError(f"❌ لا توجد MBP parquet صالحة تحتوي ts_event في: {path}")
    return pd.concat(chunks, ignore_index=True)


def _session_time_mask(times: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    start_t = pd.to_datetime(start_hhmm).time()
    end_t = pd.to_datetime(end_hhmm).time()
    if start_t <= end_t:
        return (times >= start_t) & (times < end_t)
    return (times >= start_t) | (times < end_t)


def _set_session_profile(profile: str) -> None:
    global SESSIONS
    if profile not in SESSION_PROFILES:
        known = ', '.join(sorted(SESSION_PROFILES.keys()))
        raise ValueError(f"❌ Unknown session profile '{profile}'. Available: {known}")
    SESSIONS = dict(SESSION_PROFILES[profile])


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    s = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return s.astype(np.float64)


def _combine_first_valid(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    p = primary.astype(np.float64)
    f = fallback.reindex(primary.index).astype(np.float64)
    pv = p.to_numpy()
    fv = f.to_numpy()
    ok = np.isfinite(pv)
    out = np.where(ok, pv, fv)
    return pd.Series(out, index=p.index, dtype=np.float64)


def add_london_session_running_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    مستويات نافذة لندن (``SESSIONS['london']``) بشكل **سببي**:

    - أثناء لندن: قمة/قاع **جاريان** حتى إغلاق الشمعة الحالية.
    - بعد انتهاء لندن في نفس اليوم: يُجمَّدان على آخر قمة/قاع وُصلا إليهما في الجلسة.
    - قبل لندن: لا توجد قمة/قاع بعد؛ نعبّئ بالـ ``close`` حتى لا نُدخل NaN في التصدير، والمسافة تصبح ~0.

    المسافات تقسيم على ``atr_14`` (+ eps) لتتوافق مع قواعد هجينة مثل (امتصاص + مسافة لندن + obi).
    """
    out = df.copy()
    if len(out) == 0:
        out['london_sess_high'] = np.float64(np.nan)
        out['london_sess_low'] = np.float64(np.nan)
        out['dist_to_london_high_atr'] = np.float64(0.0)
        out['dist_to_london_low_atr'] = np.float64(0.0)
        return out

    ts = pd.to_datetime(out['ts_event'], errors='coerce')
    t = ts.dt.time
    london_start, london_end = SESSIONS['london']
    in_lon = _session_time_mask(t, london_start, london_end).to_numpy(dtype=bool)
    day = ts.dt.normalize().to_numpy()

    high = pd.to_numeric(out['high'], errors='coerce').to_numpy(dtype=np.float64)
    low = pd.to_numeric(out['low'], errors='coerce').to_numpy(dtype=np.float64)
    close_s = pd.to_numeric(out['close'], errors='coerce')

    nh = np.full(len(out), np.nan, dtype=np.float64)
    nl = np.full(len(out), np.nan, dtype=np.float64)

    idx = 0
    n = len(out)
    while idx < n:
        j = idx + 1
        while j < n and day[j] == day[idx]:
            j += 1
        cur_h = np.nan
        cur_l = np.nan
        for k in range(idx, j):
            if in_lon[k]:
                cur_h = high[k] if not np.isfinite(cur_h) else max(cur_h, high[k])
                cur_l = low[k] if not np.isfinite(cur_l) else min(cur_l, low[k])
                nh[k] = cur_h
                nl[k] = cur_l
            elif np.isfinite(cur_h) and np.isfinite(cur_l):
                nh[k] = cur_h
                nl[k] = cur_l
        idx = j

    eps = 1e-12
    atr_s = pd.to_numeric(out.get('atr_14', np.nan), errors='coerce').clip(lower=eps)
    ser_h = pd.Series(nh, index=out.index).fillna(close_s).astype(np.float64)
    ser_l = pd.Series(nl, index=out.index).fillna(close_s).astype(np.float64)
    out['london_sess_high'] = ser_h
    out['london_sess_low'] = ser_l
    out['dist_to_london_high_atr'] = ((ser_h - close_s.astype(np.float64)) / atr_s.astype(np.float64)).replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.0).astype(np.float64)
    out['dist_to_london_low_atr'] = ((close_s.astype(np.float64) - ser_l) / atr_s.astype(np.float64)).replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.0).astype(np.float64)
    return out


def _tick_resample_optional(
    df: pd.DataFrame, freq: str, col: str, how: str = "mean",
) -> pd.Series | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if how == "mean":
        out = s.resample(freq).mean()
    elif how == "max":
        out = s.resample(freq).max()
    elif how == "sum":
        out = s.resample(freq).sum()
    elif how == "last":
        out = s.resample(freq).last()
    else:
        out = s.resample(freq).mean()
    return out


def apply_bar_level_catboost_parities(df_bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    يضمن ظهور أعمدة الـ advisor الـ31 لـ train_v19؛ يكمِّل ما نجمّعه من التيكات بتقريبات bar-level
    عند النقص فقط (لا يستبدل إشارة المصفاة الصالحة).
    """
    df = df_bars.copy()
    bar_sec = float(_bar_period_seconds(freq))

    close = _num_series(df, 'close')
    high = _num_series(df, 'high')
    low = _num_series(df, 'low')
    open_ = _num_series(df, 'open')
    vol = _num_series(df, 'volume')
    br = _num_series(df, 'bar_range')
    br = _combine_first_valid(br, (high - low).abs())

    bv = _num_series(df, 'buy_volume')
    if 'buy_vol' in df.columns:
        bv = _combine_first_valid(bv, _num_series(df, 'buy_vol'))
    sv = _num_series(df, 'sell_volume')
    tot_flow = (bv + sv).replace(0, np.nan)
    ofi = ((bv - sv) / tot_flow).clip(-1.0, 1.0).fillna(0.0)

    buy_ratio = _num_series(df, 'buy_ratio').clip(0.0, 1.0)
    tick_n = _num_series(df, 'tick_count').clip(lower=1.0)
    if 'num_trades' in df.columns:
        tick_n = _combine_first_valid(tick_n, _num_series(df, 'num_trades').clip(lower=1.0))

    atr = _num_series(df, 'atr_14').clip(lower=1e-12)
    micro_atr = _num_series(df, 'micro_atr').clip(lower=1e-12)

    cr = _num_series(df, 'cancel_ratio').clip(0.0, 1.0)
    abs_i = _num_series(df, 'absorption_intensity').clip(0.0, 1.0)

    # مستويات اليوم (نفس prepare_training_data / context_features)
    if 'ts_event' in df.columns:
        levels = df[['ts_event', 'close']].copy()
        levels = levels.rename(columns={'close': 'price'})
        try:
            ctx = compute_daily_weekly_levels(levels, price_col='price', ts_col='ts_event')
            df['pdh'] = ctx['pdh'].to_numpy()
            df['pdl'] = ctx['pdl'].to_numpy()
            df['dist_to_pdh'] = ctx['dist_to_pdh'].to_numpy()
            df['price_position'] = ctx['price_position'].to_numpy()
        except Exception:
            df['pdh'] = close.to_numpy()
            df['pdl'] = close.to_numpy()
            df['dist_to_pdh'] = np.zeros(len(df), dtype=np.float64)
            df['price_position'] = np.full(len(df), 0.5, dtype=np.float64)
    else:
        df['pdh'] = close.to_numpy()
        df['pdl'] = close.to_numpy()
        df['dist_to_pdh'] = np.zeros(len(df), dtype=np.float64)
        df['price_position'] = np.full(len(df), 0.5, dtype=np.float64)

    obi_proxy = ofi
    if 'order_flow_imbalance' in df.columns:
        obi_proxy = _combine_first_valid(
            obi_proxy,
            _num_series(df, 'order_flow_imbalance').clip(-1.0, 1.0),
        )
    if 'lob_imbalance' in df.columns:
        obi_proxy = _combine_first_valid(obi_proxy, _num_series(df, 'lob_imbalance').clip(-1.0, 1.0))

    spoof_proxy = (cr * 0.85).clip(0.0, 1.0)
    dur_proxy = pd.Series(bar_sec * np.clip(cr.to_numpy(), 0.0, 1.0), index=df.index, dtype=np.float64)
    trap_proxy = (abs_i * cr * 2.0).clip(0.0, 1.0)

    iet_arr = bar_sec / np.maximum(tick_n.to_numpy(), 1.0)
    iet_proxy = pd.Series(np.clip(iet_arr, 0.0, 1e6), index=df.index, dtype=np.float64)

    mid = (high + low + close) / 3.0
    mp = _num_series(df, 'micro_price')
    mp_f = _combine_first_valid(mp, mid)
    mp_f = _combine_first_valid(mp_f, close)

    wall_bid = (buy_ratio * 2.0).clip(0.0, 2.0)
    wall_ask = ((1.0 - buy_ratio) * 2.0).clip(0.0, 2.0)
    dist_wall = (close - mp_f).abs() / micro_atr.replace(0, np.nan)
    dist_wall = dist_wall.replace([np.inf, -np.inf], np.nan)

    prev_close = close.shift(1)
    gap = (open_ - prev_close).abs() / atr.replace(0, np.nan)
    gap = gap.replace([np.inf, -np.inf], np.nan)

    denom = br.replace(0, np.nan) + micro_atr * 1e-3
    liq_dens = vol / denom
    liq_dens = liq_dens.replace([np.inf, -np.inf], np.nan)

    def _assign_advisor(name: str, primary: pd.Series, fallback: pd.Series) -> None:
        s = _combine_first_valid(primary, fallback)
        df[name] = s

    _assign_advisor('obi', _num_series(df, 'obi'), obi_proxy)
    _assign_advisor('spoofing_ratio', _num_series(df, 'spoofing_ratio'), spoof_proxy)
    _assign_advisor('spoofing_duration', _num_series(df, 'spoofing_duration'), dur_proxy)
    _assign_advisor('liquidity_trap', _num_series(df, 'liquidity_trap'), trap_proxy)
    _assign_advisor('inter_event_time', _num_series(df, 'inter_event_time'), iet_proxy)
    _assign_advisor('micro_price', _num_series(df, 'micro_price'), mp_f)
    _assign_advisor('bid_wall_strength', _num_series(df, 'bid_wall_strength'), wall_bid)
    _assign_advisor('ask_wall_strength', _num_series(df, 'ask_wall_strength'), wall_ask)
    _assign_advisor('distance_to_wall', _num_series(df, 'distance_to_wall'), dist_wall)
    _assign_advisor('gap_size', _num_series(df, 'gap_size'), gap)
    _assign_advisor('liquidity_density', _num_series(df, 'liquidity_density'), liq_dens)

    df['obi'] = df['obi'].clip(-1.0, 1.0)
    df['spoofing_ratio'] = df['spoofing_ratio'].clip(0.0, 1.0)
    df['price_position'] = pd.to_numeric(df['price_position'], errors='coerce').clip(0.0, 1.0).fillna(0.5)

    # مستويات سعر مطلقة تُبقى للاشتقاق وبعض القياسات؛ CatBoost يستخدم *_rel
    for col in ('micro_price', 'pdh', 'pdl'):
        if col not in df.columns:
            df[col] = np.nan
        raw_lvl = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        df[col] = _combine_first_valid(raw_lvl, close).astype(np.float64)

    atr14 = pd.to_numeric(df['atr_14'], errors='coerce').astype(np.float64).clip(lower=1e-12)
    close_num = pd.to_numeric(df['close'], errors='coerce').astype(np.float64)
    eps_atr = 1e-8
    mp_abs = pd.to_numeric(df['micro_price'], errors='coerce').astype(np.float64)
    pdh_abs = pd.to_numeric(df['pdh'], errors='coerce').astype(np.float64)
    pdl_abs = pd.to_numeric(df['pdl'], errors='coerce').astype(np.float64)
    df['micro_price_rel'] = ((mp_abs - close_num) / (atr14 + eps_atr)).astype(np.float64)
    df['pdh_rel'] = ((pdh_abs - close_num) / (atr14 + eps_atr)).astype(np.float64)
    df['pdl_rel'] = ((pdl_abs - close_num) / (atr14 + eps_atr)).astype(np.float64)

    for col in CATBOOST_ADVISOR_FEATURES_DT:
        if col not in df.columns:
            df[col] = np.nan
        raw = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if col in ('dist_to_pdh',):
            df[col] = raw.fillna(0.0).astype(np.float64)
        elif col == 'price_position':
            df[col] = raw.clip(0.0, 1.0).fillna(0.5).astype(np.float64)
        else:
            df[col] = raw.fillna(0.0).astype(np.float64)

    return df


def attach_hybrid_liquidity_bridge(
    bars: pd.DataFrame,
    df_ticks: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    يضغط معلومات «صانع السوق» من التكات داخل كل شمعة إلى أعمدة صريحة
    (مكملة لـ OHLCV والميزات المُجمَّعة من المصفاة).

    ملاحظة: في MBO الجانب الشرائي = 'A' والبيعي = 'B' (ليس buy/sell نصاً).
    """
    out = bars.copy()
    tc = (
        pd.to_numeric(out['tick_count'], errors='coerce').fillna(0.0).astype(np.float64).clip(lower=1.0)
    )
    vol = pd.to_numeric(out['volume'], errors='coerce').fillna(0.0).astype(np.float64)
    out['num_trades'] = tc
    out['avg_trade_size'] = (vol / tc).replace([np.inf, -np.inf], 0).fillna(0.0)
    # عمود قديم: نفس buy_ratio المكمّم — النمذجة تستخدم absorption_intensity (AII من التكات).
    out['absorption_bar'] = (
        pd.to_numeric(out['buy_ratio'], errors='coerce').fillna(0.5).astype(np.float64).clip(0.0, 1.0)
    )
    bv = pd.to_numeric(out['buy_volume'], errors='coerce').fillna(0.0).astype(np.float64)
    sv = pd.to_numeric(out['sell_volume'], errors='coerce').fillna(0.0).astype(np.float64)
    sums = bv.to_numpy(dtype=np.float64) + sv.to_numpy(dtype=np.float64)
    tot = pd.Series(np.maximum(sums, 1e-9), index=out.index)
    out['order_flow_imbalance'] = np.clip((bv - sv) / tot, -1.0, 1.0)
    if 'spread' in df_ticks.columns:
        sp = pd.to_numeric(df_ticks['spread'], errors='coerce').resample(freq).mean()
        out['spread_bar'] = sp.reindex(out.index)
        out['spread_bar'] = pd.to_numeric(out['spread_bar'], errors='coerce').fillna(0.0)
    else:
        out['spread_bar'] = np.zeros(len(out), dtype=np.float32)
    return out


def _derive_order_flow_proxy_from_cvd(
    bar_cvd_delta: pd.Series,
    *,
    roll_window: int = 24,
) -> pd.Series:
    """Normalize bar CVD delta into a bounded [-1, 1] order-flow proxy."""
    cvd = pd.to_numeric(bar_cvd_delta, errors='coerce').fillna(0.0).astype(np.float64)
    ref = cvd.abs().rolling(max(int(roll_window), 2), min_periods=1).median().replace(0.0, np.nan)
    proxy = (cvd / (ref * 3.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return proxy.clip(-1.0, 1.0)


def _add_missing_six_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    فيتشرز سياق سببية (مسافات آسيا، حجم الجلسة، ذيل الشمعة، إلخ).
    لا تُضاف هنا مشتقات مركّبة (فرق إزاحة، ضرب عمودين) — الإشارات الأساسية
    ``hawkes_intensity`` / ``kyle_lambda`` / ``absorption_intensity`` تبقى كما في التجميع من التيكات.
    """
    out = df.copy().sort_values('ts_event').reset_index(drop=True)
    atr = pd.to_numeric(out['atr_14'], errors='coerce').clip(lower=1e-9).astype(np.float64)
    sd = pd.to_datetime(out['ts_event']).dt.normalize()
    ts_time = pd.to_datetime(out['ts_event']).dt.time
    asia_lo, asia_hi = SESSIONS['asia']
    is_asia = _session_time_mask(ts_time, asia_lo, asia_hi).to_numpy(dtype=bool)
    close = pd.to_numeric(out['close'], errors='coerce').astype(np.float64)

    high_num = pd.to_numeric(out['high'], errors='coerce').astype(np.float64)
    low_num = pd.to_numeric(out['low'], errors='coerce').astype(np.float64)
    _ah = np.where(is_asia, high_num.to_numpy(dtype=np.float64, copy=False), np.nan)
    _al = np.where(is_asia, low_num.to_numpy(dtype=np.float64, copy=False), np.nan)
    gdf = pd.DataFrame({'_sd': sd, '_ah': _ah, '_al': _al})

    asia_high = gdf.groupby('_sd', sort=False)['_ah'].transform(lambda s: s.expanding().max().ffill())
    asia_low = gdf.groupby('_sd', sort=False)['_al'].transform(lambda s: s.expanding().min().ffill())

    out['dist_from_asia_high'] = (
        ((close - asia_high.astype(np.float64)) / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    )
    out['dist_from_asia_low'] = (
        ((close - asia_low.astype(np.float64)) / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    )

    if 'session_cvd' in out.columns:
        cvd_s = pd.to_numeric(out['session_cvd'], errors='coerce').astype(np.float64)
        ac = cvd_s.abs()
        abs_max = (
            pd.DataFrame({'sd': sd, 'ac': ac})
            .groupby('sd', sort=False)['ac']
            .transform(lambda s: s.expanding().max().clip(lower=1e-9))
        )
        out['session_cvd_pct'] = (
            (cvd_s / abs_max).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        )
    else:
        out['session_cvd_pct'] = np.float32(0.0)

    vol = pd.to_numeric(out['volume'], errors='coerce').astype(np.float64)
    vol_mean = (
        pd.DataFrame({'sd': sd, 'v': vol})
        .groupby('sd', sort=False)['v']
        .transform(lambda s: s.expanding().mean().clip(lower=1e-9))
    )
    out['volume_vs_session_avg'] = (
        (vol / vol_mean.astype(np.float64)).replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(np.float32)
    )

    o = pd.to_numeric(out['open'], errors='coerce')
    cl = pd.to_numeric(out['close'], errors='coerce')
    body_high = pd.concat([o, cl], axis=1).max(axis=1)
    body_low = pd.concat([o, cl], axis=1).min(axis=1)
    high_s = pd.to_numeric(out['high'], errors='coerce')
    low_s = pd.to_numeric(out['low'], errors='coerce')
    upper_wick = (high_s - body_high).clip(lower=0.0)
    lower_wick = (body_low - low_s).clip(lower=0.0)
    out['wick_ratio'] = (
        ((lower_wick - upper_wick) / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    )

    _new_cols = [
        'dist_from_asia_high',
        'dist_from_asia_low',
        'session_cvd_pct',
        'volume_vs_session_avg',
        'wick_ratio',
    ]
    print(f"  ✅ _add_missing_six_features: {len(_new_cols)} أعمدة مضافة")
    return out


def aggregate_mbo_to_bars(df_mbo: pd.DataFrame, freq: str = DAY_TRADE_DEFAULT_BAR_FREQ) -> pd.DataFrame:
    """
    يجمع التيكات في bars مع الحفاظ على order flow features.

    ملاحظة للامتصاص الهجين: ``absorption_intensity`` هنا هي **قمة التيك داخل الشمعة**
    (تجميع ``max`` على التيكات)، أي مرادف عملي لـ «max absorption داخل إطار الشمعة» (مثلاً دقيقة مع freq=1min).

    Hawkes / Kyle / VNET داخل الشمعة:
    - ``hawkes_intensity``: ``max`` (ذروة العدوى في الشمعة)؛ ``hawkes_intrabar_sum``: ``sum`` على التيكات (تراكيم كثافة داخل الشمعة).
    - ``kyle_lambda``: ``max``؛ ``kyle_lambda_intrabar_mean``: متوسط λ على التيكات داخل الشمعة (أهدأ من الذروة).
    - ``vnet``: ``sum`` صافي التدفق داخل الشمعة؛ ``vnet_intrabar_last``: قيمة VNET عند آخر تيك في الشمعة.
    """
    df = df_mbo.copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.set_index('ts_event').sort_index()

    # يدعم أكثر من اسم لحجم الصفقة في ملفات MBO الخام.
    size_col = 'size'
    if size_col not in df.columns:
        for alt in ('qty', 'volume'):
            if alt in df.columns:
                size_col = alt
                break
    if size_col not in df.columns:
        raise KeyError("aggregate_mbo_to_bars requires one of: size/qty/volume")

    action_s = (
        df['action'].astype(str).str.upper()
        if 'action' in df.columns
        else pd.Series('T', index=df.index, dtype='object')
    )
    side_s = (
        df['side'].astype(str).str.upper()
        if 'side' in df.columns
        else pd.Series('', index=df.index, dtype='object')
    )
    size_s = pd.to_numeric(df[size_col], errors='coerce').fillna(0.0).astype(np.float64)

    # ── B1 fix: OHLC must come from TRADE events only ──
    # Previously bars OHLC were computed from `df['price'].resample(freq)` —
    # i.e. across ALL MBO rows (ADD / MODIFY / CANCEL / TRADE). Each MBO
    # row has a price (= the order's price). `high='max'` therefore
    # returned the highest LIMIT-ORDER price posted in the bar, typically
    # several ticks above last trade. Same for `low`. Bar range / ATR
    # were systematically inflated; MFE/MAE labels used inflated TP/SL
    # distances; regime classification was distorted.
    # Standard quant practice: OHLC of a price bar is computed from
    # TRADE prints only. Volume is sum of TRADE sizes only.
    is_trade_tick = action_s.isin(TRADE_ACTIONS)
    if is_trade_tick.any():
        trade_prices = pd.to_numeric(df.loc[is_trade_tick, 'price'], errors='coerce')
        trade_sizes = size_s.where(is_trade_tick, 0.0)
    else:
        # Pathological feed with no trade actions tagged — fall back to all
        # rows so the pipeline doesn't silently produce empty bars.
        print("⚠️  aggregate_mbo_to_bars: no rows tagged as TRADE action; "
              "OHLC will be derived from all MBO events (legacy behaviour).")
        trade_prices = pd.to_numeric(df['price'], errors='coerce')
        trade_sizes = size_s

    def _resample_num(col: str, how: str, default: float = 0.0) -> pd.Series:
        rs = _tick_resample_optional(df, freq, col, how)
        if rs is None:
            return pd.Series(default, index=bars.index, dtype=np.float64)
        return pd.to_numeric(rs, errors='coerce').reindex(bars.index).fillna(default).astype(np.float64)

    # OHLCV — trades only
    bars = trade_prices.resample(freq).agg(
        open='first', high='max', low='min', close='last'
    )
    bars['volume'] = trade_sizes.resample(freq).sum()
    # Forward-fill any quote-only bars (no trades that interval) — they
    # retain the last trade price for open=high=low=close; volume=0.
    bars[['open', 'high', 'low', 'close']] = bars[['open', 'high', 'low', 'close']].ffill()

    # CVD
    has_tick_cvd = 'cvd' in df.columns
    if has_tick_cvd:
        tick_cvd = pd.to_numeric(df['cvd'], errors='coerce').ffill().fillna(0.0).astype(np.float64)
    else:
        is_trade = action_s.isin(TRADE_ACTIONS)
        # ملاحظة feed المشروع: side='A' يُعامل كـ buy aggressor و'B' كـ sell aggressor.
        is_buy = side_s.isin({'A', 'BUY', 'ASK'})
        is_sell = side_s.isin({'B', 'SELL', 'BID'})
        signed = np.zeros(len(df), dtype=np.float64)
        signed[(is_trade & is_buy).to_numpy()] = size_s[(is_trade & is_buy)].to_numpy(dtype=np.float64)
        signed[(is_trade & is_sell).to_numpy()] = -size_s[(is_trade & is_sell)].to_numpy(dtype=np.float64)
        tick_cvd = pd.Series(signed, index=df.index, dtype=np.float64).cumsum()
    _cvd_rs = tick_cvd.resample(freq)
    _cvd_last = _cvd_rs.last()
    _cvd_first = _cvd_rs.first()
    _cvd_count = _cvd_rs.count()
    bars['cvd'] = _cvd_last.reindex(bars.index).ffill().fillna(0.0)
    _delta_full = (_cvd_last - _cvd_first).where(_cvd_count > 1, 0.0).fillna(0.0)
    bars['bar_cvd_delta'] = _delta_full.reindex(bars.index).fillna(0.0)
    if 'session_cvd' in df.columns:
        bars['session_cvd'] = _resample_num('session_cvd', 'last', 0.0)
    else:
        signed_step = tick_cvd.diff().fillna(tick_cvd)
        session_cvd = signed_step.groupby(signed_step.index.normalize()).cumsum()
        bars['session_cvd'] = session_cvd.resample(freq).last().reindex(bars.index).fillna(0.0)

    # Order Flow (peak + dispersion داخل الشمعة)
    bars['kyle_lambda'] = _resample_num('kyle_lambda', 'max', 0.0)
    bars['hawkes_intensity'] = _resample_num('hawkes_intensity', 'max', 0.0)
    bars['absorption_intensity'] = _resample_num('absorption_intensity', 'max', 0.0)
    bars['cancel_ratio'] = _resample_num('cancel_ratio', 'max', 0.0)
    bars['absorption_std'] = _resample_num('absorption_intensity', 'max', 0.0)
    bars['cancel_std'] = _resample_num('cancel_ratio', 'max', 0.0)
    bars['kyle_std'] = _resample_num('kyle_lambda', 'max', 0.0)
    bars['vnet'] = _resample_num('vnet', 'sum', 0.0)
    bars['hawkes_intrabar_sum'] = _resample_num('hawkes_intensity', 'sum', 0.0)
    bars['kyle_lambda_intrabar_mean'] = _resample_num('kyle_lambda', 'mean', 0.0)
    bars['vnet_intrabar_last'] = _resample_num('vnet', 'last', 0.0)
    bars['volume_burst'] = _resample_num('volume_burst', 'max', 0.0)
    bars['liquidity_sweep'] = _resample_num('liquidity_sweep', 'max', 0.0)
    if 'absorption_intensity' in df.columns:
        bars['absorption_std'] = (
            pd.to_numeric(df['absorption_intensity'], errors='coerce')
            .resample(freq)
            .std()
            .reindex(bars.index)
            .fillna(0.0)
            .astype(np.float64)
        )
    if 'cancel_ratio' in df.columns:
        bars['cancel_std'] = (
            pd.to_numeric(df['cancel_ratio'], errors='coerce')
            .resample(freq)
            .std()
            .reindex(bars.index)
            .fillna(0.0)
            .astype(np.float64)
        )
    if 'kyle_lambda' in df.columns:
        bars['kyle_std'] = (
            pd.to_numeric(df['kyle_lambda'], errors='coerce')
            .resample(freq)
            .std()
            .reindex(bars.index)
            .fillna(0.0)
            .astype(np.float64)
        )

    # VWAP
    bars['vwap_z_score'] = _resample_num('vwap_z_score', 'last', 0.0)
    bars['current_vwap'] = _resample_num('current_vwap', 'last', np.nan)

    # Momentum
    bars['cvd_momentum'] = _resample_num('cvd_momentum', 'last', 0.0)
    bars['cvd_price_divergence'] = _resample_num('cvd_price_divergence', 'last', 0.0)
    bars['trend_strength'] = _resample_num('trend_strength', 'last', 0.0)
    bars['correction_depth'] = _resample_num('correction_depth', 'last', 0.0)

    # Microstructure
    bars['micro_atr'] = _resample_num('micro_atr', 'mean', 0.0)
    bars['micro_atr_max'] = _resample_num('micro_atr', 'max', 0.0)
    bars['fisher_signal'] = _resample_num('fisher_signal', 'last', 0.0)
    bars['anomaly'] = _resample_num('anomaly', 'max', 0.0)

    # Tick VWAP كنقطة أساس لـ micro_price (قبل الفلاتر؛ يكمّله apply_bar_level_catboost_parities لاحقًا)
    turnover = (
        pd.to_numeric(df['price'], errors='coerce').fillna(0)
        * size_s
    ).resample(freq).sum()
    bars['_turn_sum'] = turnover.astype(np.float64)
    vol_f = pd.to_numeric(bars['volume'], errors='coerce').astype(np.float64).clip(lower=1e-9)
    bars['micro_price'] = bars['_turn_sum'] / vol_f
    bars.drop(columns=['_turn_sum'], errors='ignore', inplace=True)

    # أعمدة advisor إن وُجدت على التيك (تُحمَّل من المصفاة قبل التجليع)
    for col, how in (
        ('obi', 'mean'),
        ('spoofing_ratio', 'max'),
        ('spoofing_duration', 'max'),
        ('liquidity_trap', 'max'),
        ('bid_wall_strength', 'mean'),
        ('ask_wall_strength', 'mean'),
        ('distance_to_wall', 'mean'),
        ('gap_size', 'mean'),
        ('liquidity_density', 'mean'),
        ('inter_event_time', 'mean'),
    ):
        rs = _tick_resample_optional(df, freq, col, how)
        if rs is not None:
            bars[col] = rs

    # Buy/Sell
    buy_mask = side_s.isin({'A', 'BUY', 'ASK', '1', '+1', 'LONG', 'L'})
    sell_mask = side_s.isin({'B', 'SELL', 'BID', '-1', 'SHORT', 'SH'})
    buy_vol = size_s[buy_mask].resample(freq).sum()
    sell_vol = size_s[sell_mask].resample(freq).sum()

    # Fallback when side is missing/non-informative (common with stage1 refined artifacts).
    if float(buy_vol.sum() + sell_vol.sum()) <= 0.0:
        if 'vnet' in df.columns:
            vnet_s = pd.to_numeric(df['vnet'], errors='coerce').fillna(0.0).astype(np.float64)
            buy_vol = size_s[vnet_s > 0.0].resample(freq).sum()
            sell_vol = size_s[vnet_s < 0.0].resample(freq).sum()
        if float(buy_vol.sum() + sell_vol.sum()) <= 0.0 and 'obi' in df.columns:
            obi_s = pd.to_numeric(df['obi'], errors='coerce').fillna(0.0).astype(np.float64)
            buy_vol = size_s[obi_s > 0.0].resample(freq).sum()
            sell_vol = size_s[obi_s < 0.0].resample(freq).sum()
        if float(buy_vol.sum() + sell_vol.sum()) <= 0.0:
            px_s = pd.to_numeric(df['price'], errors='coerce').ffill().fillna(0.0).astype(np.float64)
            dpx = px_s.diff().fillna(0.0)
            buy_vol = size_s[dpx >= 0.0].resample(freq).sum()
            sell_vol = size_s[dpx < 0.0].resample(freq).sum()
    buy_vol = pd.to_numeric(buy_vol, errors='coerce').reindex(bars.index).fillna(0.0).astype(np.float64)
    sell_vol = pd.to_numeric(sell_vol, errors='coerce').reindex(bars.index).fillna(0.0).astype(np.float64)
    bars['buy_volume'] = buy_vol
    bars['sell_volume'] = sell_vol
    total = (buy_vol + sell_vol).clip(lower=1.0)
    bars['buy_ratio'] = (buy_vol / total).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    # Tick count
    bars['tick_count'] = df['price'].resample(freq).count()
    bars['mbo_bar_coverage'] = _mbo_bar_coverage_from_tick_series(bars['tick_count'])

    # جسر السيولة الهجين: أسماء صريحة (num_trades, avg_trade_size, …)
    bars = attach_hybrid_liquidity_bridge(bars, df, freq)

    ofi = pd.to_numeric(bars.get('order_flow_imbalance', 0.0), errors='coerce').fillna(0.0).astype(np.float64)
    if float(ofi.abs().sum()) <= 1e-9 or int(ofi.nunique(dropna=True)) <= 1:
        ofi_proxy = _derive_order_flow_proxy_from_cvd(
            pd.to_numeric(bars.get('bar_cvd_delta', 0.0), errors='coerce').fillna(0.0),
            roll_window=max(12, _bars_for_target_minutes(freq, 60)),
        )
        bars['order_flow_imbalance'] = ofi_proxy.astype(np.float32)
        buy_ratio_cur = pd.to_numeric(bars.get('buy_ratio', 0.5), errors='coerce').fillna(0.5).astype(np.float64)
        if float(buy_ratio_cur.std(ddof=0)) <= 1e-6:
            bars['buy_ratio'] = ((ofi_proxy + 1.0) * 0.5).clip(0.0, 1.0).astype(np.float32)

    # ── Velocity / micro-dynamics داخل الشمعة (MBO ticks) ───────────────
    _cnt_safe = _cvd_count.replace(0, 1)
    _velocity_full = (_delta_full / _cnt_safe).where(_cvd_count > 1, 0.0).fillna(0.0)
    bars['cvd_velocity'] = _velocity_full.reindex(bars.index).fillna(0.0)
    bar_d = pd.to_numeric(bars.get('bar_cvd_delta', 0.0), errors='coerce').fillna(0.0)
    bars['cvd_net_direction'] = np.sign(
        bar_d.to_numpy(dtype=np.float64),
    ).astype(np.int8)

    tick_ct = bars['tick_count'].clip(lower=1)
    if 'obi' in df.columns:
        # Vectorized: sign-change events per bar via groupby، بدل Python lambda لكل bar
        _obi_vals = pd.to_numeric(df['obi'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
        if _obi_vals.size >= 2:
            _signs = np.sign(_obi_vals)
            _sign_change = np.zeros(_obi_vals.size, dtype=np.int64)
            _sign_change[1:] = (np.diff(_signs) != 0).astype(np.int64)
            # Mask sign changes that cross bar boundaries (don't count them)
            _grp_key = df.index.floor(freq)
            _new_bar = np.zeros(len(_grp_key), dtype=bool)
            _new_bar[0] = True
            _new_bar[1:] = _grp_key[1:] != _grp_key[:-1]
            _sign_change[_new_bar] = 0
            _rev_series = pd.Series(_sign_change, index=df.index).resample(freq).sum()
            bars['imb_reversals'] = _rev_series.reindex(bars.index).fillna(0.0).astype(np.float64)
        else:
            bars['imb_reversals'] = 0.0
    else:
        bars['imb_reversals'] = 0.0
    obi_base = pd.to_numeric(
        bars['obi'] if 'obi' in bars.columns else bars.get('order_flow_imbalance', 0.0),
        errors='coerce',
    ).fillna(0.0)
    bars['obi_net'] = obi_base.astype(np.float32)
    bars['obi_direction'] = np.sign(obi_base.to_numpy(dtype=np.float64)).astype(np.int8)
    cr_sum = pd.to_numeric(df['cancel_ratio'], errors='coerce').resample(freq).sum() if 'cancel_ratio' in df.columns else pd.Series(
        np.zeros(len(bars)),
        index=bars.index,
        dtype=np.float64,
    )
    bars['cancel_volume_ratio'] = (
        pd.to_numeric(cr_sum, errors='coerce').divide(pd.to_numeric(tick_ct, errors='coerce')).fillna(0.0)
    )

    # حذف bars فارغة
    bars = bars[bars['volume'] > 0].copy()
    bars = bars.reset_index()

    return bars


def add_day_trading_features(df: pd.DataFrame, freq: str = DAY_TRADE_DEFAULT_BAR_FREQ) -> pd.DataFrame:
    """يضيف features تقنية للـ Day Trading.

    vwap_dist: VWAP من التيكات داخل الشمعة الحالية فقط (current_vwap عند إغلاق الشمعة)،
    لا تجمع عدة شموع — مدة الصف = pandas ``freq`` (الافتراضي للمسار ``DAY_TRADE_DEFAULT_BAR_FREQ`` = دقيقة).
    يُضاف أيضاً vwap_dist_1h_roll: VWAP typical×volume سبقي على نافذة تقويمية ≈1h من عدد الشمع المشتق من freq (تأكيد اتجاه للشمع الأدق مثل 1min أو 5min).
    """
    df = df.copy().sort_values('ts_event').reset_index(drop=True)

    # Bar features
    df['bar_range']  = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    df['body_ratio'] = (body / df['bar_range'].clip(lower=1e-8)).clip(0, 1)

    # ── B5 fix: session-gap aware True Range ──
    # GBPUSD futures (6B) trade ~Sun 18:00 ET → Fri 17:00 ET. Friday close →
    # Sunday open is a ~65h gap. Previously |high - close.shift(1)| treated
    # this gap as a single bar interval, inflating ATR at the first Sunday
    # bar by orders of magnitude. Holidays + DST transitions amplify the bug.
    # Detect "session breaks" as inter-bar gaps > 3× the modal gap, and
    # NaN-out the cross-gap True Range components for those bars (the bar's
    # TR falls back to its own H-L, which is the correct local volatility).
    ts = pd.to_datetime(df['ts_event'])
    bar_dt = ts.diff().dt.total_seconds()
    median_dt = float(bar_dt.dropna().median()) if bar_dt.notna().any() else 0.0
    if median_dt > 0:
        # Anything >3x the typical inter-bar spacing is a session gap.
        session_break = (bar_dt > median_dt * 3.0).fillna(False)
    else:
        session_break = pd.Series(False, index=df.index)
    df['is_session_break'] = session_break.astype(np.int8)

    # ── B4 fix: ATR with proper min_periods ──
    # min_periods=1 produced ATR(1)=TR(1), ATR(2)=mean(TR1, TR2), etc., for
    # the first 13 bars after every gap. Downstream code (label_by_outcome,
    # assign_regime_label, TP/SL distances) used these tiny-sample averages
    # as if they were real ATR(14). Now: require full 14 samples; before
    # then ATR is NaN and downstream code must handle it.
    hl   = df['high'] - df['low']
    prev_close = df['close'].shift(1)
    hcp  = (df['high'] - prev_close).abs()
    lcp  = (df['low']  - prev_close).abs()
    # Mask out cross-session-break components: weekend gap pretending to be
    # a 15m move would make TR explode.
    hcp = hcp.where(~session_break, np.nan)
    lcp = lcp.where(~session_break, np.nan)
    tr = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    # Causal rolling ATR(14); NaN for bars 0..12 — must be handled downstream
    df['atr_14'] = tr.rolling(14, min_periods=14).mean()
    # Provide a warm-up fallback so legacy callers don't crash; expanding
    # mean is statistically biased low for small windows but at least
    # finite and clearly distinguishable from the true ATR.
    df['atr_14_warmup'] = tr.expanding(min_periods=1).mean()
    df['atr_14'] = df['atr_14'].fillna(df['atr_14_warmup'])
    df.drop(columns=['atr_14_warmup'], inplace=True, errors='ignore')

    wb_1h = max(2, _bars_for_target_minutes(freq, 60))
    df = add_rolling_vwap_dist_features(
        df, window_bars=wb_1h, anchor_col='vwap_roll_1h', dist_col='vwap_dist_1h_roll'
    )

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.clip(lower=1e-8)
    df['rsi_14'] = (100 - (100 / (1 + rs))).clip(0, 100)

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    nb_short = max(2, _bars_for_target_minutes(freq, 30))
    df['return_6b']       = df['close'].pct_change(nb_short)
    df['volume_ratio_6b'] = df['volume'] / df['volume'].rolling(nb_short, min_periods=1).mean().clip(lower=1)
    df['cvd_slope_6b']    = df['cvd'].diff(nb_short) / float(nb_short)

    cvw = pd.to_numeric(df['current_vwap'], errors='coerce').astype(np.float64).clip(lower=1e-8)
    df['vwap_dist'] = (
        pd.to_numeric(df['close'], errors='coerce').astype(np.float64) - cvw
    ) / cvw

    for label, tgt_min in (('1h', 60), ('4h', 240)):
        nb = max(2, _bars_for_target_minutes(freq, tgt_min))
        df[f'return_{label}'] = df['close'].pct_change(nb)
        df[f'volume_ratio_{label}'] = df['volume'] / df['volume'].rolling(nb, min_periods=1).mean().clip(lower=1)
        df[f'cvd_slope_{label}'] = df['cvd'].diff(nb) / float(nb)

    # Session tags
    t = df['ts_event'].dt.time
    london_start, london_end = SESSIONS['london']
    overlap_start, overlap_end = SESSIONS['overlap']
    ny_start, ny_end = SESSIONS['ny']
    df['is_london'] = _session_time_mask(t, london_start, london_end).astype(np.int8)
    df['is_overlap'] = _session_time_mask(t, overlap_start, overlap_end).astype(np.int8)
    df['is_ny'] = _session_time_mask(t, ny_start, ny_end).astype(np.int8)

    df = add_london_session_running_levels(df)

    # LOB imbalance مُجمَّع (buy pressure) مع fallback robust عند انهيار buy_ratio.
    base_lob = (pd.to_numeric(df.get('buy_ratio', 0.5), errors='coerce') - 0.5) * 2.0
    ofi = pd.to_numeric(df.get('order_flow_imbalance', np.nan), errors='coerce')
    cvd_proxy = _derive_order_flow_proxy_from_cvd(
        pd.to_numeric(df.get('bar_cvd_delta', 0.0), errors='coerce').fillna(0.0),
        roll_window=max(12, _bars_for_target_minutes(freq, 60)),
    )
    if float(base_lob.fillna(0.0).std(ddof=0)) <= 1e-6:
        lob = _combine_first_valid(ofi, cvd_proxy)
    else:
        lob = _combine_first_valid(base_lob, ofi)
        lob = _combine_first_valid(lob, cvd_proxy)
    df['lob_imbalance'] = pd.to_numeric(lob, errors='coerce').fillna(0.0).clip(-1.0, 1.0)

    df = apply_bar_level_catboost_parities(df, freq=freq)
    return df.fillna(0)


def assign_regime_label(
    df: pd.DataFrame,
    *,
    roll_window: int = 100,
    min_periods: int = 20,
    volatile_atr_ratio: float = 1.5,
    volatile_hawkes_z: float = 1.5,
    trending_atr_ratio: float = 0.8,
    trending_strength_min: float = 0.55,
    low_liquidity_cov: float = 0.30,
) -> pd.DataFrame:
    """
    يعين regime_label بشكل سببي على مستوى الشموع.
    مهم: volatile يتطلب BOTH (ATR مرتفع + Hawkes مرتفع) لتجنب false positives.
    """
    out = df.copy()
    if len(out) == 0:
        out['regime_label'] = pd.Series(dtype='object')
        out['regime_cluster'] = pd.Series(dtype=np.int8)
        return out

    atr = pd.to_numeric(out.get('atr_14', 0.0), errors='coerce').fillna(0.0).astype(np.float64)
    atr_med = atr.rolling(roll_window, min_periods=min_periods).median().replace(0.0, np.nan)
    atr_ratio = (atr / atr_med).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    hawkes = pd.to_numeric(out.get('hawkes_intensity', 0.0), errors='coerce').fillna(0.0).astype(np.float64)
    hk_roll = hawkes.rolling(roll_window, min_periods=min_periods)
    hawk_z = ((hawkes - hk_roll.mean()) / (hk_roll.std() + 1e-9)).fillna(0.0)

    cvd_signed = pd.to_numeric(
        out.get('bar_cvd_delta', out.get('cvd_velocity_signed', 0.0)),
        errors='coerce',
    ).fillna(0.0).astype(np.float64)
    cvd_ref = cvd_signed.abs().rolling(roll_window, min_periods=min_periods).median().fillna(0.0)
    trend_strength = pd.to_numeric(out.get('trend_strength', 0.0), errors='coerce').fillna(0.0).abs()

    volatile_mask = (atr_ratio > float(volatile_atr_ratio)) & (hawk_z > float(volatile_hawkes_z))
    trending_mask = (
        (~volatile_mask)
        & (atr_ratio > float(trending_atr_ratio))
        & ((cvd_signed.abs() > (cvd_ref + 1e-12)) | (trend_strength > float(trending_strength_min)))
    )

    if 'mbp_bar_coverage' in out.columns:
        cov = pd.to_numeric(out['mbp_bar_coverage'], errors='coerce').fillna(0.0).astype(np.float64)
        lowliq_mask = cov < float(low_liquidity_cov)
    else:
        lowliq_mask = pd.Series(False, index=out.index)

    regime_values = np.select(
        [lowliq_mask.to_numpy(), volatile_mask.to_numpy(), trending_mask.to_numpy()],
        ['low_liquidity', 'volatile', 'trending'],
        default='ranging',
    )
    out['regime_label'] = pd.Series(regime_values, index=out.index, dtype='object')
    out['regime_cluster'] = out['regime_label'].map(
        {'trending': 0, 'ranging': 1, 'volatile': 2, 'low_liquidity': 3},
    ).fillna(1).astype(np.int8)

    counts = out['regime_label'].value_counts().to_dict()
    total = max(len(out), 1)
    print("  ✅ Regime labels assigned:")
    for reg in ('trending', 'ranging', 'volatile', 'low_liquidity'):
        n_reg = int(counts.get(reg, 0))
        print(f"     {reg:12s}: {n_reg:,} ({n_reg/total:.1%})")
    return out


def add_event_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    يصنع event_direction من تصويت مصادر اتجاه:
      1) bar_cvd_delta (فرق CVD داخل الشمعة؛ كان مكرراً سابقاً كـ cvd_velocity_signed)
      2) obi_direction
      3) kalman_direction
      4) vwap_dist_1h_roll (إن وُجد): إشارة انحراف السعر عن VWAP سببي ~ساعة / ATR

    +1 LONG-bias | -1 SHORT-bias | 0 ambiguous
    """
    out = df.copy()
    cvd_signed = pd.to_numeric(
        out.get('bar_cvd_delta', out.get('cvd_velocity_signed', 0.0)),
        errors='coerce',
    ).fillna(0.0).astype(np.float64)

    if 'obi_direction' in out.columns:
        obi_dir = pd.to_numeric(out['obi_direction'], errors='coerce').fillna(0).astype(np.int8)
    else:
        obi_raw = pd.to_numeric(out.get('obi', out.get('order_flow_imbalance', 0.0)), errors='coerce').fillna(0.0)
        obi_dir = np.sign(obi_raw.to_numpy(dtype=np.float64)).astype(np.int8)
        obi_dir = pd.Series(obi_dir, index=out.index, dtype=np.int8)

    kalman_dir = pd.to_numeric(out.get('kalman_direction', 0), errors='coerce').fillna(0).astype(np.int8)
    cvd_arr = cvd_signed.to_numpy(dtype=np.float64)
    obi_arr = obi_dir.to_numpy(dtype=np.int8)
    kalman_arr = kalman_dir.to_numpy(dtype=np.int8)
    has_vwap_roll = 'vwap_dist_1h_roll' in out.columns
    if has_vwap_roll:
        vwap_h = pd.to_numeric(out['vwap_dist_1h_roll'], errors='coerce').fillna(0.0).astype(np.float64)
        vwap_vote = np.sign(vwap_h).astype(np.int8)
    else:
        vwap_h = np.zeros(len(out), dtype=np.float64)
        vwap_vote = np.zeros(len(out), dtype=np.int8)

    vote_need = 3 if has_vwap_roll else 2
    vote = (
        np.sign(cvd_arr).astype(np.int8)
        + np.sign(obi_arr).astype(np.int8)
        + np.sign(kalman_arr).astype(np.int8)
        + vwap_vote
    )
    direction = np.where(
        vote >= vote_need, 1, np.where(vote <= -vote_need, -1, 0)
    ).astype(np.int8)

    # إذا التصويت الصارم (أغلبية: 3/4 عند وجود VWAP roll أو 2/3 بدونه) غامض، نستخدم اتجاهاً أضعف لكن سببياً.
    # This prevents valid directional bars from being discarded before outcome labeling.
    if 'obi' in out.columns:
        obi_raw = pd.to_numeric(out['obi'], errors='coerce').fillna(0.0).astype(np.float64)
    elif 'order_flow_imbalance' in out.columns:
        obi_raw = pd.to_numeric(out['order_flow_imbalance'], errors='coerce').fillna(0.0).astype(np.float64)
    else:
        obi_raw = pd.Series(obi_arr.astype(np.float64), index=out.index, dtype=np.float64)

    cvd_scale = pd.Series(np.abs(cvd_arr), index=out.index).rolling(50, min_periods=5).median()
    cvd_norm = pd.Series(cvd_arr, index=out.index) / cvd_scale.replace(0.0, np.nan)
    cvd_norm = cvd_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    obi_norm = obi_raw.clip(-1.0, 1.0)
    if has_vwap_roll:
        vwap_scale = pd.Series(np.abs(vwap_h), index=out.index).rolling(50, min_periods=5).median()
        vwap_norm = pd.Series(vwap_h, index=out.index) / vwap_scale.replace(0.0, np.nan)
        vwap_norm = vwap_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        fallback_score = (
            0.38 * cvd_norm.to_numpy(dtype=np.float64)
            + 0.30 * obi_norm.to_numpy(dtype=np.float64)
            + 0.17 * np.sign(kalman_arr).astype(np.float64)
            + 0.15 * vwap_norm.to_numpy(dtype=np.float64)
        )
    else:
        fallback_score = (
            0.45 * cvd_norm.to_numpy(dtype=np.float64)
            + 0.35 * obi_norm.to_numpy(dtype=np.float64)
            + 0.20 * np.sign(kalman_arr).astype(np.float64)
        )
    fallback_dir = np.where(fallback_score > 0.10, 1, np.where(fallback_score < -0.10, -1, 0)).astype(np.int8)
    weak_vote_dir = np.where(vote > 0, 1, np.where(vote < 0, -1, 0)).astype(np.int8)
    direction = np.where(direction != 0, direction, np.where(fallback_dir != 0, fallback_dir, weak_vote_dir)).astype(np.int8)
    out['event_direction'] = direction
    out['event_direction_score'] = fallback_score.astype(np.float32)

    if 'is_event' in out.columns:
        ev_mask = pd.to_numeric(out['is_event'], errors='coerce').fillna(0).astype(np.int8) == 1
        if bool(ev_mask.any()):
            ev_dir = out.loc[ev_mask, 'event_direction']
            vc = ev_dir.value_counts().to_dict()
            n = int(ev_mask.sum())
            print(
                "  🧭 Event direction votes: "
                f"long={int(vc.get(1, 0)):,} ({int(vc.get(1, 0))/max(n,1):.1%}) | "
                f"short={int(vc.get(-1, 0)):,} ({int(vc.get(-1, 0))/max(n,1):.1%}) | "
                f"amb={int(vc.get(0, 0)):,} ({int(vc.get(0, 0))/max(n,1):.1%})"
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DeepLOB tensors — نافذة 50 بار تاريخية (محاذاة DeepLOBCNN)
# ══════════════════════════════════════════════════════════════════════════════

DEEPLOB_TIME_STEPS_DEFAULT = 50  # متوافق مع modules.deeplob_cnn.N_TIME_STEPS


def attach_mbp_bar_coverage(
    df_bars: pd.DataFrame,
    df_mbp: pd.DataFrame | None,
    freq: str,
    *,
    expected_snap_s: float = 0.5,
) -> pd.DataFrame:
    """نسبة تغطية MBP المتوقعة داخل كل شمعة (0–1)، حسب عدد snapshots / توقع كل expected_snap_s."""
    out = df_bars.copy()
    if df_mbp is None or len(df_mbp) == 0:
        out['mbp_bar_coverage'] = np.float32(0.0)
        return out
    mbp = df_mbp.copy()
    mbp['ts_event'] = pd.to_datetime(mbp['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    mbp = mbp.dropna(subset=['ts_event'])
    mbp['_bk'] = mbp['ts_event'].dt.floor(freq)
    cnt_ser = mbp.groupby('_bk', sort=False).size()
    bk = pd.to_datetime(out['ts_event'], errors='coerce').dt.floor(freq)
    merged = bk.map(cnt_ser).fillna(0).astype(np.float64).to_numpy()
    expect_raw = float(_bar_period_seconds(freq)) / float(max(expected_snap_s, 1e-6))
    expect_slots = float(max(expect_raw, 1.0))
    cov = np.clip(merged / expect_slots, 0.0, 1.0).astype(np.float32)
    out['mbp_bar_coverage'] = cov
    del mbp
    return out


def _normalize_lob_tensor_nonflat(
    tensors: np.ndarray,
    *,
    min_samples: int = 10,
    eps: float = 1e-12,
    clip_std: float = 6.0,
) -> np.ndarray:
    """FIX #4: (µ,σ) لكل زوج (مستوى سعر، قناة) عبر N×T — ثم قصّ لحصر ذيول الشواذ."""
    out = tensors.astype(np.float32, copy=True)
    if out.ndim != 4:
        return out
    n_b, t_b, p_b, c_b = out.shape
    for c in range(c_b):
        for p in range(p_b):
            sl = out[:, :, p, c].reshape(-1)
            nz = sl[np.abs(sl) > eps]
            if len(nz) < min_samples:
                continue
            mu = float(np.mean(nz))
            sd = float(np.std(nz))
            if sd > 1e-8:
                col = (out[:, :, p, c] - mu) / sd
                if clip_std is not None and clip_std > 0:
                    col = np.clip(col, -clip_std, clip_std)
                out[:, :, p, c] = col.astype(np.float32)
    return out


def _coverage_stats(series: pd.Series | np.ndarray, *, low_threshold: float) -> dict:
    vals = pd.to_numeric(pd.Series(series), errors='coerce').fillna(0.0).astype(np.float64).to_numpy()
    if vals.size == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'p25': 0.0,
            'p75': 0.0,
            'low_ratio': 0.0,
            'threshold': float(low_threshold),
        }
    return {
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'p25': float(np.percentile(vals, 25)),
        'p75': float(np.percentile(vals, 75)),
        'low_ratio': float(np.mean(vals < float(low_threshold))),
        'threshold': float(low_threshold),
    }


def _run_event_gate_audit_safe(out_path: str, output_dir: str) -> dict | None:
    """
    Post-write auto-diagnostic. Runs the event-gate audit on the parquet we
    just wrote and prints a single-line verdict. Wrapped in try/except so a
    missing diagnostic module or audit failure never breaks the pipeline.

    Returns the audit summary dict on success, None on any failure.
    """
    try:
        from pathlib import Path as _Path
        from tools.diagnostics.audit_event_gate import run_audit
    except Exception as exc:
        print(f"   ⚠️  event-gate audit unavailable: {type(exc).__name__}: {exc}")
        return None

    try:
        audit_dir = os.path.join(output_dir, '_audit_event_gate')
        summary = run_audit(_Path(out_path), _Path(audit_dir))
    except Exception as exc:
        print(f"   ⚠️  event-gate audit failed: {type(exc).__name__}: {exc}")
        return None

    verdict = str(summary.get('verdict', 'INCOMPLETE'))
    rate = summary.get('overall_event_rate')
    icon = {
        'HEALTHY':             '✅',
        'LOW_RATE':            '⚠️ ',
        'STARVED':             '🚨',
        'WARMUP_HEAVY':        '⚠️ ',
        'COMPONENT_DOMINATED': '🚨',
        'REGIME_STARVED':      '⚠️ ',
        'INSUFFICIENT_DATA':   '❔',
        'INCOMPLETE':          '❔',
    }.get(verdict, '❔')
    rate_str = f"{rate:.1%}" if isinstance(rate, (int, float)) else "n/a"
    print(f"\n   {icon} Event gate: {verdict} (rate={rate_str}) → {audit_dir}/")

    # Loud diagnostic when the gate is the root cause of high NEUTRAL
    if verdict in ('STARVED', 'COMPONENT_DOMINATED'):
        print("      ← الـ NEUTRAL العالي مرجعه gate وليس market behavior")
        diag = summary.get('diag', {})
        for k, v in diag.items():
            if k != 'n_rows':
                print(f"      ← {k}: {v}")
    elif verdict in ('LOW_RATE', 'WARMUP_HEAVY', 'REGIME_STARVED'):
        diag = summary.get('diag', {})
        flagged = {k: v for k, v in diag.items() if k not in ('n_rows', 'overall_rate')}
        if flagged:
            print(f"      ← diag: {flagged}")

    return summary


def _mbo_bar_coverage_from_tick_series(tc: pd.Series) -> np.ndarray:
    """كثافة شريط MBO (tick_count) مقابل وسط محلي ∈ [0,1] — موازٍ لفكرة تغطية MBP."""
    t = pd.to_numeric(tc, errors='coerce').fillna(0.0).astype(np.float64)
    med = t.rolling(EVENT_ZSCORE_WINDOW, min_periods=EVENT_ZSCORE_MIN_PERIODS).median()
    den = np.maximum(med.to_numpy(dtype=np.float64), 1.0)
    den = np.where(np.isfinite(den), den, np.maximum(t.to_numpy(dtype=np.float64), 1.0))
    return np.minimum(t.to_numpy(dtype=np.float64) / den, 1.0).astype(np.float32)


def _trade_footprint_bar(
    mbo_slice_lo: int,
    mbo_slice_hi: int,
    *,
    mbo_ts: np.ndarray,
    mbo_action: np.ndarray,
    mbo_side: np.ndarray,
    mbo_price: np.ndarray,
    mbo_size: np.ndarray,
    bid0: float,
    ask0: float,
    levels: int,
    tick_med: float,
    fallback_tick: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    buy_fp = np.zeros(levels, dtype=np.float32)
    sell_fp = np.zeros(levels, dtype=np.float32)
    tick = float(ask0 - bid0) if (ask0 > 0 and bid0 > 0 and ask0 > bid0) else float(fallback_tick)
    if tick <= 0 or mbo_slice_hi <= mbo_slice_lo:
        return buy_fp, sell_fp
    ask_anchor = ask0 if ask0 > 0 else (bid0 + tick if bid0 > 0 else 0.0)
    bid_anchor = bid0 if bid0 > 0 else (ask0 - tick if ask0 > 0 else 0.0)
    for k in range(mbo_slice_lo, mbo_slice_hi):
        act_k = str(mbo_action[k]).strip().upper()
        if act_k not in TRADE_ACTIONS and act_k not in ('', 'NAN', 'NONE'):
            continue
        sz = float(mbo_size[k])
        px = float(mbo_price[k])
        if sz <= 0 or px <= 0:
            continue
        side_k = str(mbo_side[k]).strip().upper()
        if side_k in ('A', 'ASK', 'BUY', 'BOT', '1', '+1', 'LONG', 'L'):
            dist = abs((px - ask_anchor) / tick)
            lvl = max(0, min(levels - 1, int(round(dist))))
            buy_fp[lvl] += np.float32(sz / tick_med)
        elif side_k in ('B', 'BID', 'S', 'SELL', '-1', 'SHORT', 'SH'):
            dist = abs((bid_anchor - px) / tick)
            lvl = max(0, min(levels - 1, int(round(dist))))
            sell_fp[lvl] += np.float32(sz / tick_med)
        else:
            mid = 0.5 * (ask_anchor + bid_anchor) if (ask_anchor > 0 and bid_anchor > 0) else px
            dist = abs((px - mid) / tick)
            lvl = max(0, min(levels - 1, int(round(dist))))
            if px >= mid:
                buy_fp[lvl] += np.float32(sz / tick_med)
            else:
                sell_fp[lvl] += np.float32(sz / tick_med)
    return buy_fp, sell_fp


# ════════════════════════════════════════════════════════════════════════════
# 9-channel LOB representation
# ════════════════════════════════════════════════════════════════════════════
#
# Channel layout per (T-bar, P-level=20):
#   ch0  depth_log         log1p(size at level)                  [keep — original ch0]
#   ch1  trade_imbalance   (buy_vol - sell_vol)/(buy + sell)     [keep — original ch1]
#   ch2  trade_vol_log     log1p(total trade vol at level)       [keep — original ch2]
#   ch3  bid_depth_log     log1p(bid_sz at level, zero at ask)   [NEW separated bid]
#   ch4  ask_depth_log     log1p(ask_sz at level, zero at bid)   [NEW separated ask]
#   ch5  buy_vol_log       log1p(buy trade vol at level)         [NEW separated buy]
#   ch6  sell_vol_log      log1p(sell trade vol at level)        [NEW separated sell]
#   ch7  wall_flag         1.0 if size > 3× per-snapshot median  [NEW wall detection]
#   ch8  depth_velocity    (depth_now - depth_prev)/max(prev,1)  [NEW temporal δ]
#
# Levels 0..9  = bid side (reversed: 0 = deepest bid, 9 = best bid)
# Levels 10..19 = ask side (10 = best ask, 19 = deepest ask)
N_LOB_CHANNELS = 9


def _build_9ch_snapshot(
    depth_combined: np.ndarray,     # (P=20,) log1p of bid+ask sizes
    raw_depth_combined: np.ndarray, # (P=20,) raw bid+ask sizes (no log)
    bid_sz_row: np.ndarray,         # (n_levels=10,) raw bid sizes, best..deepest
    ask_sz_row: np.ndarray,         # (n_levels=10,) raw ask sizes, best..deepest
    full_buy: np.ndarray,           # (P=20,) raw buy trade vol per level
    full_sell: np.ndarray,          # (P=20,) raw sell trade vol per level
    prev_raw_depth: np.ndarray | None,  # (P=20,) raw depth at previous snapshot
    levels: int,
    wall_factor: float = 3.0,
) -> np.ndarray:
    """Compose 9-channel feature for ONE snapshot. Returns (P, 9)."""
    P = depth_combined.shape[0]
    out = np.zeros((P, N_LOB_CHANNELS), dtype=np.float32)

    # ch0 — combined depth (original)
    out[:, 0] = depth_combined.astype(np.float32)

    # ch1 — trade imbalance per level
    tot_fp_raw = full_buy + full_sell
    imb = np.divide(full_buy - full_sell, np.maximum(tot_fp_raw, 1e-9))
    out[:, 1] = imb.astype(np.float32)

    # ch2 — total trade vol (original)
    out[:, 2] = np.log1p(np.maximum(tot_fp_raw, 0.0)).astype(np.float32)

    # ch3 — bid_depth_log (zero at ask levels)
    bid_depth = np.zeros(P, dtype=np.float32)
    # bid levels are P=0..9, reversed (deepest..best); bid_sz_row is best..deepest
    bid_depth[:levels] = np.log1p(np.maximum(bid_sz_row[::-1].astype(np.float32), 0.0))
    out[:, 3] = bid_depth

    # ch4 — ask_depth_log (zero at bid levels)
    ask_depth = np.zeros(P, dtype=np.float32)
    ask_depth[levels:] = np.log1p(np.maximum(ask_sz_row.astype(np.float32), 0.0))
    out[:, 4] = ask_depth

    # ch5 — buy_vol_log (mostly at ask side)
    out[:, 5] = np.log1p(np.maximum(full_buy.astype(np.float32), 0.0))

    # ch6 — sell_vol_log (mostly at bid side)
    out[:, 6] = np.log1p(np.maximum(full_sell.astype(np.float32), 0.0))

    # ch7 — wall_flag: large size relative to snapshot median (excludes zeros)
    nz = raw_depth_combined[raw_depth_combined > 0]
    if len(nz) > 0:
        med = float(np.median(nz))
        threshold = wall_factor * med
        out[:, 7] = (raw_depth_combined > threshold).astype(np.float32)

    # ch8 — depth_velocity vs previous snapshot
    if prev_raw_depth is not None:
        delta = raw_depth_combined - prev_raw_depth
        denom = np.maximum(prev_raw_depth, 1.0)
        out[:, 8] = (delta / denom).astype(np.float32)
        # Clip extreme jumps (orderbook reset events)
        out[:, 8] = np.clip(out[:, 8], -10.0, 10.0)

    return out


def build_rolling_lob_tensors_from_mbp(
    df_mbo: pd.DataFrame,
    df_mbp: pd.DataFrame,
    df_bars: pd.DataFrame,
    *,
    freq: str,
    lookback_bars: int = DEEPLOB_TIME_STEPS_DEFAULT,
    levels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    لكل شمعة i: tensor (lookback, 20, 9) حيث البعد الزمني = شموع سابقة حقيقية.
    9 channels: depth, trade_imb, trade_vol, bid_depth, ask_depth,
                buy_vol, sell_vol, wall_flag, depth_velocity.
    لكل شمعة مصدر: لقطة MBP ذات أقصى |imbalance| داخل الشمعة + بصمة تداول كاملة بنفس الهندسة.
    """
    bars = df_bars.sort_values('ts_event').reset_index(drop=True)
    df_mbp = df_mbp.copy()
    df_mbp['ts_event'] = pd.to_datetime(df_mbp['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    df_mbp = df_mbp.dropna(subset=['ts_event']).sort_values('ts_event')
    df_mbo = df_mbo.sort_values('ts_event').reset_index(drop=True)

    bid_px_cols = [f'bid_px_{i:02d}' for i in range(levels)]
    ask_px_cols = [f'ask_px_{i:02d}' for i in range(levels)]
    bid_sz_cols = [f'bid_sz_{i:02d}' for i in range(levels)]
    ask_sz_cols = [f'ask_sz_{i:02d}' for i in range(levels)]

    def _col_or_zeros(frame: pd.DataFrame, col: str) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
        return pd.Series(np.zeros(len(frame), dtype=np.float64), index=frame.index, dtype=np.float64)

    mbp_ts = df_mbp['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    bid_px = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in bid_px_cols]).T
    ask_px = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in ask_px_cols]).T
    bid_sz = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in bid_sz_cols]).T
    ask_sz = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in ask_sz_cols]).T
    spread0 = (ask_px[:, 0] - bid_px[:, 0]) if (ask_px.shape[1] and bid_px.shape[1]) else np.array([], dtype=np.float64)
    spread0 = spread0[np.isfinite(spread0) & (spread0 > 0.0)]
    fallback_tick = float(np.median(spread0)) if spread0.size else 0.0

    mbo_ts = df_mbo['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    if 'action' in df_mbo.columns:
        mbo_action = df_mbo['action'].astype(str).str.upper().to_numpy()
    else:
        # Stage1 final artifacts may not retain raw action column.
        # Treat positive-size rows as trade-like rows for footprint reconstruction.
        mbo_action = np.full(len(df_mbo), 'T', dtype=object)

    if 'side' in df_mbo.columns:
        mbo_side = df_mbo['side'].astype(str).str.upper().to_numpy()
    else:
        # Infer aggressor side when raw side is unavailable.
        side_guess = np.full(len(df_mbo), '', dtype=object)
        if 'vnet' in df_mbo.columns:
            vnet_arr = pd.to_numeric(df_mbo['vnet'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
            side_guess[vnet_arr > 0.0] = 'A'
            side_guess[vnet_arr < 0.0] = 'B'
        unresolved = side_guess == ''
        if bool(np.any(unresolved)):
            px = pd.to_numeric(df_mbo.get('price', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
            dpx = np.diff(px, prepend=px[0] if len(px) else 0.0)
            side_guess[unresolved & (dpx >= 0.0)] = 'A'
            side_guess[unresolved & (dpx < 0.0)] = 'B'
        mbo_side = side_guess
    mbo_price = pd.to_numeric(df_mbo.get('price', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    mbo_size_col = _resolve_mbo_size_column(df_mbo) if len(df_mbo.columns) else 'size'
    mbo_size = pd.to_numeric(df_mbo.get(mbo_size_col, 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)

    tick_med = float(
        pd.to_numeric(df_mbo.get(mbo_size_col, 0), errors='coerce').replace(0.0, np.nan).median() or 1.0,
    )
    tick_med = max(tick_med, 1e-6)

    n_bars = len(bars)
    n_lv2 = levels * 2
    T = int(max(lookback_bars, 1))
    bar_ns = np.timedelta64(int(_bar_period_seconds(freq) * 1e9), 'ns')

    # bar_snapshots[bi] = (snap_9ch, raw_depth) or None
    # snap_9ch is (P, 9) float32; raw_depth needed for velocity in next bar
    bar_snapshots: list[tuple[np.ndarray, np.ndarray] | None] = []
    ts_bar = bars['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)

    bars_with_snapshot = 0
    prev_raw_depth: np.ndarray | None = None
    for bi in range(n_bars):
        t0 = ts_bar[bi]
        t1 = t0 + bar_ns
        lo_m = int(np.searchsorted(mbp_ts, t0, side='left'))
        hi_m = int(np.searchsorted(mbp_ts, t1, side='left'))
        if hi_m <= lo_m:
            bar_snapshots.append(None)
            continue
        idx_m = np.arange(lo_m, hi_m, dtype=np.int32)
        b_depth_row = bid_sz[idx_m].sum(axis=1)
        a_depth_row = ask_sz[idx_m].sum(axis=1)
        tot_d = b_depth_row + a_depth_row + 1e-9
        imb_mag = np.abs((b_depth_row - a_depth_row) / tot_d)
        # Blend top-3 imbalance-weighted snapshots
        order = np.argsort(-imb_mag)
        top_k = int(min(3, len(order)))
        top_local = order[:top_k].astype(np.int64, copy=False)
        w_raw = imb_mag[top_local].astype(np.float64, copy=False)
        w_sum = float(np.sum(w_raw))
        blend_w = (w_raw / w_sum) if w_sum > 1e-12 else (np.ones(top_k, dtype=np.float64) / max(top_k, 1))

        raw_depth_acc = np.zeros(n_lv2, dtype=np.float64)
        bid_sz_acc = np.zeros(levels, dtype=np.float64)
        ask_sz_acc = np.zeros(levels, dtype=np.float64)
        bid0_acc = 0.0
        ask0_acc = 0.0
        for wi, jloc in zip(blend_w, top_local):
            row_idx = int(idx_m[jloc])
            bid_row = np.maximum(bid_sz[row_idx].astype(np.float64), 0.0)
            ask_row = np.maximum(ask_sz[row_idx].astype(np.float64), 0.0)
            raw = np.concatenate([bid_row[::-1], ask_row], axis=0)
            raw_depth_acc += float(wi) * raw
            bid_sz_acc += float(wi) * bid_row
            ask_sz_acc += float(wi) * ask_row
            if bid_px.shape[1]:
                bid0_acc += float(wi) * float(bid_px[row_idx, 0])
            if ask_px.shape[1]:
                ask0_acc += float(wi) * float(ask_px[row_idx, 0])
        depth_combined = np.log1p(raw_depth_acc).astype(np.float32)
        bid0 = float(bid0_acc)
        ask0 = float(ask0_acc)

        # Trade footprint within this bar window
        lo_t = int(np.searchsorted(mbo_ts, t0, side='left'))
        hi_t = int(np.searchsorted(mbo_ts, t1, side='left'))
        buy_fp, sell_fp = _trade_footprint_bar(
            lo_t, hi_t,
            mbo_ts=mbo_ts, mbo_action=mbo_action, mbo_side=mbo_side,
            mbo_price=mbo_price, mbo_size=mbo_size,
            bid0=bid0, ask0=ask0,
            levels=levels, tick_med=tick_med, fallback_tick=fallback_tick,
        )
        full_buy = np.zeros(n_lv2, dtype=np.float32)
        full_sell = np.zeros(n_lv2, dtype=np.float32)
        full_buy[levels:] = buy_fp.astype(np.float32, copy=False)
        full_sell[:levels] = sell_fp[::-1].astype(np.float32, copy=False)

        # Compose 9-channel snapshot
        snap_9ch = _build_9ch_snapshot(
            depth_combined=depth_combined,
            raw_depth_combined=raw_depth_acc.astype(np.float32),
            bid_sz_row=bid_sz_acc.astype(np.float32),
            ask_sz_row=ask_sz_acc.astype(np.float32),
            full_buy=full_buy,
            full_sell=full_sell,
            prev_raw_depth=prev_raw_depth,
            levels=levels,
        )
        bar_snapshots.append((snap_9ch, raw_depth_acc.astype(np.float32)))
        prev_raw_depth = raw_depth_acc.astype(np.float32)
        bars_with_snapshot += 1

    tensors = np.zeros((n_bars, T, n_lv2, N_LOB_CHANNELS), dtype=np.float32)
    timestamps = np.zeros(n_bars, dtype='datetime64[ns]')
    roll_cov = np.zeros(n_bars, dtype=np.float32)

    for bi in range(n_bars):
        timestamps[bi] = ts_bar[bi]
        filled = 0
        for lag in range(T):
            src = bi - (T - 1 - lag)
            if src < 0:
                continue
            snap = bar_snapshots[src]
            if snap is None:
                continue
            snap_9ch, _ = snap
            tensors[bi, lag, :, :] = snap_9ch
            filled += 1
        roll_cov[bi] = float(filled) / float(T)

    tensors = _normalize_lob_tensor_nonflat(tensors)
    snap_ratio = float(bars_with_snapshot) / float(max(n_bars, 1))
    mean_roll_cov = float(np.mean(roll_cov)) if len(roll_cov) else 0.0
    low_roll_ratio = float(np.mean(roll_cov < 0.50)) if len(roll_cov) else 1.0
    print(
        "  📊 LOB coverage telemetry: "
        f"bar_snapshots={bars_with_snapshot:,}/{n_bars:,} ({snap_ratio:.1%}) | "
        f"roll_mean={mean_roll_cov:.1%} | roll_low(<50%)={low_roll_ratio:.1%}"
    )
    if mean_roll_cov < 0.5 or low_roll_ratio > 0.4:
        print(
            "  ⚠️ rolling LOB coverage weak — CNN quality may degrade unless MBP density improves."
        )
    return tensors, timestamps, roll_cov


def build_rolling_lob_tensors_mbo_only(
    df_mbo: pd.DataFrame,
    df_bars: pd.DataFrame,
    *,
    freq: str,
    lookback_bars: int = DEEPLOB_TIME_STEPS_DEFAULT,
    n_levels: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """بدون MBP: كل «خطوة زمنية» = شمعة سابقة؛ القنوات من تدفق التيكات فقط (نفس الشكل DeepLOB)."""
    df_mbo = df_mbo.copy()
    df_mbo['ts_event'] = pd.to_datetime(df_mbo['ts_event'])
    df_mbo['bar_key'] = df_mbo['ts_event'].dt.floor(freq)
    size_num = pd.to_numeric(df_mbo.get('size', 0), errors='coerce').fillna(0.0)
    df_mbo['_size_num'] = size_num
    tick_med = max(float(size_num.median() or 1.0), 1e-6)

    bars = df_bars.sort_values('ts_event').reset_index(drop=True)
    n_bars = len(bars)
    T = int(max(lookback_bars, 1))

    # ── O(bars × ticks) → O(ticks): pre-aggregate once via groupby، ثم alignment vectorized
    side_a = df_mbo['side'].astype(str).str.upper().eq('A')
    side_b = df_mbo['side'].astype(str).str.upper().eq('B')
    action_t = df_mbo['action'].astype(str).str.upper().eq('T')
    tick_count_g = df_mbo.groupby('bar_key').size()
    buy_vol_g  = df_mbo.loc[side_a, ['bar_key', '_size_num']].groupby('bar_key')['_size_num'].sum()
    sell_vol_g = df_mbo.loc[side_b, ['bar_key', '_size_num']].groupby('bar_key')['_size_num'].sum()
    buy_t_g    = df_mbo.loc[side_a & action_t, ['bar_key', '_size_num']].groupby('bar_key')['_size_num'].sum()
    sell_t_g   = df_mbo.loc[side_b & action_t, ['bar_key', '_size_num']].groupby('bar_key')['_size_num'].sum()

    ts_bar = bars['ts_event'].to_numpy()
    ts_bar_floored = pd.to_datetime(ts_bar).floor(freq)
    has_ticks = tick_count_g.reindex(ts_bar_floored).fillna(0).to_numpy() > 0
    bv = buy_vol_g.reindex(ts_bar_floored).fillna(0.0).to_numpy(dtype=np.float64)
    sv = sell_vol_g.reindex(ts_bar_floored).fillna(0.0).to_numpy(dtype=np.float64)
    bt = buy_t_g.reindex(ts_bar_floored).fillna(0.0).to_numpy(dtype=np.float64)
    st = sell_t_g.reindex(ts_bar_floored).fillna(0.0).to_numpy(dtype=np.float64)
    tot_v = bv + sv
    imb_arr = (bv - sv) / np.maximum(tot_v, 1.0)
    c1_arr = bt / tick_med
    c2_arr = st / tick_med

    # MBO-only fallback: we don't have book depth, so most channels are
    # degenerate. We emit zeros for book-specific channels (ch0, ch3, ch4,
    # ch7, ch8) and fill the trade/flow channels (ch1, ch2, ch5, ch6) with
    # bar-level aggregates broadcast across all P levels.
    feats: list[tuple[float, float, float, float, float] | None] = [
        (
            float(imb_arr[i]),                        # trade imbalance
            float(np.log1p(tot_v[i])),                # total trade vol log
            float(np.log1p(bv[i])),                   # buy vol log
            float(np.log1p(sv[i])),                   # sell vol log
            float(c1_arr[i] - c2_arr[i]),             # placeholder velocity-ish
        ) if bool(has_ticks[i]) else None
        for i in range(n_bars)
    ]

    tensors = np.zeros((n_bars, T, n_levels, N_LOB_CHANNELS), dtype=np.float32)
    timestamps = bars['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    roll_cov = np.zeros(n_bars, dtype=np.float32)

    for bi in range(n_bars):
        filled = 0
        for lag in range(T):
            src = bi - (T - 1 - lag)
            if src < 0:
                continue
            f = feats[src]
            if f is None:
                continue
            imb, tvol, bvol, svol, vel = f
            # ch0: depth — degenerate (no MBP), zero
            # ch1: trade imbalance
            tensors[bi, lag, :, 1] = np.float32(imb)
            # ch2: total trade vol log
            tensors[bi, lag, :, 2] = np.float32(tvol)
            # ch3, ch4: bid/ask depth — degenerate without MBP
            # ch5: buy trade vol log
            tensors[bi, lag, :, 5] = np.float32(bvol)
            # ch6: sell trade vol log
            tensors[bi, lag, :, 6] = np.float32(svol)
            # ch7: wall flag — degenerate
            # ch8: depth velocity — approximate with vol-imbalance velocity
            tensors[bi, lag, :, 8] = np.float32(vel)
            filled += 1
        roll_cov[bi] = float(filled) / float(T)

    tensors = _normalize_lob_tensor_nonflat(tensors)
    return tensors, timestamps.copy(), roll_cov


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3a: Event Gate — يكشف لحظات الـ Informed Flow الحقيقي (المشكلة F + I)
# ══════════════════════════════════════════════════════════════════════════════

def detect_microstructure_events(
    df: pd.DataFrame,
    *,
    threshold_scale: float = 1.0,
    threshold_shift: float = 0.0,
    min_event_rate: float = 0.12,
) -> pd.DataFrame:
    """
    يكشف اللحظات التي يكون فيها informed flow حقيقي ويُضيف:
        - is_event    (int8)   : 1 = حدث حقيقي، 0 = ضجيج
        - event_score (float32): درجة قوة الحدث [0.0, 1.0]

    المنطق:
        hawkes_z  > 1.0 → أوردرات تتسارع فوق الطبيعي
        absorb_z  > 1.0 → صانع سوق نشط بشكل غير عادي
        kyle_z    > 0.5 → السعر يستجيب للـ flow
        cvd_align > 0.6 → ضغط في اتجاه واحد
        mbp_roll_lob_coverage > EVENT_LOB_COVERAGE_CUT (وزن اختياري) → تغطية عمق LOB في نافذة MBP roll
        mbo_bar_coverage > EVENT_MBO_COVERAGE_CUT (وزن اختياري) → كثافة تيكات MBO مقابل وسط محلي

    Regime Gate (المشكلة I):
        Volatile regime → عتبة أعلى (0.75) لأن الإشارات الضعيفة مضللة
        Trending/Ranging → عتبة قياسية (0.60)

    النتيجة المتوقعة:
        من 100% bars → ~20-30% events حقيقية
        NEUTRAL ينخفض من 70-80% إلى ~35-45%
    """
    df = df.copy()

    def _zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(EVENT_ZSCORE_WINDOW, min_periods=EVENT_ZSCORE_MIN_PERIODS)
        return ((series - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)

    # ── حساب Z-scores ────────────────────────────────────────────────────────
    hawkes_z = _zscore(pd.to_numeric(df.get('hawkes_intensity', pd.Series(0.0, index=df.index)),
                                      errors='coerce').fillna(0.0))
    absorb_z = _zscore(pd.to_numeric(df.get('absorption_intensity', pd.Series(0.0, index=df.index)),
                                      errors='coerce').fillna(0.0))
    kyle_z   = _zscore(pd.to_numeric(df.get('kyle_lambda', pd.Series(0.0, index=df.index)),
                                      errors='coerce').fillna(0.0))

    # CVD alignment: نسبة الـ slices في اتجاه واحد (أو CVD momentum كبديل)
    if 'cvd_direction_pct' in df.columns:
        cvd_align = pd.to_numeric(df['cvd_direction_pct'], errors='coerce').fillna(0.5)
    elif 'cvd_momentum' in df.columns:
        # normalize CVD momentum إلى [0,1] كبديل
        cm = pd.to_numeric(df['cvd_momentum'], errors='coerce').fillna(0.0)
        cvd_align = (cm.abs() / (cm.abs().rolling(50, min_periods=5).max() + 1e-9)).clip(0.0, 1.0)
    else:
        cvd_align = pd.Series(0.5, index=df.index)

    cov_raw = pd.to_numeric(df.get('mbp_roll_lob_coverage', pd.Series(0.0, index=df.index)),
                             errors='coerce').fillna(0.0)
    cov = cov_raw.clip(lower=0.0, upper=1.0)

    mbo_cov_raw = pd.to_numeric(df.get('mbo_bar_coverage', pd.Series(0.0, index=df.index)),
                                errors='coerce').fillna(0.0)
    mbo_cov = mbo_cov_raw.clip(lower=0.0, upper=1.0)

    # ── حساب event_score المرجح ──────────────────────────────────────────────
    w = EVENT_SCORE_WEIGHTS
    cov_w = float(w.get('mbp_roll_cov_above_cut', 0.0))
    cov_cut = float(EVENT_LOB_COVERAGE_CUT)
    mbo_w = float(w.get('mbo_tick_cov_above_cut', 0.0))
    mbo_cut = float(EVENT_MBO_COVERAGE_CUT)

    event_score = (
        (hawkes_z  > 1.0).astype(np.float32) * w['hawkes_z_above_1']  +
        (absorb_z  > 1.0).astype(np.float32) * w['absorb_z_above_1']  +
        (kyle_z    > 0.5).astype(np.float32) * w['kyle_z_above_05']   +
        (cvd_align > 0.6).astype(np.float32) * w['cvd_align_above_06']
    ).astype(np.float32)
    if cov_w > 0.0:
        event_score = event_score + (cov > cov_cut).astype(np.float32) * np.float32(cov_w)
    if mbo_w > 0.0:
        event_score = event_score + (mbo_cov > mbo_cut).astype(np.float32) * np.float32(mbo_w)

    continuous_score = (
        hawkes_z.clip(lower=0.0, upper=3.0).astype(np.float32) / np.float32(3.0) * w['hawkes_z_above_1'] +
        absorb_z.clip(lower=0.0, upper=3.0).astype(np.float32) / np.float32(3.0) * w['absorb_z_above_1'] +
        kyle_z.clip(lower=0.0, upper=2.0).astype(np.float32) / np.float32(2.0) * w['kyle_z_above_05'] +
        cvd_align.clip(lower=0.0, upper=1.0).astype(np.float32) * w['cvd_align_above_06']
    ).astype(np.float32)
    if cov_w > 0.0:
        continuous_score = continuous_score + cov.astype(np.float32) * np.float32(cov_w)
    if mbo_w > 0.0:
        continuous_score = continuous_score + mbo_cov.astype(np.float32) * np.float32(mbo_w)

    # ── Regime Gate (المشكلة I) ───────────────────────────────────────────────
    if 'regime_label' in df.columns:
        regime_s = df['regime_label'].astype(str)
        # عتبة ديناميكية: volatile يحتاج score أعلى
        base_threshold = regime_s.map(REGIME_EVENT_THRESHOLD).fillna(0.60).astype(np.float32)
    else:
        base_threshold = pd.Series(0.60, index=df.index, dtype=np.float32)

    scale = float(max(threshold_scale, 0.01))
    shift = float(threshold_shift)
    threshold = (base_threshold.astype(np.float64) * scale + shift).clip(0.05, 0.95).astype(np.float32)

    # نسخة قابلة للكتابة: pandas/pyarrow قد يعيد boolean ndarray read-only
    pass_raw = np.array(event_score >= threshold, dtype=bool)
    event_mask = pass_raw.copy()
    adaptive_added = 0
    target_rate = float(np.clip(min_event_rate, 0.0, 0.60))
    if target_rate > 0.0 and len(df):
        target_events = int(np.ceil(len(df) * target_rate))
        current_events = int(event_mask.sum())
        if current_events < target_events:
            rank_score = np.maximum(
                event_score.to_numpy(dtype=np.float32),
                continuous_score.to_numpy(dtype=np.float32),
            )
            if 'regime_label' in df.columns:
                # pandas/pyarrow: boolean ndarray from .to_numpy() may be read-only
                candidate_mask = np.array(
                    df['regime_label'].astype(str).ne('low_liquidity'),
                    dtype=bool,
                )
            else:
                candidate_mask = np.ones(len(df), dtype=bool)
            candidate_mask &= np.isfinite(rank_score) & (rank_score > 0.0)
            candidate_mask &= ~event_mask
            need = max(target_events - current_events, 0)
            if need > 0 and bool(np.any(candidate_mask)):
                candidate_idx = np.flatnonzero(candidate_mask)
                order = candidate_idx[np.argsort(rank_score[candidate_idx])[::-1]]
                chosen = order[:need]
                event_mask[chosen] = True
                adaptive_added = int(len(chosen))

    df['event_score'] = np.maximum(
        event_score.to_numpy(dtype=np.float32),
        continuous_score.to_numpy(dtype=np.float32),
    ).astype(np.float32)
    df['is_event'] = event_mask.astype(np.int8)

    # ── إحصاءات التشخيص ──────────────────────────────────────────────────────
    n_total = len(df)
    n_event = int(df['is_event'].sum())
    print(
        f"  📊 Event Detection: {n_event:,}/{n_total:,} bars = {n_event/max(n_total,1):.1%} events "
        f"(threshold_scale={scale:.2f}, shift={shift:+.2f}, min_rate={target_rate:.1%}, adaptive_added={adaptive_added})"
    )
    if 'regime_label' in df.columns:
        for reg in ['trending', 'ranging', 'volatile']:
            mask = df['regime_label'] == reg
            n_reg = int(mask.sum())
            n_ev  = int((mask & (df['is_event'] == 1)).sum())
            if n_reg > 0:
                print(f"     {reg:8s}: {n_ev:,}/{n_reg:,} = {n_ev/n_reg:.1%}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b: Label by Outcome — First Barrier Hit (المشكلة G + H + J + K)
# ══════════════════════════════════════════════════════════════════════════════

# حدود الجلسات (UTC hour) — نهاية كل جلسة تداول
_SESSION_END_HOUR: dict[str, int] = {
    'asia'   : 8,
    'london' : 16,
    'overlap': 16,
    'ny'     : 21,
}
_FALLBACK_MAX_BARS: int = 20  # حد أقصى مطلق عند غياب معلومات الجلسة
_KALMAN_EVENT_FLOOR: float = 0.70

# neutral_reason parity with labels_v19
NEUTRAL_REASON_NONE       = 0
NEUTRAL_REASON_TIMEOUT    = 1
NEUTRAL_REASON_LONG_SL    = 2
NEUTRAL_REASON_SHORT_SL   = 3
NEUTRAL_REASON_KALMAN     = 4
NEUTRAL_REASON_WEAK_EVENT = 5


def add_kalman_trend(
    df: pd.DataFrame,
    *,
    obs_noise: float = 1e-3,
    trans_noise: float = 1e-5,
) -> pd.DataFrame:
    """
    يضيف kalman_trend / kalman_direction لفلترة الاتجاهات الضعيفة عكس الترند.
    إذا pykalman غير متاح، نستخدم EWMA fallback للحفاظ على نفس العقد.
    """
    out = df.copy()
    close_src = out['close'] if 'close' in out.columns else pd.Series(np.zeros(len(out), dtype=np.float64), index=out.index)
    close = pd.to_numeric(close_src, errors='coerce').ffill().fillna(0.0).astype(np.float64)
    if len(close) == 0:
        out['kalman_trend'] = np.float32(0.0)
        out['kalman_direction'] = np.int8(0)
        return out
    if _KALMAN_OK and KalmanFilter is not None:
        kf = KalmanFilter(
            transition_matrices=[[1.0]],
            observation_matrices=[[1.0]],
            initial_state_mean=[float(close.iloc[0])],
            observation_covariance=[[float(max(obs_noise, 1e-9))]],
            transition_covariance=[[float(max(trans_noise, 1e-12))]],
        )
        state_means, _ = kf.filter(close.to_numpy(dtype=np.float64).reshape(-1, 1))
        trend = pd.Series(state_means[:, 0], index=out.index, dtype=np.float64)
    else:
        trend = close.ewm(span=12, adjust=False).mean()
        print("  ⚠️ pykalman غير متاح — using EWMA trend fallback for kalman_direction.")
    direction = np.sign(trend.diff().fillna(0.0)).astype(np.int8)
    out['kalman_trend'] = trend.astype(np.float32)
    out['kalman_direction'] = direction
    return out


def label_by_outcome(
    df: pd.DataFrame,
    *,
    default_tp_mult: float = 1.5,
    default_sl_mult: float = 1.0,
    default_max_bars: int = 6,
    min_atr: float = 0.0003,
    kalman_event_floor: float = _KALMAN_EVENT_FLOOR,
    weak_event_to_directional: bool = False,
    weak_event_min_move_atr: float = 0.35,
    sl_to_opposite: bool = False,
    include_weak_directional_in_train: bool = False,
    use_event_score_tier_labels: bool | None = None,
    timeout_mfe_mae: bool = True,
    timeout_mfe_mae_ratio: float = 2.0,
    timeout_mfe_min_move_atr: float = 1.0,
    apply_event_direction_veto: bool = False,
) -> pd.DataFrame:
    """
    يلصق الليبل بناءً على أول حاجز يُضرب (First Barrier Hit).

    apply_event_direction_veto (A2): الفيتو مُعطَّل افتراضياً (False). عند True
    يُسمح لـ event_direction (CVD+OBI+Kalman voting) بمنع اتجاه (allow_long/
    allow_short) في كل من الـ barrier scan والـ MFE/MAE rescue.
    لماذا OFF افتراضياً: الـ diagnostic على real data أثبت أن هذا الـ voting
    يخطئ ~70% من الحالات؛ تفعيله يُسقط ~60% من الـ directional signal الحقيقي
    ويعيد فرض انحياز اتجاهي فوق الـ scan المتماثل المُصلَّح في A1 (event_dir>0
    يُلغي short_hit نهائياً → نرجع لـ LONG-bias لكن بالفيتو هذه المرة). الافتراض
    الآمن هو إبقاؤه مطفأً وترك event_direction عموداً (feature) يتعلّمه النموذج
    بدل أن يكون veto صلباً. مُتاح للتفعيل التجريبي عبر --apply-event-direction-veto.

    Scan بعد A1 (Finding #7): symmetric single-pair barrier.
        - upper = entry + tp_mult × ATR، lower = entry − tp_mult × ATR
        - أول لمس يحسم؛ عند لمس الاتجاهين في نفس الـ bar نختار الـ barrier
          الأقرب لـ open كقياس heuristic لأول-لمس داخل الشمعة.
        - حالات long_sl/short_sl لم تعد ممكنة (الـ scan الجديد يصدّر {long_tp,
          short_tp, timeout} فقط).
    خصائص أخرى:
        1. Timeout = نهاية الجلسة الحالية (لا يتجاوز حدودها)
        2. TP/SL multipliers مخصصة لكل Regime من REGIME_TP_SL (مع شرائح
           event_score الاختيارية للحدث)
        3. Max horizon مخصص لكل Regime من REGIME_MAX_BARS (أو شرائح
           event_score عند التفعيل)

    الأعمدة المُضافة:
        bias_label     (int8):   0=LONG, 1=SHORT, 2=NEUTRAL
        path_outcome   (int8):   0=long_tp, 1=short_tp, 4=timeout.
            ملاحظة: codes 2,3,5,6 (long_sl/short_sl/weak_long/weak_short)
            لم تعد تُنتَج بعد إصلاح A1 (symmetric single-pair scan) +
            Phase 0 (إزالة weak-event branch). الكودات محفوظة في الـ enum
            للاتساق الخلفي مع readers تاريخية فقط — لا تتوقّع رؤيتها في
            مخرجات جديدة.
        neutral_reason (int8):   0=none, 1=timeout, 4=kalman.
            codes 2,3 (long_sl/short_sl) و 5 (weak_event) محفوظة في الـ enum
            للاتساق الخلفي فقط — الـ scan الجديد لا يصدّرها.
        trade_duration (int16):  عدد bars لنهاية الحدث
        signal_quality (int8):   0=ضعيف, 1=جيد, 2=ممتاز (جلسة لندن/overlap)
        forward_return (float32): عائد إلى إغلاق شمعة نهاية الأفق القصوى المسموحة للصف
            (max_bars/regime/tier) — للتشخيص؛ قد يختلف N بين الصفوف.
            لاختبار IC بهدف أفق ثابت استخدم عمود `fwd_ret_clean` (يُضاف من Refinery).
        event_label_tier (int8): -1 غير مستخدم أو غير حدث؛ 0 ضعيف؛ 1 وسط؛ 2 قوي (عند تفعيل الشرائح)
    """
    df = df.copy().sort_values('ts_event').reset_index(drop=True)
    n = len(df)

    tier_labels_on = (
        bool(DAYTRADE_EVENT_SCORE_TIER_LABELS) if use_event_score_tier_labels is None else bool(use_event_score_tier_labels)
    )
    if float(EVENT_LABEL_SCORE_MID_MIN) > float(EVENT_LABEL_SCORE_STRONG_MIN):
        raise ValueError(
            'regime_config: EVENT_LABEL_SCORE_MID_MIN يجب أن يكون <= EVENT_LABEL_SCORE_STRONG_MIN'
        )

    # ── arrays الخروج ─────────────────────────────────────────────────────────
    bias_label     = np.full(n, 2, dtype=np.int8)
    path_outcome   = np.full(n, 4, dtype=np.int8)   # 4 = timeout
    neutral_reason = np.full(n, NEUTRAL_REASON_NONE, dtype=np.int8)
    trade_duration = np.zeros(n, dtype=np.int16)
    signal_quality = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)
    event_label_tier = np.full(n, -1, dtype=np.int8)

    # ── arrays السعر ──────────────────────────────────────────────────────────
    close  = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=np.float64)
    high   = pd.to_numeric(df['high'],  errors='coerce').to_numpy(dtype=np.float64)
    low    = pd.to_numeric(df['low'],   errors='coerce').to_numpy(dtype=np.float64)
    open_  = pd.to_numeric(df['open'],  errors='coerce').to_numpy(dtype=np.float64) if 'open' in df.columns else close.copy()
    atr    = pd.to_numeric(df['atr_14'], errors='coerce').to_numpy(dtype=np.float64)

    # ── regime + session ──────────────────────────────────────────────────────
    has_regime  = 'regime_label'  in df.columns
    has_session = 'session'       in df.columns
    has_event   = 'is_event'      in df.columns

    regime_arr  = df['regime_label'].astype(str).to_numpy()  if has_regime  else None
    session_arr = df['session'].astype(str).to_numpy()       if has_session else None
    is_ev_arr   = df['is_event'].to_numpy(dtype=np.int8)     if has_event   else np.ones(n, dtype=np.int8)
    ev_score_src = df['event_score'] if 'event_score' in df.columns else pd.Series(0.0, index=df.index)
    kalman_src = df['kalman_direction'] if 'kalman_direction' in df.columns else pd.Series(0, index=df.index)
    event_dir_src = df['event_direction'] if 'event_direction' in df.columns else pd.Series(0, index=df.index)
    event_score_arr = pd.to_numeric(ev_score_src, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    kalman_dir_arr = pd.to_numeric(kalman_src, errors='coerce').fillna(0).to_numpy(dtype=np.int8)
    event_dir_arr = pd.to_numeric(event_dir_src, errors='coerce').fillna(0).to_numpy(dtype=np.int8)

    # جلسات لندن/overlap → signal_quality = 2 (premium)
    london_ok = np.zeros(n, dtype=bool)
    for col in ('is_london', 'is_overlap'):
        if col in df.columns:
            london_ok |= pd.to_numeric(df[col], errors='coerce').fillna(0).astype(bool).to_numpy()

    # ── helper: نهاية الجلسة ────────────────────────────────────────────────
    ts_arr = df['ts_event'].to_numpy()

    def _session_end(i: int, max_bars_r: int) -> int:
        """يحسب آخر index مسموح به (session-end OR regime max_bars، أيهما أصغر)."""
        cap = min(i + max_bars_r, n - 1)
        if session_arr is None:
            return cap

        sess = session_arr[i]
        end_hour = _SESSION_END_HOUR.get(sess, 21)

        for k in range(i + 1, cap + 1):
            ts = ts_arr[k]
            try:
                hour = pd.Timestamp(ts).hour
            except Exception:
                continue
            if hour >= end_hour:
                return k - 1   # آخر bar قبل نهاية الجلسة
        return cap

    # ── الحلقة الرئيسية ───────────────────────────────────────────────────────
    for i in range(n):
        regime_i = str(regime_arr[i]) if has_regime else 'ranging'
        base_max_bars = int(REGIME_MAX_BARS.get(regime_i, default_max_bars))
        tp_mult_base, sl_mult_base = REGIME_TP_SL.get(regime_i, (default_tp_mult, default_sl_mult))

        event_label_tier[i] = -1
        if tier_labels_on and bool(is_ev_arr[i]):
            es = float(event_score_arr[i])
            s_hi = float(EVENT_LABEL_SCORE_STRONG_MIN)
            s_mid = float(EVENT_LABEL_SCORE_MID_MIN)
            if es >= s_hi:
                tp_mult, sl_mult, max_bars_r = EVENT_LABEL_TIER_STRONG
                event_label_tier[i] = 2
            elif es >= s_mid:
                tp_mult, sl_mult, max_bars_r = EVENT_LABEL_TIER_MID
                event_label_tier[i] = 1
            else:
                tp_mult, sl_mult, max_bars_r = EVENT_LABEL_TIER_WEAK
                event_label_tier[i] = 0
            max_bars_r = int(max_bars_r)
        else:
            tp_mult, sl_mult = tp_mult_base, sl_mult_base
            max_bars_r = base_max_bars

        atr_i = max(float(atr[i]), min_atr)
        entry = float(close[i])
        sess_end = _session_end(i, max_bars_r)
        fwd_idx = min(i + max_bars_r, n - 1)
        forward_return[i] = float((close[fwd_idx] - entry) / max(entry, 1e-8))

        # Phase 0 root fix: removed the `if not is_ev_arr[i]: continue` gate.
        # Empirical IC evidence (gate_kill_ratio=0.58, with 5/6 STRONG features
        # showing kill_ratio > 1.0 meaning the gate ANTI-selected) proved the
        # gate was discarding training data that carried at least as much
        # forward-return signal as the bars it kept. Every bar now enters the
        # forward TP/SL scan below; NEUTRAL only arises from genuine path
        # outcomes (timeout, SL-first, or MFE/MAE rescue failing the
        # MFE/MAE ratio + min-move check) — never from the gate.
        # is_event / event_score / event_score_tier columns are still written
        # so the model can use them as features / confidence weights instead
        # of having them gate the labeling.

        # TP/SL للحدث + مسارات الحدث (بعد شرائح event_score إن فُعّلت)
        event_score_i = float(event_score_arr[i])
        kalman_dir_i = int(kalman_dir_arr[i])
        event_dir_i = int(event_dir_arr[i])
        allow_long = True
        allow_short = True
        # event_direction veto (A2): مُعطّل افتراضياً. تفعيله يعيد فرض انحياز
        # اتجاهي فوق الـ scan المتماثل (A1) ويُسقط ~60% من الـ directional signal
        # الحقيقي (الـ voting يخطئ ~70% على real data). يُترك event_direction
        # عموداً يتعلّمه النموذج بدل أن يكون veto صلباً.
        if apply_event_direction_veto:
            if event_dir_i > 0:
                allow_short = False
            elif event_dir_i < 0:
                allow_long = False
            else:
                if kalman_dir_i == 1 and event_score_i < float(kalman_event_floor):
                    allow_short = False
                elif kalman_dir_i == -1 and event_score_i < float(kalman_event_floor):
                    allow_long = False

        # A1 root fix (Finding #7): symmetric single-pair barrier scan.
        # The old four-barrier scan (tp_long, sl_long, tp_short, sl_short) had
        # a fatal LONG-first ordering bug: any downward path crossing tp_short
        # at -tp_dist necessarily crossed sl_long at -sl_dist first (since
        # tp_dist > sl_dist). The code evaluated the LONG side before SHORT,
        # so it captured long_sl and never reached short_tp. Empirical proof:
        # Q2 2025 had 0 short_tp labels out of 17,304 bars — mathematically
        # impossible to short-profit under the old scan. Fix: one symmetric
        # pair (upper, lower) at distance barrier_mult*ATR; first touch wins.
        # On rare same-bar dual hits we tie-break by distance-to-open.
        # path_outcome 2,3,5,6 (long_sl/short_sl/weak_long/weak_short) are
        # no longer produced; codes preserved in the enum for backward read
        # compatibility, but the new scan emits only {0,1,4}.
        barrier_mult = float(tp_mult)
        upper = entry + barrier_mult * atr_i
        lower = entry - barrier_mult * atr_i

        first_ev: tuple[str, int] | None = None

        for j in range(i + 1, sess_end + 1):
            h = float(high[j])
            l = float(low[j])
            o = float(open_[j]) if np.isfinite(open_[j]) else 0.5 * (h + l)

            long_hit  = allow_long  and (h >= upper)
            short_hit = allow_short and (l <= lower)

            if long_hit and short_hit:
                # Same-bar ambiguity: pick barrier closer to bar open as the
                # likely first-touch heuristic (intra-bar order unobservable
                # on bar data).
                d_up = abs(upper - o)
                d_lo = abs(lower - o)
                first_ev = (('long_tp' if d_up <= d_lo else 'short_tp'), j)
                break
            if long_hit:
                first_ev = ('long_tp', j)
                break
            if short_hit:
                first_ev = ('short_tp', j)
                break

        # ── تعيين الليبل ─────────────────────────────────────────────────────
        # MFE/MAE rescue يُستدعى فقط عند timeout (لم يُلمَس barrier صريح). في
        # الـ scan الجديد (symmetric single-pair) أي لمس = TP صريح في اتجاهه؛
        # حالات long_sl/short_sl لم تعد ممكنة فحُذِفت فروعها.
        def _mfe_mae_decide(end_idx: int) -> int:
            if (not timeout_mfe_mae) or end_idx <= i:
                return 2
            win = close[i + 1: end_idx + 1]
            if win.size == 0:
                return 2
            d = win - entry
            mfe = max(0.0, float(np.max(d)))
            mae = max(0.0, float(-np.min(d)))
            min_move = float(timeout_mfe_min_move_atr) * atr_i
            ratio = float(timeout_mfe_mae_ratio)
            if allow_long and mfe > ratio * mae and mfe >= min_move:
                return 0
            if allow_short and mae > ratio * mfe and mae >= min_move:
                return 1
            return 2

        if first_ev is None:
            # Timeout — لم يُلمَس أي barrier → قرار MFE/MAE على كامل النافذة.
            trade_duration[i] = max(0, sess_end - i)
            decided = _mfe_mae_decide(sess_end)
            bias_label[i]   = decided
            path_outcome[i] = 4  # 4 = timeout
            if decided == 2:
                if (not allow_long) or (not allow_short):
                    neutral_reason[i] = NEUTRAL_REASON_KALMAN
                else:
                    neutral_reason[i] = NEUTRAL_REASON_TIMEOUT
            else:
                signal_quality[i] = 2 if london_ok[i] else 1
                neutral_reason[i] = NEUTRAL_REASON_NONE
        else:
            ev_type, ev_j = first_ev
            trade_duration[i] = ev_j - i
            sq = 2 if london_ok[i] else 1

            if ev_type == 'long_tp':
                bias_label[i]   = 0   # LONG ✅ (TP صريح)
                path_outcome[i] = 0
                signal_quality[i] = sq
            elif ev_type == 'short_tp':
                bias_label[i]   = 1   # SHORT ✅ (TP صريح)
                path_outcome[i] = 1
                signal_quality[i] = sq

    # ── كتابة النتائج ─────────────────────────────────────────────────────────
    df['bias_label']     = bias_label
    df['path_outcome']   = path_outcome
    df['neutral_reason'] = neutral_reason
    df['trade_duration'] = trade_duration
    df['signal_quality'] = signal_quality
    df['forward_return'] = forward_return
    df['event_label_tier'] = event_label_tier

    # event_flag: فاز بـ TP فقط (للتدريب الفعلي)
    df['event_flag'] = (
        (df['is_event'] == 1) &
        (df['bias_label'].isin([0, 1])) &
        (df['signal_quality'] > 0)
    ).astype(np.int8)

    # train_event_flag: افتراضيًا directional داخل events فقط.
    if include_weak_directional_in_train:
        df['train_event_flag'] = (df['bias_label'] != 2).astype(np.int8)
    else:
        df['train_event_flag'] = (
            (df['is_event'] == 1) &
            (df['bias_label'] != 2)
        ).astype(np.int8)

    # إحصاءات التشخيص
    lbl_counts = {
        'LONG'    : int((df['bias_label'] == 0).sum()),
        'SHORT'   : int((df['bias_label'] == 1).sum()),
        'NEUTRAL' : int((df['bias_label'] == 2).sum()),
    }
    ev_rate = float(df['event_flag'].mean())
    train_r = float(df['train_event_flag'].mean())
    print(f"  🏷️  Labels: {lbl_counts} | event_flag={ev_rate:.1%} | train_pool={train_r:.1%}")
    if tier_labels_on and has_event:
        ev_m = pd.to_numeric(df['is_event'], errors='coerce').fillna(0).astype(np.int8).to_numpy() == 1
        if bool(ev_m.any()):
            t = df.loc[ev_m, 'event_label_tier'].to_numpy(dtype=np.int64)
            print(
                f"     event_score tiers (among events): "
                f"strong={int((t == 2).sum())} mid={int((t == 1).sum())} weak={int((t == 0).sum())}"
            )

    n0, n1 = lbl_counts['LONG'], lbl_counts['SHORT']
    if min(n0, n1) > 0:
        imb = max(n0, n1) / min(n0, n1)
        print(f"     LONG/SHORT imbalance ratio: {imb:.2f}:1 "
              f"({'✅ متوازن' if imb < 2.0 else '⚠️ يحتاج class_weight'})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Multi-task Label Diagnostics (Phase 1 — Plan الجديدة)
# ══════════════════════════════════════════════════════════════════════════════
# يضيف أعمدة تشخيصية بعد label_by_outcome دون كسر backward compat.
# الهدف: تفصيل سبب NEUTRAL إلى 6 فئات + إضافة tradability_label منفصل
# عن direction، عشان يُدرَّب multi-head MetaLearner لاحقاً.
#
# Cost defaults معايرة لـ 6B (British Pound futures):
DEFAULT_SPREAD_TICKS = 2.0        # bid-ask spread (~1-2 ticks for 6B)
DEFAULT_TICK_SIZE = 0.0001        # 0.0001 USD per tick
DEFAULT_FEES_ATR_PROXY = 0.15     # fees as fraction of ATR (proxy)

NEUTRAL_TYPE_NONE = 0
NEUTRAL_TYPE_NO_TRADE_EDGE = 1
NEUTRAL_TYPE_AMBIGUOUS = 2
NEUTRAL_TYPE_LATE_MOVE = 3
NEUTRAL_TYPE_LOW_EXPECTANCY = 4
NEUTRAL_TYPE_STOP_FIRST = 5
NEUTRAL_TYPE_DATA_QUALITY = 6
NEUTRAL_TYPE_NON_EVENT = 7

NEUTRAL_TYPE_NAMES = {
    0: 'directional',           # bias_label = LONG/SHORT (نظيف)
    1: 'no_trade_edge',         # كلا MFE/MAE < 0.5 ATR (لا حركة)
    2: 'ambiguous',             # كلاهما > 0.5 ATR، متقاربان
    3: 'late_move',             # حركة قوية لكن بعد الـ horizon
    4: 'low_expectancy',        # حركة موجودة لكن أصغر من cost
    5: 'stop_first',            # MAE قبل MFE (SL يُضرب أولاً)
    6: 'data_quality_block',    # ATR/data ناقصة
    7: 'non_event',             # is_event=False (مش مرشح للتداول أصلاً)
}


def compute_multitask_label_diagnostics(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 24,
    spread_ticks: float = DEFAULT_SPREAD_TICKS,
    tick_size: float = DEFAULT_TICK_SIZE,
    fees_atr_proxy: float = DEFAULT_FEES_ATR_PROXY,
) -> pd.DataFrame:
    """يضيف diagnostic columns للـ multi-task training (Phase 1 من Multi-task Plan).

    الأعمدة المُضافة (10 أعمدة):
        mfe, mae                : extremes الخامة بـ price units
        mfe_atr, mae_atr        : normalized بـ ATR
        time_to_first_touch     : عدد bars حتى أول touch لـ 0.5 ATR
        stop_first_flag         : 1 لو MAE > 1 ATR قبل MFE > 1 ATR
        net_expectancy_proxy    : net expected gain بعد spread+fees (in ATR units)
        neutral_type            : سبب الـ NEUTRAL (6 فئات + 0 للـ directional)
        tradability_label       : 0/1 — صف نظيف للتداول
        market_state_label      : 5-class causal (مأخوذ من regime_label الموجود)

    الـ tradability_label = 1 يعني:
        bias_label ∈ {LONG, SHORT}
        AND net_expectancy_proxy > 0 (profitable بعد costs)
        AND stop_first_flag == 0 (لم يُضرب SL قبل TP)
    """
    out = df.copy()
    n = len(out)

    if 'close' not in out.columns:
        print("  ⚠️  multitask_diagnostics: 'close' مفقود — skip")
        return out

    close = pd.to_numeric(out['close'], errors='coerce').to_numpy(dtype=np.float64)
    atr_col = 'atr_14' if 'atr_14' in out.columns else ('atr' if 'atr' in out.columns else None)
    if atr_col is None:
        print("  ⚠️  multitask_diagnostics: 'atr_14'/'atr' مفقود — skip")
        return out
    atr = pd.to_numeric(out[atr_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)

    # ── حساب MFE/MAE + sequence-aware (stop_first) ──────────────────────
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)
    mfe_atr_arr = np.zeros(n, dtype=np.float64)
    mae_atr_arr = np.zeros(n, dtype=np.float64)
    time_to_touch = np.zeros(n, dtype=np.int32)
    stop_first = np.zeros(n, dtype=np.int8)

    for i in range(n):
        end = min(i + horizon_bars + 1, n)
        if end <= i + 1 or atr[i] <= 1e-12:
            continue
        win = close[i + 1: end] - close[i]
        if win.size == 0:
            continue
        mfe_i = max(0.0, float(np.max(win)))
        mae_i = max(0.0, float(-np.min(win)))
        mfe[i] = mfe_i
        mae[i] = mae_i
        mfe_atr_arr[i] = mfe_i / atr[i]
        mae_atr_arr[i] = mae_i / atr[i]

        # stop_first: المسار يضرب SL (1 ATR) قبل TP (1 ATR)
        atr_i = atr[i]
        tp_idx = int(np.argmax(win >= atr_i)) if (win >= atr_i).any() else -1
        sl_idx = int(np.argmax(win <= -atr_i)) if (win <= -atr_i).any() else -1
        if sl_idx >= 0 and (tp_idx < 0 or sl_idx < tp_idx):
            stop_first[i] = 1

        # time_to_first_touch: أول touch لـ 0.5 ATR (أي اتجاه)
        threshold = 0.5 * atr[i]
        hit_mask = np.abs(win) >= threshold
        if hit_mask.any():
            time_to_touch[i] = int(np.argmax(hit_mask)) + 1

    out['mfe'] = mfe.astype(np.float32)
    out['mae'] = mae.astype(np.float32)
    out['mfe_atr'] = mfe_atr_arr.astype(np.float32)
    out['mae_atr'] = mae_atr_arr.astype(np.float32)
    out['time_to_first_touch'] = time_to_touch
    out['stop_first_flag'] = stop_first

    # ── net_expectancy_proxy: gain بعد spread + fees ──────────────────
    cost_atr = float(fees_atr_proxy) + np.where(
        atr > 1e-12,
        spread_ticks * tick_size / np.maximum(atr, 1e-12),
        0.0,
    )
    bias = (
        out['bias_label'].to_numpy() if 'bias_label' in out.columns
        else np.full(n, 2, dtype=np.int8)
    )
    net_exp = np.where(
        bias == 0, mfe_atr_arr - cost_atr,
        np.where(
            bias == 1, mae_atr_arr - cost_atr,
            np.maximum(mfe_atr_arr, mae_atr_arr) - cost_atr,
        ),
    )
    out['net_expectancy_proxy'] = net_exp.astype(np.float32)

    # ── neutral_type: تصنيف الـ NEUTRAL لـ 6 فئات ──────────────────────
    nt = np.zeros(n, dtype=np.int8)  # 0 = directional (سيبقى فقط للـ LONG/SHORT الحقيقية)
    is_neutral = (bias == 2)
    is_event = (
        out['is_event'].to_numpy().astype(bool) if 'is_event' in out.columns
        else np.ones(n, dtype=bool)
    )
    has_atr = atr > 1e-12

    # non_event NEUTRAL: bars خارج pool التداول → فئة منفصلة (مش "directional")
    nt[is_neutral & ~is_event] = NEUTRAL_TYPE_NON_EVENT

    # data_quality_block: ATR ناقص
    nt[is_neutral & is_event & ~has_atr] = NEUTRAL_TYPE_DATA_QUALITY

    # داخل event-NEUTRAL valid (has_atr) — نصنّف
    valid = is_neutral & is_event & has_atr & (nt == 0)
    small = 0.5
    cond_edge = valid & (mfe_atr_arr < small) & (mae_atr_arr < small)
    nt[cond_edge] = NEUTRAL_TYPE_NO_TRADE_EDGE
    cond_stop = valid & (nt == 0) & (stop_first == 1)
    nt[cond_stop] = NEUTRAL_TYPE_STOP_FIRST
    cond_low_exp = valid & (nt == 0) & (mfe_atr_arr >= small) & (net_exp < 0)
    nt[cond_low_exp] = NEUTRAL_TYPE_LOW_EXPECTANCY
    cond_late = valid & (nt == 0) & (mfe_atr_arr > 2.0) & (time_to_touch > int(horizon_bars * 0.7))
    nt[cond_late] = NEUTRAL_TYPE_LATE_MOVE
    cond_amb = valid & (nt == 0) & (mfe_atr_arr > small) & (mae_atr_arr > small)
    nt[cond_amb] = NEUTRAL_TYPE_AMBIGUOUS
    # fallback: أي NEUTRAL باقي → ambiguous
    nt[valid & (nt == 0)] = NEUTRAL_TYPE_AMBIGUOUS

    out['neutral_type'] = nt

    # ── tradability_label: clean trade indicator ─────────────────────
    is_directional = (bias == 0) | (bias == 1)
    is_profitable = net_exp > 0
    is_clean = stop_first == 0
    out['tradability_label'] = (is_directional & is_profitable & is_clean).astype(np.int8)

    # ── market_state_label: 5-class (mapped من regime_label الموجود) ──
    # Mapping:
    #   trending      → 'trend'
    #   ranging       → 'chop'
    #   volatile      → 'volatile'
    #   low_liquidity → 'liquidity_vacuum'
    #   (mean_reversion سيُضاف لاحقاً عبر classifier منفصل)
    regime_to_state = {
        'trending': 'trend',
        'ranging': 'chop',
        'volatile': 'volatile',
        'low_liquidity': 'liquidity_vacuum',
    }
    if 'regime_label' in out.columns:
        out['market_state_label'] = out['regime_label'].astype(str).map(regime_to_state).fillna('chop')
        state_codes = {'trend': 0, 'chop': 1, 'volatile': 2, 'liquidity_vacuum': 3, 'mean_reversion': 4}
        out['market_state_code'] = out['market_state_label'].map(state_codes).fillna(1).astype(np.int8)

    # ── Diagnostics print ────────────────────────────────────────────
    print("\n📊 Multi-task Diagnostics:")
    print("─" * 60)
    if 'tradability_label' in out.columns:
        trad = int(out['tradability_label'].sum())
        print(f"  tradability_label = 1: {trad:,} ({trad/n*100:.1f}%)")
    nt_counts = pd.Series(nt).value_counts().sort_index()
    print(f"  neutral_type distribution:")
    for code, name in NEUTRAL_TYPE_NAMES.items():
        cnt = int(nt_counts.get(code, 0))
        if cnt > 0:
            print(f"    {name:22s}: {cnt:>6,} ({cnt/n*100:5.1f}%)")
    print(f"  MFE in ATR units: median={np.median(mfe_atr_arr[mfe_atr_arr>0]):.2f}, "
          f"p90={np.percentile(mfe_atr_arr[mfe_atr_arr>0], 90):.2f}")
    print(f"  stop_first events: {int(stop_first.sum()):,} "
          f"({stop_first.sum()/n*100:.1f}%)")
    print(f"  net_expectancy_proxy > 0: {int((net_exp > 0).sum()):,} "
          f"({(net_exp > 0).sum()/n*100:.1f}%)")
    print("─" * 60)

    return out


def compute_strict_execution_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 6,
    barrier_atr_mult: float = 1.5,
    min_atr: float = 0.0003,
) -> pd.DataFrame:
    """B1 — MT5-faithful strict triple-barrier execution labels.

    Head 2 of the dual-target pipeline. While bias_label (label_by_outcome)
    is the directional-bias target the SSL backbone / bias model learns —
    and may carry an MFE/MAE rescue on timeout — the execution head needs
    a *tradeable* label with no historical look-back: if a barrier is
    touched, the first touch is frozen; if not, the row is NEUTRAL and
    masked out (never rescued from the horizon close).

    Delegates to label_engine_v2.label_triple_barrier_atr_vectorized with
    rescue_on_timeout=False so it shares A1's no-short_tp-bug barrier
    geometry. Additive: does not read or modify bias_label / path_outcome
    or any existing column.

    Session-break aware: a row whose forward window (i+1 .. i+horizon)
    crosses an is_session_break boundary is forced exec_valid=False — a
    trade cannot span a session gap, so the strict outcome is undefined.

    Columns added:
        exec_label (int8) : 0=LONG, 1=SHORT, 2=NEUTRAL (strict first-touch)
        exec_path  (int8) : label_engine_v2 PATH_* (1=TP_LONG, 2=TP_SHORT,
                            5=TIMEOUT_NEUTRAL); rescue paths never appear.
        exec_valid (bool) : True iff a hard barrier was touched AND the
                            forward window did not cross a session break.
                            This is the per-bar loss mask the execution
                            head filters on.
    """
    from modules.label_engine_v2 import (
        label_triple_barrier_atr_vectorized,
        TripleBarrierConfig,
    )

    out = df.copy()
    n = len(out)
    if n == 0 or 'close' not in out.columns:
        out['exec_label'] = np.zeros(n, dtype=np.int8)
        out['exec_path'] = np.zeros(n, dtype=np.int8)
        out['exec_valid'] = np.zeros(n, dtype=bool)
        return out

    close = pd.to_numeric(out['close'], errors='coerce').to_numpy(dtype=np.float64)
    atr_col = 'atr_14' if 'atr_14' in out.columns else ('atr' if 'atr' in out.columns else None)
    if atr_col is None:
        print("  ⚠️  strict_execution_labels: 'atr_14'/'atr' مفقود — skip")
        out['exec_label'] = np.full(n, 2, dtype=np.int8)
        out['exec_path'] = np.zeros(n, dtype=np.int8)
        out['exec_valid'] = np.zeros(n, dtype=bool)
        return out
    atr = pd.to_numeric(out[atr_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    atr = np.maximum(atr, float(min_atr))

    horizons = np.full(n, int(horizon_bars), dtype=np.int32)
    cfg = TripleBarrierConfig(
        tp_atr_mult=float(barrier_atr_mult),
        sl_atr_mult=float(barrier_atr_mult),   # symmetric pair (matches A1)
        min_move_atr_mult=0.0,                  # irrelevant with rescue off
        horizon_default=int(horizon_bars),
        rescue_on_timeout=False,                # ← strict: no historical look-back
    )
    res = label_triple_barrier_atr_vectorized(close, atr, horizons=horizons, config=cfg)
    exec_label = res['bias'].astype(np.int8)
    exec_path = res['path'].astype(np.int8)
    exec_valid = res['valid'].astype(bool)

    # ── session-break invalidation ────────────────────────────────────
    # window for row i = bars (i+1 .. i+horizon]; invalidate if any is a break.
    if 'is_session_break' in out.columns:
        sb = pd.to_numeric(out['is_session_break'], errors='coerce').fillna(0).to_numpy().astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(sb)])   # csum[k] = breaks in [0,k)
        idx = np.arange(n)
        lo = np.minimum(idx + 1, n)
        hi = np.minimum(idx + 1 + int(horizon_bars), n)
        window_break = (csum[hi] - csum[lo]) > 0
        exec_valid = exec_valid & ~window_break

    out['exec_label'] = exec_label
    out['exec_path'] = exec_path
    out['exec_valid'] = exec_valid

    n_long = int((exec_label == 0).sum())
    n_short = int((exec_label == 1).sum())
    n_valid = int(exec_valid.sum())
    print("\n🎯 Strict Execution Labels (B1 — dual-target head 2):")
    print("─" * 60)
    print(f"  exec_valid (tradeable, masked-in): {n_valid:,} ({n_valid/max(n,1)*100:.1f}%)")
    print(f"  exec LONG / SHORT (both directions reachable): {n_long:,} / {n_short:,}")
    print(f"  exec NEUTRAL (masked-out):         {int((exec_label==2).sum()):,} "
          f"({(exec_label==2).sum()/max(n,1)*100:.1f}%)")
    print("─" * 60)

    return out


def compute_ssl_directional_target(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 6,
    use_log: bool = True,
    clip: float = 0.1,
) -> pd.DataFrame:
    """B2 — continuous directional SSL target over a fixed horizon.

    Head 1 of the dual-target pipeline. Where exec_label (B1) is the
    discrete, masked, tradeable label for the execution head, this is the
    continuous "where is price going" target the SSL backbone learns on
    EVERY bar — no event gate, no barrier, no NEUTRAL. Even dead chop bars
    get a (microscopic) directional value so the backbone keeps learning
    market context continuously.

        next_price_delta[i] = log(close[i+H] / close[i])     (use_log=True)
                            = close[i+H] / close[i] - 1       (use_log=False)

    Distinct from the SSL data_loader's on-the-fly `next_price` (a 1-bar
    log-return feeding the multi-task next_price head): this is the H-bar
    directional-bias target, persisted in the parquet so the dual-target
    is explicit and auditable alongside exec_label.

    Causal & gate-free, but right-edge / session-break aware:
        next_price_delta_valid[i] = False when i+H runs off the dataset OR
        the window (i+1 .. i+H] crosses an is_session_break (a horizon
        that spans a weekend/holiday gap is not a real H-bar move).

    Columns added:
        next_price_delta       (float32)
        next_price_delta_valid (bool)
    """
    out = df.copy()
    n = len(out)
    if n == 0 or 'close' not in out.columns:
        out['next_price_delta'] = np.zeros(n, dtype=np.float32)
        out['next_price_delta_valid'] = np.zeros(n, dtype=bool)
        return out

    H = max(int(horizon_bars), 1)
    close_raw = pd.to_numeric(out['close'], errors='coerce').to_numpy(dtype=np.float64)
    # CAUSAL fill only (ffill) — never copy a future close backwards.
    close = pd.Series(close_raw).ffill().to_numpy(dtype=np.float64)

    delta = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n > H:
        c0 = close[:-H]
        cH = close[H:]
        ok = np.isfinite(c0) & np.isfinite(cH) & (c0 > 0) & (cH > 0)
        if use_log:
            vals = np.log(np.maximum(cH, 1e-9) / np.maximum(c0, 1e-9))
        else:
            vals = cH / np.maximum(c0, 1e-9) - 1.0
        delta[:-H] = np.where(ok, vals, 0.0)
        valid[:-H] = ok

    delta = np.clip(delta, -float(clip), float(clip))

    # session-break invalidation
    if 'is_session_break' in out.columns:
        sb = pd.to_numeric(out['is_session_break'], errors='coerce').fillna(0).to_numpy().astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(sb)])
        idx = np.arange(n)
        lo = np.minimum(idx + 1, n)
        hi = np.minimum(idx + 1 + H, n)
        window_break = (csum[hi] - csum[lo]) > 0
        valid = valid & ~window_break

    out['next_price_delta'] = delta.astype(np.float32)
    out['next_price_delta_valid'] = valid

    n_valid = int(valid.sum())
    pos = int((delta[valid] > 0).sum()) if n_valid else 0
    neg = int((delta[valid] < 0).sum()) if n_valid else 0
    print(f"\n📈 SSL Directional Target (B2 — dual-target head 1, H={H} bars):")
    print("─" * 60)
    print(f"  next_price_delta_valid: {n_valid:,} ({n_valid/max(n,1)*100:.1f}%) "
          f"— gate-free (every bar gets a value)")
    print(f"  sign split (valid): up={pos:,} / down={neg:,}")
    print("─" * 60)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Day Trading Labels
# ══════════════════════════════════════════════════════════════════════════════


def _apply_train_event_pool(
    df: pd.DataFrame,
    *,
    strict_session_atr: bool,
) -> pd.DataFrame:
    """
    train_event_flag: افتراضيًا كل LONG/SHORT (فوز TP فقط — خسائر SL تصبح NEUTRAL بعد إصلاح الليبل).

    لو strict_session_atr=True: يضيق المجموعة إلى جلسات نشطة + حد أدنى للـ ATR.
    """
    out = df.copy()
    base = (out['bias_label'] != 2).astype(np.int8)
    if not strict_session_atr:
        out['train_event_flag'] = base
        return out

    atr_median = float(pd.to_numeric(out['atr_14'], errors='coerce').median() or 0.0)
    floor = max(atr_median * 0.5, 1e-12)
    strong_session = (
        (pd.to_numeric(out.get('is_london', 0), errors='coerce').fillna(0) > 0)
        | (pd.to_numeric(out.get('is_overlap', 0), errors='coerce').fillna(0) > 0)
        | (pd.to_numeric(out.get('is_ny', 0), errors='coerce').fillna(0) > 0)
    )
    above_atr = pd.to_numeric(out['atr_14'], errors='coerce').fillna(0.0) >= floor
    out['train_event_flag'] = (base.astype(bool) & strong_session & above_atr).astype(np.int8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Soft Labels (Regime-aware + engine)
# ══════════════════════════════════════════════════════════════════════════════

def add_soft_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soft labels day-trading-aware:
      LONG  -> [0.55 .. 1.00]
      SHORT -> [0.00 .. 0.45]
      NEUTRAL -> 0.50
    القوة = 0.6*event_score + 0.4*speed (مدة أقصر = أقوى).
    """
    out = df.copy()
    n = len(out)
    soft = np.full(n, 0.5, dtype=np.float32)
    labels_src = out['bias_label'] if 'bias_label' in out.columns else pd.Series(2, index=out.index)
    events_src = out['is_event'] if 'is_event' in out.columns else pd.Series(0, index=out.index)
    score_src = out['event_score'] if 'event_score' in out.columns else pd.Series(0.0, index=out.index)
    duration_src = out['trade_duration'] if 'trade_duration' in out.columns else pd.Series(0, index=out.index)
    labels = pd.to_numeric(labels_src, errors='coerce').fillna(2).to_numpy(dtype=np.int8)
    events = pd.to_numeric(events_src, errors='coerce').fillna(0).to_numpy(dtype=np.int8)
    ev_score = pd.to_numeric(score_src, errors='coerce').fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64)
    duration = pd.to_numeric(duration_src, errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    regime_s = out.get('regime_label', pd.Series('ranging', index=out.index)).astype(str).to_numpy()
    for i in np.where(events == 1)[0]:
        label_i = int(labels[i])
        if label_i == 2:
            continue
        regime_i = str(regime_s[i]) if i < len(regime_s) else 'ranging'
        max_dur = float(max(int(REGIME_MAX_BARS.get(regime_i, 6)), 1))
        speed = float(np.clip(1.0 - (duration[i] / max_dur), 0.0, 1.0))
        strength = float(np.clip(ev_score[i] * 0.6 + speed * 0.4, 0.0, 1.0))
        if label_i == 0:
            soft[i] = np.float32(0.55 + strength * 0.45)
        elif label_i == 1:
            soft[i] = np.float32(0.45 - strength * 0.45)
    conf = np.clip(np.abs(soft - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    out['soft_label'] = soft
    out['label_confidence'] = conf
    out['soft_sample_weight'] = (1.0 + conf).astype(np.float32)
    return out


def attach_soft_labels_dt(
    df: pd.DataFrame,
    *,
    soft_label_mode: str = 'analytical',
    soft_label_n_scenarios: int = 200,
    soft_label_horizon_std: float = 0.30,
    soft_label_tp_std: float = 0.20,
    soft_label_sl_std: float = 0.20,
    soft_label_random_seed: int = 42,
) -> pd.DataFrame:
    """
    يطبق soft labels على مستوى الـ bars:
      1) day-trading regime-aware soft labels (مضمون دائماً)
      2) enrich اختياري عبر soft_label_engine (إذا متاح) دون فقد soft_label الأساسي
    """
    base = add_soft_labels(df)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from modules.soft_label_engine import SoftLabelEngine, SoftLabelConfig
        mode = str(soft_label_mode).strip().lower()
        if mode not in {'analytical', 'monte_carlo'}:
            print(f"  ⚠️ soft_label_mode غير معروف ({soft_label_mode}) — fallback إلى analytical")
            mode = 'analytical'
        config = SoftLabelConfig(
            mode=mode,
            n_scenarios=int(max(1, soft_label_n_scenarios)),
            horizon_std=float(max(0.0, soft_label_horizon_std)),
            tp_std=float(max(0.0, soft_label_tp_std)),
            sl_std=float(max(0.0, soft_label_sl_std)),
            random_seed=int(soft_label_random_seed),
        )
        engine = SoftLabelEngine(config=config)
        enriched = engine.attach_soft_labels(base.copy())
        # Keep backward-compatible behavior for analytical mode (base regime-aware soft).
        # For monte_carlo mode, preserve engine-produced soft_label so training
        # actually sees MC separation (instead of being overwritten by base soft).
        if mode == 'analytical':
            enriched['soft_label'] = base['soft_label'].astype(np.float32)
        else:
            enriched['soft_label'] = pd.to_numeric(
                enriched.get('soft_label', base['soft_label']),
                errors='coerce',
            ).fillna(base['soft_label']).astype(np.float32)
        base_conf = pd.to_numeric(base.get('label_confidence', 0.5), errors='coerce').fillna(0.5).astype(np.float32)
        if 'label_confidence' in enriched.columns:
            eng_conf = pd.to_numeric(enriched['label_confidence'], errors='coerce').fillna(0.5).astype(np.float32)
            enriched['label_confidence'] = np.maximum(eng_conf, base_conf).astype(np.float32)
        else:
            enriched['label_confidence'] = base_conf
        if 'soft_sample_weight' in enriched.columns:
            eng_w = pd.to_numeric(enriched['soft_sample_weight'], errors='coerce').fillna(1.0).astype(np.float32)
            scale = (0.75 + 0.5 * base_conf).astype(np.float32)
            enriched['soft_sample_weight'] = (eng_w * scale).astype(np.float32)
        else:
            enriched['soft_sample_weight'] = base['soft_sample_weight'].astype(np.float32)
        print(f"  ✅ Soft Labels مُرفقة (day-trading regime-aware + {mode} enrich)")
        return enriched
    except Exception as e:
        print(f"  ⚠️ Soft label engine غير متاح: {e} — using day-trading regime-aware soft labels فقط.")
        return base


def _build_daytrade_split_context(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.80,
) -> dict:
    n = int(len(df))
    if n == 0:
        return {
            'split_idx': 0,
            'split_time': pd.NaT,
            'train_row_ok': np.array([], dtype=bool),
            'holdout_row_ok': np.array([], dtype=bool),
            'purged_row_ok': np.array([], dtype=bool),
        }

    ts = pd.to_datetime(df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"❌ Invalid ts_event rows in daytrade artifact: {bad}")
    if 'label_end_ts' in df.columns:
        label_end = pd.to_datetime(df['label_end_ts'], utc=True, errors='coerce').dt.tz_localize(None)
        label_end = label_end.fillna(ts)
    else:
        label_end = ts.copy()

    split_idx = min(max(int(n * float(train_frac)), 1), max(n - 1, 1))
    split_time = ts.iloc[min(split_idx, n - 1)]
    row_ids = np.arange(n)
    train_row_ok = (row_ids < split_idx) & (label_end.values < split_time.to_datetime64())
    holdout_row_ok = row_ids >= split_idx
    purged_row_ok = (~train_row_ok) & (~holdout_row_ok)

    if not bool(np.any(train_row_ok)):
        train_row_ok = row_ids < split_idx
        purged_row_ok = (~train_row_ok) & (~holdout_row_ok)

    return {
        'split_idx': int(split_idx),
        'split_time': split_time,
        'train_row_ok': train_row_ok.astype(bool),
        'holdout_row_ok': holdout_row_ok.astype(bool),
        'purged_row_ok': purged_row_ok.astype(bool),
    }


def _build_daytrade_contract(
    *,
    df: pd.DataFrame,
    mbo_path: str,
    mbp_path: str | None,
    output_dir: str,
    freq: str,
    horizon_bars: int,
    session_profile: str,
    split_meta: dict,
) -> dict:
    ts = pd.to_datetime(df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    ts_min = None if ts.empty else str(ts.min())
    ts_max = None if ts.empty else str(ts.max())
    contract_seed = {
        'source_csv': os.path.abspath(str(mbo_path)),
        'mbp_source': None if not mbp_path else os.path.abspath(str(mbp_path)),
        'rows': int(len(df)),
        'freq': str(freq),
        'horizon_bars': int(horizon_bars),
        'session_profile': str(session_profile),
        'ts_min': ts_min,
        'ts_max': ts_max,
        'split_time': split_meta.get('split_time'),
    }
    dataset_id = hashlib.sha256(
        json.dumps(contract_seed, sort_keys=True).encode('utf-8'),
    ).hexdigest()[:16]
    # C2: schema_version bumped to phase1-c2 (Phase 0 + A1 + A2 + B1 + B2
    # + C1 + C2 all live). The dataset_meta.json sidecar carries the full
    # label-spec list; this contract field is the short tag for manifests.
    return {
        'source_csv': os.path.abspath(str(mbo_path)),
        'mbp_source': None if not mbp_path else os.path.abspath(str(mbp_path)),
        'dataset_id': dataset_id,
        'schema_version': 'phase1-c2',
        'label_mode': 'v19-daytrading',
        'session_profile': str(session_profile),
        'split_time': split_meta.get('split_time'),
        'ts_min': ts_min,
        'ts_max': ts_max,
        'output_dir': os.path.abspath(output_dir),
    }


def _sanitize_artifact_tag(tag: str | None) -> str | None:
    """آمن لأسماء الملفات: حروف أرقام و - _ . فقط."""
    if tag is None:
        return None
    s = str(tag).strip()
    if not s:
        return None
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)
    cleaned = cleaned.strip("._-")
    return cleaned or None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_PREFLIGHT_META_SCHEMA = "day_trading_preflight_v1"


def _normalize_ts_event_series(raw: pd.Series) -> pd.Series:
    """توحيد الطابع الزمني إلى naive UTC (متسق مع باقي مسار اليوم)."""
    return pd.to_datetime(raw, utc=True, errors='coerce').dt.tz_localize(None)


def _sanitize_mbo_ticks_for_daytrade(df: pd.DataFrame) -> pd.DataFrame:
    """
    مواءمة تقارير جودة MBO: فرز، إسقاط ts_event/price الفاسدة، حذف التكرار الكامل،
    ثم حذف تكرار المفتاح عند توفر أعمدة تفريق (sequence / order_id + symbol أو instrument_id).
    لا يُجمّع صفوف مختلفة لها نفس الطابع الزمني إلا إذا وُجد مفتاح صريح لذلك.
    """
    if df.empty:
        return df
    out = df.copy()
    n0 = len(out)
    if 'ts_event' not in out.columns:
        raise KeyError('MBO requires ts_event')
    out['ts_event'] = _normalize_ts_event_series(out['ts_event'])
    out = out.dropna(subset=['ts_event'])
    if 'price' in out.columns:
        px = pd.to_numeric(out['price'], errors='coerce')
        ok_px = px.notna() & np.isfinite(px.to_numpy(dtype=np.float64)) & (px > 0)
        out = out.loc[ok_px].copy()
    sym_col = 'symbol' if 'symbol' in out.columns else ('instrument_id' if 'instrument_id' in out.columns else None)
    sort_cols: list[str] = ['ts_event']
    for extra in ('sequence', 'seq', 'sequence_id', 'order_id'):
        if extra in out.columns:
            sort_cols.append(extra)
            break
    out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    n1 = len(out)
    out = out.drop_duplicates(keep='last').reset_index(drop=True)
    n2 = len(out)
    subset_candidates: tuple[tuple[str, ...], ...] = ()
    if sym_col is not None:
        subset_candidates = (
            (sym_col, 'ts_event', 'sequence'),
            (sym_col, 'ts_event', 'seq'),
            (sym_col, 'ts_event', 'sequence_id'),
            (sym_col, 'ts_event', 'order_id'),
        )
    n3 = len(out)
    for sub in subset_candidates:
        if all(c in out.columns for c in sub):
            out = out.drop_duplicates(subset=list(sub), keep='last').reset_index(drop=True)
            print(
                f"  [sanitize MBO] rows {n0:,} -> {n1:,} (valid ts/price) -> {n2:,} (dedupe full) "
                f"-> {len(out):,} (dedupe key {sub})",
            )
            return out
    print(
        f"  [sanitize MBO] rows {n0:,} -> {n1:,} (valid ts/price) -> {n2:,} (dedupe full); "
        "no sequence/order key cols — skipped key dedupe",
    )
    return out


def _sanitize_mbp_ticks_for_daytrade(df: pd.DataFrame) -> pd.DataFrame:
    """فرز، تكرار كامل، إسقاط BBO المتقاطع، ثم تكرار مفتاح عند symbol+sequence إن وُجد."""
    if df.empty:
        return df
    out = df.copy()
    n0 = len(out)
    if 'ts_event' not in out.columns:
        raise ValueError('MBP requires ts_event')
    out['ts_event'] = _normalize_ts_event_series(out['ts_event'])
    out = out.dropna(subset=['ts_event'])
    sort_cols: list[str] = ['ts_event']
    for extra in ('sequence', 'seq', 'sequence_id'):
        if extra in out.columns:
            sort_cols.append(extra)
            break
    out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    out = out.drop_duplicates(keep='last').reset_index(drop=True)
    n1 = len(out)
    if 'bid_px_00' in out.columns and 'ask_px_00' in out.columns:
        bid = pd.to_numeric(out['bid_px_00'], errors='coerce')
        ask = pd.to_numeric(out['ask_px_00'], errors='coerce')
        crossed = (ask < bid) & bid.notna() & ask.notna()
        out = out.loc[~crossed].reset_index(drop=True)
    n2 = len(out)
    sym_col = 'symbol' if 'symbol' in out.columns else ('instrument_id' if 'instrument_id' in out.columns else None)
    if sym_col is not None:
        for sub in (
            (sym_col, 'ts_event', 'sequence'),
            (sym_col, 'ts_event', 'seq'),
            (sym_col, 'ts_event', 'sequence_id'),
        ):
            if all(c in out.columns for c in sub):
                out = out.drop_duplicates(subset=list(sub), keep='last').reset_index(drop=True)
                print(
                    f"  [sanitize MBP] rows {n0:,} -> {n1:,} (dedupe full) -> {n2:,} (drop crossed BBO) "
                    f"-> {len(out):,} (dedupe key {sub})",
                )
                return out
    print(
        f"  [sanitize MBP] rows {n0:,} -> {n1:,} (dedupe full) -> {n2:,} (drop crossed BBO); "
        "no symbol+sequence — skipped key dedupe",
    )
    return out


def load_mbo_ticks_enriched(
    mbo_dir: str,
    *,
    sanitize_input_ticks: bool = True,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """تحميل MBO (مجلد أو ملف) و enrich_mbo_with_core_microstructure."""
    mbo_path = os.path.abspath(str(mbo_dir))
    ext = os.path.splitext(mbo_path)[1].lower()
    single_file_exts = (".csv", ".gz", ".zst", ".parquet", ".pq", ".snappy")

    if os.path.isfile(mbo_path):
        if ext in (".csv", ".gz", ".zst"):
            df_mbo = pd.read_csv(mbo_path, low_memory=False, compression="infer")
        elif ext in (".parquet", ".pq", ".snappy"):
            df_mbo = pd.read_parquet(mbo_path)
        else:
            raise FileNotFoundError(f"❌ صيغة ملف MBO غير مدعومة: {mbo_path}")
    elif os.path.isdir(mbo_path):
        files = sorted(glob.glob(os.path.join(mbo_path, "mbo_final_*.parquet")))
        if not files:
            files = sorted(glob.glob(os.path.join(mbo_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"❌ لا توجد ملفات parquet في المجلد: {mbo_path}")
        chunks = [pd.read_parquet(f) for f in files]
        df_mbo = pd.concat(chunks, ignore_index=True)
    elif ext in single_file_exts:
        raise FileNotFoundError(
            f"❌ ملف MBO غير موجود على الديسك: {mbo_path}\n"
            "   استخدم مسارًا صحيحًا لملف csv/parquet، أو مجلدًا يحوي shards (*.parquet). "
            "مثال بعد refinery: pipeline_*\\normalized\\mbo"
        )
    else:
        raise FileNotFoundError(
            f"❌ مسار MBO غير صالح (لا ملف ولا مجلد): {mbo_path}"
        )
    if sanitize_input_ticks:
        df_mbo = _sanitize_mbo_ticks_for_daytrade(df_mbo)
    else:
        df_mbo["ts_event"] = pd.to_datetime(df_mbo["ts_event"])
        df_mbo = df_mbo.sort_values("ts_event").reset_index(drop=True)
    df_mbo = enrich_mbo_with_core_microstructure(df_mbo, n_workers=n_workers)
    return df_mbo


def _write_preflight_checkpoint(
    df_bars: pd.DataFrame,
    *,
    output_dir: str,
    suffix: str,
    freq: str,
    session_profile: str,
) -> str:
    preflight_fn = f"day_trading_preflight{suffix}.parquet"
    preflight_path = os.path.join(output_dir, preflight_fn)
    df_bars.to_parquet(preflight_path, index=False)
    meta = {
        "schema": _PREFLIGHT_META_SCHEMA,
        "freq": freq,
        "session_profile": session_profile,
        "n_rows": int(len(df_bars)),
        "n_cols": int(len(df_bars.columns)),
        "ts_min": str(df_bars["ts_event"].min()),
        "ts_max": str(df_bars["ts_event"].max()),
        "artifact_suffix": suffix,
    }
    stem = os.path.splitext(preflight_fn)[0]
    meta_path = os.path.join(output_dir, f"{stem}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return preflight_path


def _load_preflight_checkpoint(
    preflight_parquet: str,
    *,
    freq: str,
    session_profile: str,
) -> pd.DataFrame:
    path = os.path.abspath(str(preflight_parquet))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ preflight parquet not found: {path}")
    base = os.path.splitext(os.path.basename(path))[0]
    meta_path = os.path.join(os.path.dirname(path), f"{base}_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema") != _PREFLIGHT_META_SCHEMA:
            print(f"  ⚠️ preflight meta schema={meta.get('schema')!r} (expected {_PREFLIGHT_META_SCHEMA!r})")
        if meta.get("freq") != freq:
            raise ValueError(
                f"❌ Preflight freq mismatch: checkpoint={meta.get('freq')!r} CLI={freq!r}"
            )
        if meta.get("session_profile") != session_profile:
            raise ValueError(
                f"❌ Preflight session_profile mismatch: checkpoint={meta.get('session_profile')!r} "
                f"CLI={session_profile!r} — أعد بناء preflight بنفس الملف الشخصي للجلسة."
            )
    else:
        print(f"  ⚠️ لا يوجد {meta_path} — تخطي التحقق من freq/session_profile")
    df = pd.read_parquet(path)
    if "ts_event" not in df.columns:
        raise ValueError("❌ preflight parquet يجب أن يحتوي عمود ts_event")
    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    return df


def add_fwd_ret_clean_column(
    df: pd.DataFrame,
    n_bars: int,
    *,
    col_name: str = "fwd_ret_clean",
) -> pd.DataFrame:
    """
    عائد أمامي سببي بسيط بنفس تعريف pandas shift(-N):

        (close[t+N] - close[t]) / close[t]

    — لا يعتمد على مسار TP/SL أو على max_bars المتغير بين الصفوف (شرائح الحدث/regime).
    مناسب لتشخيص IC في alpha_validation دون أن يُعرَّف الهدف بنفس منطق الليبل.

    آخر N صفًا تكون NaN (لا يوجد مستقبل كافٍ).
    """
    out = df.copy()
    c = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
    n = int(max(n_bars, 0))
    arr = np.full(len(c), np.nan, dtype=np.float32)
    if n > 0 and len(c) > n:
        lo = c[:-n]
        hi = c[n:]
        m = np.isfinite(lo) & np.isfinite(hi) & (np.abs(lo) > 1e-12)
        seg = np.where(m, (hi / lo - 1.0).astype(np.float64), np.nan).astype(np.float32)
        arr[: len(seg)] = seg
    out[col_name] = arr
    return out


# الأعمدة الـ 12 التي يضيفها _attach_price_cycle_features (4 swing + 5 phase + 3 fractal)
_PRICE_CYCLE_STRUCTURAL_COLS: tuple[str, ...] = (
    'cycle_structure_score',     # HH/HL/LH/LL pattern strength ∈ [-1, 1]
    'cycle_bars_since_swing',    # normalized [0, 1]
    'cycle_trend_maturity',      # young/mature/exhausted ∈ {0, 0.5, 1}
    'cycle_momentum_decay',      # 0 = strong momentum, 1 = decayed
    'cycle_phase_acc_prob',      # P(Accumulation) Wyckoff
    'cycle_phase_markup_prob',   # P(Markup)
    'cycle_phase_dist_prob',     # P(Distribution)
    'cycle_phase_markdown_prob', # P(Markdown)
    'cycle_position',            # موقع داخل الدورة ∈ [0, 1]
    'cycle_hurst',               # Hurst exponent (>0.5=trending, <0.5=mean-revert)
    'cycle_fractal_dim',         # fractal dimension
    'cycle_mtf_alignment',       # multi-timeframe (1×/4×/16×) alignment
)


def _attach_price_cycle_features(df_bars: pd.DataFrame) -> pd.DataFrame:
    """يضيف 12 ميزة بنيوية من modules/price_cycle (Wyckoff + swing + fractal).

    يحتاج أعمدة OHLCV في df_bars. كل الميزات سببية (لا look-ahead).
    Skip بأمان لو N < 20 (لازم >= ATR window=14 للـ derived features).
    """
    from modules.price_cycle.data_structures import BarSequence
    from modules.price_cycle.feature_pipeline import build_cycle_features

    required = ('ts_event', 'open', 'high', 'low', 'close', 'volume')
    missing = [c for c in required if c not in df_bars.columns]
    if missing:
        raise KeyError(f"price_cycle: أعمدة مفقودة: {missing}")

    if len(df_bars) < 20:
        print(f"  ⚠️  price_cycle: skip (N={len(df_bars)} < 20 minimum)")
        for col in _PRICE_CYCLE_STRUCTURAL_COLS:
            if col not in df_bars.columns:
                df_bars[col] = np.float32(0.0)
        return df_bars

    ts_ns = pd.to_datetime(df_bars['ts_event']).astype('datetime64[ns]').astype(np.int64).to_numpy()
    bars_seq = BarSequence(
        timestamps_ns=ts_ns,
        open  = pd.to_numeric(df_bars['open'],   errors='coerce').fillna(0.0).to_numpy(dtype=np.float64),
        high  = pd.to_numeric(df_bars['high'],   errors='coerce').fillna(0.0).to_numpy(dtype=np.float64),
        low   = pd.to_numeric(df_bars['low'],    errors='coerce').fillna(0.0).to_numpy(dtype=np.float64),
        close = pd.to_numeric(df_bars['close'],  errors='coerce').fillna(0.0).to_numpy(dtype=np.float64),
        volume= pd.to_numeric(df_bars['volume'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64),
    )

    feats = build_cycle_features(bars_seq, generate_weak_labels=False)
    structural = feats.structural_features  # (T, 12)
    assert structural.shape[1] == len(_PRICE_CYCLE_STRUCTURAL_COLS), (
        f"price_cycle: expected {len(_PRICE_CYCLE_STRUCTURAL_COLS)} structural "
        f"features, got {structural.shape[1]}"
    )

    n_added = 0
    for i, col in enumerate(_PRICE_CYCLE_STRUCTURAL_COLS):
        if col not in df_bars.columns:
            df_bars[col] = structural[:, i].astype(np.float32)
            n_added += 1
    print(f"  ✅ +{n_added} cycle structural features (swing + Wyckoff + fractal)")
    return df_bars


def run_day_trading_refinery(
    mbo_dir: str | None,
    mbp_path: str | None,
    output_dir: str,
    freq: str = DAY_TRADE_DEFAULT_BAR_FREQ,
    horizon_bars: int = 6,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    build_lob_tensors: bool = True,
    *,
    session_profile: str = 'daytrade_default',
    strict_train_pool: bool = False,
    event_threshold_scale: float = 1.0,
    event_threshold_shift: float = 0.0,
    min_event_rate: float = 0.12,
    kalman_event_floor: float = _KALMAN_EVENT_FLOOR,
    weak_event_to_directional: bool = False,
    weak_event_min_move_atr: float = 0.35,
    sl_to_opposite: bool = False,
    include_weak_directional_in_train: bool = False,
    soft_label_mode: str = 'analytical',
    soft_label_n_scenarios: int = 200,
    soft_label_horizon_std: float = 0.30,
    soft_label_tp_std: float = 0.20,
    soft_label_sl_std: float = 0.20,
    soft_label_random_seed: int = 42,
    artifact_tag: str | None = None,
    save_preflight: bool = True,
    preflight_parquet: str | None = None,
    reuse_lob_tensors_dir: str | None = None,
    event_score_tier_labels: bool | None = None,
    fwd_ret_clean_bars: int | None = None,
    sanitize_max_bar_return_pct: float | None = None,
    sanitize_input_ticks: bool = True,
    enrich_v19_2: bool = False,
    enrich_v19_2_skip_missing: bool = True,
    n_workers: int | None = None,
    add_seasonal_features_flag: bool = True,
    add_cycle_features_flag: bool = True,
    timeout_mfe_mae_ratio: float = 2.0,
    timeout_mfe_min_move_atr: float = 1.0,
    apply_event_direction_veto: bool = False,
    add_multitask_diagnostics: bool = True,
) -> str:
    """
    Pipeline كاملة: MBO → Day Trading Dataset
    المخرج جاهز مباشرة لـ train_v19.py

    enrich_v19_2 (Phase 3 من تقرير الدمج): لو True، يطبّق
    modules.feature_enrichment.enrich_features على الـ output النهائي
    لإضافة ~21 feature (sim_*, bp_*, zone_full, ghz_*). الـ default = False
    للحفاظ على backward compat.

    مع رن كامل (بدون preflight_parquet): افتراضيًا يُحفظ checkpoint ما قبل Event Gate
    (save_preflight=True) لتجارب عتبات لاحقة عبر --preflight-parquet.
    sanitize_max_bar_return_pct: إن وُجد، تُسقط الشمعات ذات |(close−open)/open| أكبر منه قبل Event Gate.
    sanitize_input_ticks: إن True، تنظيف MBO/MBP بعد التحميل (فرز، تكرار، BBO متقاطع على MBP).
    """
    os.makedirs(output_dir, exist_ok=True)
    _set_session_profile(session_profile)

    tag = _sanitize_artifact_tag(artifact_tag)
    suffix = f"_{tag}" if tag else ""
    features_fn = f"day_trading_features{suffix}.parquet"
    manifest_fn = f"day_trading_manifest{suffix}.json"
    split_fn = f"refinery_split{suffix}.json"
    lob_fn = f"lob_tensors{suffix}.npy"
    lob_ts_fn = f"lob_tensor_timestamps{suffix}.npy"
    artifact_filenames = {
        "features_parquet": features_fn,
        "day_trading_manifest": manifest_fn,
        "refinery_split": split_fn,
        "lob_tensors": lob_fn if build_lob_tensors else None,
        "lob_tensor_timestamps": lob_ts_fn if build_lob_tensors else None,
    }

    print("\n" + "="*65)
    print("📊 Day Trading Refinery — QuantSystem V19")
    print(f"   Timeframe: {freq} | Horizon: {horizon_bars} bars")
    print(f"   Session profile: {session_profile} | windows={SESSIONS}")
    if tag:
        print(f"   🏷️  artifact_tag={tag!r} → outputs: {features_fn}, …")
    print(
        "   EventGate tuning: "
        f"threshold_scale={float(event_threshold_scale):.2f}, "
        f"threshold_shift={float(event_threshold_shift):+.2f}, "
        f"min_event_rate={float(min_event_rate):.1%}, "
        f"kalman_floor={float(kalman_event_floor):.2f}"
    )
    _tier_lbl = (
        bool(DAYTRADE_EVENT_SCORE_TIER_LABELS) if event_score_tier_labels is None else bool(event_score_tier_labels)
    )
    print(
        "   Event-score tier labels: "
        f"{'ON' if _tier_lbl else 'OFF'} "
        f"(strong≥{float(EVENT_LABEL_SCORE_STRONG_MIN)} mid≥{float(EVENT_LABEL_SCORE_MID_MIN)})"
    )
    print(
        "   Label policy: "
        f"weak_to_dir={bool(weak_event_to_directional)} "
        f"(min_move_atr={float(weak_event_min_move_atr):.2f}) | "
        f"sl_to_opposite={bool(sl_to_opposite)} | "
        f"include_weak_train={bool(include_weak_directional_in_train)}"
    )
    print(
        "   Soft labels: "
        f"mode={str(soft_label_mode).lower()} | "
        f"n_scenarios={int(max(1, soft_label_n_scenarios))} | "
        f"h_std={float(max(0.0, soft_label_horizon_std)):.2f} | "
        f"tp_std={float(max(0.0, soft_label_tp_std)):.2f} | "
        f"sl_std={float(max(0.0, soft_label_sl_std)):.2f} | "
        f"seed={int(soft_label_random_seed)}"
    )
    print("="*65)

    df_mbo: pd.DataFrame | None = None
    df_mbp: pd.DataFrame | None = None
    mbo_path_contract = os.path.abspath(str(mbo_dir)) if mbo_dir else ""

    if preflight_parquet:
        print("\n📂 Preflight — تحميل شموع ما قبل Event Gate (تخطي MBO/تجميع/ميزات)...")
        df_bars = _load_preflight_checkpoint(
            preflight_parquet, freq=freq, session_profile=session_profile
        )
        print(
            f"  ✅ {len(df_bars):,} bars | {len(df_bars.columns)} أعمدة | "
            f"{df_bars['ts_event'].min()} → {df_bars['ts_event'].max()}"
        )
        df_bars = _add_missing_six_features(df_bars)
        df_bars = _sanitize_bars_extreme_close_returns(df_bars, sanitize_max_bar_return_pct)
        if not mbo_path_contract:
            mbo_path_contract = os.path.abspath(str(preflight_parquet))
    else:
        # ── 1. تحميل MBO ──────────────────────────────────────────────
        print("\n📥 تحميل MBO data...")
        if not mbo_dir:
            raise ValueError("❌ mbo_dir مطلوب بدون --preflight-parquet")
        df_mbo = load_mbo_ticks_enriched(mbo_dir, sanitize_input_ticks=sanitize_input_ticks, n_workers=n_workers)
        mbo_path_contract = os.path.abspath(str(mbo_dir))
        print(f"  ✅ {len(df_mbo):,} تيك | {df_mbo['ts_event'].min()} → {df_mbo['ts_event'].max()}")

        # ── 1b. تحميل MBP (اختياري) ─────────────────────────────────────────────
        if mbp_path:
            print("\n📥 تحميل MBP10 data (optional)...")
            mbp_p = str(mbp_path)
            df_mbp = _read_mbp_table(mbp_p, levels=10)
            if df_mbp is not None and len(df_mbp):
                if sanitize_input_ticks:
                    df_mbp = _sanitize_mbp_ticks_for_daytrade(df_mbp)
                else:
                    df_mbp['ts_event'] = pd.to_datetime(df_mbp['ts_event'])
                    df_mbp = df_mbp.sort_values('ts_event').reset_index(drop=True)
                print(f"  ✅ {len(df_mbp):,} mbp rows | {df_mbp['ts_event'].min()} → {df_mbp['ts_event'].max()}")

        # ── 2. Aggregate → Bars ───────────────────────────────────────
        print(f"\n⏱️  تجميع في {freq} bars...")
        df_bars = aggregate_mbo_to_bars(df_mbo, freq=freq)
        if df_mbp is not None and len(df_mbp):
            df_bars = attach_mbp_bar_coverage(df_bars, df_mbp, freq)
        else:
            df_bars['mbp_bar_coverage'] = np.float32(0.0)
        print(f"  ✅ {len(df_bars):,} bar")

        # ── 3. Day Trading Features ───────────────────────────────────
        print("\n🔧 بناء Day Trading features...")
        df_bars = add_day_trading_features(df_bars, freq=freq)
        df_bars = enrich_bars_with_intrabar(df_bars, df_mbo, freq=freq, n_slices=12)
        df_bars = _add_missing_six_features(df_bars)
        if df_mbp is not None and len(df_mbp):
            df_bars = enrich_bars_with_intrabar_mbp(df_bars, df_mbp, freq=freq, n_slices=12, levels=10)
            spread_bar = pd.to_numeric(df_bars.get('spread_bar', 0.0), errors='coerce').fillna(0.0)
            mbp_spread = pd.to_numeric(df_bars.get('mbp_spread_mean', np.nan), errors='coerce')
            if float(spread_bar.abs().sum()) <= 1e-12 and mbp_spread.notna().any():
                df_bars['spread_bar'] = mbp_spread.fillna(0.0).astype(np.float32)
        print("\n📈 Kalman Trend Filter...")
        df_bars = add_kalman_trend(df_bars)
        print("\n🧭 Regime assignment...")
        df_bars = assign_regime_label(df_bars)
        print(f"  ✅ {len(df_bars.columns)} feature")

        # ── Seasonal Map: 21 ميزة موسمية سببية من ts_event ────────────
        if add_seasonal_features_flag:
            print("\n🗓️  Seasonal map features...")
            try:
                from modules.seasonal_map import add_seasonal_features as _add_seasonal
                cols_before = len(df_bars.columns)
                df_bars = _add_seasonal(df_bars, ts_col='ts_event')
                print(f"  ✅ +{len(df_bars.columns) - cols_before} seasonal features")
            except Exception as e:
                print(f"  ⚠️  seasonal_map تجاوزت: {e}")

        # ── Price Cycle (PR #18): 12 ميزة بنيوية (Wyckoff + swing + fractal) ─
        if add_cycle_features_flag:
            print("\n🌊 Price cycle structural features...")
            try:
                df_bars = _attach_price_cycle_features(df_bars)
            except Exception as e:
                print(f"  ⚠️  price_cycle تجاوزت: {e}")

        df_bars = _sanitize_bars_extreme_close_returns(df_bars, sanitize_max_bar_return_pct)

        if save_preflight:
            pp = _write_preflight_checkpoint(
                df_bars,
                output_dir=output_dir,
                suffix=suffix,
                freq=freq,
                session_profile=session_profile,
            )
            print(f"  💾 حفظ preflight للتجارب السريعة على العتبات: {pp}")

    # ── 4. Event Gate + Labels ────────────────────────────────────
    if 'tick_count' in df_bars.columns:
        df_bars['mbo_bar_coverage'] = _mbo_bar_coverage_from_tick_series(df_bars['tick_count'])
    elif 'num_trades' in df_bars.columns:
        df_bars['mbo_bar_coverage'] = _mbo_bar_coverage_from_tick_series(df_bars['num_trades'])
    elif 'mbo_bar_coverage' not in df_bars.columns:
        df_bars['mbo_bar_coverage'] = np.float32(0.0)
    print(f"\n🔍 كشف Microstructure Events (Event Gate — المشكلة F+I)...")
    df_bars = detect_microstructure_events(
        df_bars,
        threshold_scale=event_threshold_scale,
        threshold_shift=event_threshold_shift,
        min_event_rate=min_event_rate,
    )
    print("\n🧭 Directional voting (CVD + OBI + Kalman)...")
    df_bars = add_event_direction(df_bars)

    print(f"\n🏷️  بناء Labels — First Barrier Hit — regime + event_score tiers (إن فُعّلت)...")
    print(f"   TP/SL per regime: {REGIME_TP_SL}")
    print(f"   Max bars per regime: {REGIME_MAX_BARS}")
    print(f"   Sprint 19 MFE/MAE: ratio={timeout_mfe_mae_ratio}, min_move_atr={timeout_mfe_min_move_atr}")
    print(f"   event_direction veto: {'ENABLED (opt-in)' if apply_event_direction_veto else 'disabled (default — raw symmetric scan)'}")
    df_labeled = label_by_outcome(
        df_bars,
        default_tp_mult=tp_atr_mult,
        default_sl_mult=sl_atr_mult,
        default_max_bars=horizon_bars,
        kalman_event_floor=kalman_event_floor,
        weak_event_to_directional=weak_event_to_directional,
        weak_event_min_move_atr=weak_event_min_move_atr,
        sl_to_opposite=sl_to_opposite,
        include_weak_directional_in_train=include_weak_directional_in_train,
        use_event_score_tier_labels=event_score_tier_labels,
        timeout_mfe_mae_ratio=timeout_mfe_mae_ratio,
        timeout_mfe_min_move_atr=timeout_mfe_min_move_atr,
        apply_event_direction_veto=apply_event_direction_veto,
    )

    # توافق backward: أضف label_end_ts إذا لم توجد
    if 'label_end_ts' not in df_labeled.columns:
        df_labeled['label_end_ts'] = df_labeled['ts_event']
    if 'label_horizon_steps' not in df_labeled.columns:
        df_labeled['label_horizon_steps'] = horizon_bars
    if 'effective_horizon' not in df_labeled.columns:
        df_labeled['effective_horizon'] = df_labeled.get('trade_duration', horizon_bars)

    label_dist   = df_labeled['bias_label'].value_counts().to_dict()
    event_strict = float(df_labeled['event_flag'].mean())
    train_pool   = float(df_labeled['train_event_flag'].mean())
    nc0 = int((df_labeled['bias_label'] == 0).sum())
    nc1 = int((df_labeled['bias_label'] == 1).sum())
    imb = float(max(nc0, nc1)) / float(max(min(nc0, nc1), 1))
    print(
        f"  ✅ Labels: {label_dist} | event_flag={event_strict:.1%} | "
        f"train_pool={train_pool:.1%} | LONG/SHORT ratio≈{imb:.2f}:1"
    )

    # ── 4.5 Multi-task Diagnostics (Phase 1 من Multi-task Plan) ────────
    # يضيف 10 أعمدة جديدة بدون كسر الـ bias_label القديم:
    #   mfe/mae/mfe_atr/mae_atr/time_to_first_touch/stop_first_flag
    #   net_expectancy_proxy/neutral_type/tradability_label/market_state_label
    if add_multitask_diagnostics:
        df_labeled = compute_multitask_label_diagnostics(
            df_labeled,
            horizon_bars=int(horizon_bars * 4),  # نافذة أوسع للتشخيص
        )

    # ── 4.6 Strict Execution Labels (B1 — dual-target head 2) ──────────
    # additive: exec_label/exec_path/exec_valid عبر label_engine_v2 strict
    # (rescue OFF). لا يلمس bias_label؛ يُغذّي الـ supervised execution head
    # عبر قناع exec_valid (session-break aware).
    df_labeled = compute_strict_execution_labels(
        df_labeled,
        horizon_bars=int(horizon_bars),
        barrier_atr_mult=float(tp_atr_mult),
    )

    # ── 4.7 SSL Directional Target (B2 — dual-target head 1) ───────────
    # additive: next_price_delta + next_price_delta_valid — continuous
    # H-bar directional target for SSL backbone، gate-free على كل البارات.
    df_labeled = compute_ssl_directional_target(
        df_labeled,
        horizon_bars=int(horizon_bars),
    )

    # ── 5. Soft Labels ────────────────────────────────────────────
    print("\n🧪 إرفاق Soft Labels...")
    df_final = attach_soft_labels_dt(
        df_labeled,
        soft_label_mode=soft_label_mode,
        soft_label_n_scenarios=soft_label_n_scenarios,
        soft_label_horizon_std=soft_label_horizon_std,
        soft_label_tp_std=soft_label_tp_std,
        soft_label_sl_std=soft_label_sl_std,
        soft_label_random_seed=soft_label_random_seed,
    ).copy()
    df_final['mbp_roll_lob_coverage'] = np.float32(0.0)

    # ── 6. LOB tensors: نافذة DeepLOB = 50 شمعة تاريخية حقيقية (وليس شرائح داخل bar واحد)
    if build_lob_tensors:
        tensors_path = os.path.join(output_dir, lob_fn)
        ts_path = os.path.join(output_dir, lob_ts_fn)

        if reuse_lob_tensors_dir:
            print("\n👁️  إعادة استخدام LOB tensors (نسخ إلى مجلد المخرجات)...")
            src_dir = os.path.abspath(str(reuse_lob_tensors_dir))
            src_t = os.path.join(src_dir, lob_fn)
            src_ts = os.path.join(src_dir, lob_ts_fn)
            if not os.path.isfile(src_t) or not os.path.isfile(src_ts):
                raise FileNotFoundError(
                    f"❌ reuse LOB: غير موجود\n   {src_t}\n   {src_ts}\n"
                    f"   (استخدم نفس --artifact-tag أو نفس أسماء lob_tensors*_*.npy)"
                )
            tensors = np.load(src_t, allow_pickle=False)
            ts_raw = np.load(src_ts, allow_pickle=False)
            tensor_ts = ts_raw.astype('datetime64[ns]')
            if int(tensors.shape[0]) != int(len(df_final)):
                raise ValueError(
                    f"❌ reuse LOB: عدد الصفوف غير متطابق "
                    f"tensors.shape[0]={tensors.shape[0]} len(df_final)={len(df_final)}"
                )
            # تغطية النافذة غير متاحة من الملف القديم — أصفار لتجنب ميتاداتا مضللة
            roll_cov = np.zeros(len(df_final), dtype=np.float32)
            df_final['mbp_roll_lob_coverage'] = roll_cov
            np.save(tensors_path, tensors)
            np.save(ts_path, tensor_ts.astype('datetime64[ns]').astype(np.int64))
            print(f"  ✅ LOB reused: {tensors.shape} → {tensors_path}")
            print("  💡 mbp_roll_lob_coverage=0 عند الإعادة من ملف فقط (لا إعادة حساب)")
        else:
            print("\n👁️  بناء Rolling LOB tensors لـ DeepLOB (50-bar history × 20 levels × 3ch)...")
            if df_mbo is None:
                if not mbo_dir:
                    raise ValueError(
                        "❌ بناء LOB بعد preflight يتطلب --mbo لتحميل التيكات، "
                        "أو مرّر --reuse-lob-tensors-dir، أو --no-lob"
                    )
                print("  📥 تحميل MBO/MBP من أجل LOB فقط...")
                df_mbo = load_mbo_ticks_enriched(mbo_dir, sanitize_input_ticks=sanitize_input_ticks, n_workers=n_workers)
                if mbp_path:
                    df_mbp = _read_mbp_table(str(mbp_path), levels=10)
                    if df_mbp is not None and len(df_mbp):
                        if sanitize_input_ticks:
                            df_mbp = _sanitize_mbp_ticks_for_daytrade(df_mbp)
                        else:
                            df_mbp['ts_event'] = pd.to_datetime(df_mbp['ts_event'])
                            df_mbp = df_mbp.sort_values('ts_event').reset_index(drop=True)
            if df_mbp is not None and len(df_mbp):
                tensors, tensor_ts, roll_cov = build_rolling_lob_tensors_from_mbp(
                    df_mbo,
                    df_mbp,
                    df_final,
                    freq=freq,
                    lookback_bars=DEEPLOB_TIME_STEPS_DEFAULT,
                    levels=10,
                )
            else:
                tensors, tensor_ts, roll_cov = build_rolling_lob_tensors_mbo_only(
                    df_mbo,
                    df_final,
                    freq=freq,
                    lookback_bars=DEEPLOB_TIME_STEPS_DEFAULT,
                    n_levels=20,
                )
            df_final['mbp_roll_lob_coverage'] = roll_cov.astype(np.float32)

            np.save(tensors_path, tensors)
            np.save(ts_path, tensor_ts.astype('datetime64[ns]').astype(np.int64))
            print(f"  ✅ LOB Tensors: {tensors.shape} → {tensors_path}")
            print(
                "  💡 البعد الزمني = الشموع السابقة؛ مع MBP: لقطة peak-imbalance لكل شمعة. "
                f"متوسط التغطية في النافذة={float(np.mean(roll_cov)):.2f}"
            )

    # لا نُصدّر raw__* هنا: مسار اليوم لا يُطبّق RobustScaler على أعمدة الـ advisor قبل الحفظ،
    # فكان raw__ نسخاً مطابقة 100% لـ cvd و liquidity_density إلخ → إهدار عمود مكرر.
    # prepare_training_data يُنشئ raw__ قبل تحجيم MODEL_FEATURE_COLS (هناك فرق حقيقي).
    # train_v19 / predict_v19 يستخدمان العمود الأساسي إذا غاب raw__ (_raw_stat_frame).
    df_out = df_final.copy()

    # تأكد من وجود الأعمدة المطلوبة لـ train_v19.py
    required_cols = ['ts_event', 'label_end_ts', 'bias_label', 'signal_quality',
                     'forward_return', 'event_flag', 'train_event_flag',
                     'soft_label', 'label_confidence', 'soft_sample_weight',
                     'is_event', 'event_score', 'event_direction', 'path_outcome', 'trade_duration',
                     'neutral_reason', 'kalman_direction', 'regime_label', 'regime_cluster',
                     'dataset_slice', 'is_train_slice', 'is_holdout_slice', 'is_purged_slice']
    for col in required_cols:
        if col not in df_out.columns:
            df_out[col] = 0

    # السعر المرجعي للباك تست والمحرك
    if 'price' not in df_out.columns and 'close' in df_out.columns:
        df_out['price'] = pd.to_numeric(df_out['close'], errors='coerce')

    # عائد أمامي بأفق ثابت N (إغلاق→إغلاق) لتشخيص IC — منفصل عن forward_return المرتبط بحدود الليبل/max_bars
    if fwd_ret_clean_bars is not None and int(fwd_ret_clean_bars) < 0:
        n_fwd_clean = None
    elif fwd_ret_clean_bars is None:
        n_fwd_clean = int(horizon_bars)
    else:
        n_fwd_clean = max(int(fwd_ret_clean_bars), 0)

    if n_fwd_clean is not None and n_fwd_clean > 0:
        df_out = add_fwd_ret_clean_column(df_out, n_fwd_clean)
        print(
            f"   📎 fwd_ret_clean: N={n_fwd_clean} "
            f"(عائد إغلاق→إغلاق؛ استخدمه مع alpha_validation --forward-col fwd_ret_clean؛ "
            f"آخر {n_fwd_clean} صفًا = NaN)"
        )
    else:
        n_fwd_clean = None
        if fwd_ret_clean_bars is not None and int(fwd_ret_clean_bars) < 0:
            print("   📎 fwd_ret_clean: معطّل (--fwd-ret-clean-bars -1)")

    # split contract متوافق مع المصفاة الأساسية (train / holdout / purged)
    split_ctx = _build_daytrade_split_context(df_out, train_frac=0.80)
    train_mask = split_ctx['train_row_ok']
    holdout_mask = split_ctx['holdout_row_ok']
    purged_mask = split_ctx['purged_row_ok']
    df_out['is_train_slice'] = train_mask.astype(np.int8)
    df_out['is_holdout_slice'] = holdout_mask.astype(np.int8)
    df_out['is_purged_slice'] = purged_mask.astype(np.int8)
    df_out['dataset_slice'] = np.where(
        train_mask,
        'train',
        np.where(holdout_mask, 'holdout', 'purged'),
    )
    split_time = split_ctx['split_time']
    split_meta = {
        'split_idx': int(split_ctx['split_idx']),
        'split_time': None if pd.isna(split_time) else str(split_time),
        'train_rows': int(np.sum(train_mask)),
        'holdout_rows': int(np.sum(holdout_mask)),
        'purged_rows': int(np.sum(purged_mask)),
        'total_rows': int(len(df_out)),
    }
    split_path = os.path.join(output_dir, split_fn)
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(
        "   🧩 Split contract: "
        f"train={split_meta['train_rows']:,}, "
        f"holdout={split_meta['holdout_rows']:,}, "
        f"purged={split_meta['purged_rows']:,} | "
        f"split_time={split_meta['split_time']}"
    )

    df_out = finalize_daytrade_parquet_export(df_out)

    # Phase 3 (تقرير الدمج): V19.2 feature enrichment.
    # يضيف sim_* (18) + bp_* (9) + zone_full + level_distance + ghz_*.
    if enrich_v19_2:
        try:
            from modules.feature_enrichment import enrich_features, EnrichmentConfig
            print("\n🔮 Phase 3 enrichment (V19.2 features)...")
            n_before = len(df_out.columns)
            cfg = EnrichmentConfig(
                add_simulators=True,
                add_wall_depth=False,  # يحتاج MBP منفصل
                add_iceberg=False,     # يحتاج MBO منفصل
                add_session_mapping=True,
                add_bell_pairs=True,
                skip_missing_cols=enrich_v19_2_skip_missing,
                verbose=False,
            )
            df_out = enrich_features(df_out, config=cfg)
            n_added = len(df_out.columns) - n_before
            print(f"   ✅ {n_added} عمود مُضاف ({n_before} → {len(df_out.columns)})")
        except Exception as exc:
            print(f"   ⚠️ enrichment تخطّي ({type(exc).__name__}: {exc})")

    out_path = os.path.join(output_dir, features_fn)

    # Phase 0 fix: rename columns to match downstream consumer expectations.
    # The IC audit + event_gate audit + verify_data_health expect specific
    # canonical names. Without this, downstream tools report "source missing".
    _phase0_renames = {}
    if 'obi' in df_out.columns and 'obi_net' not in df_out.columns:
        _phase0_renames['obi'] = 'obi_net'
    if 'cvd' in df_out.columns and 'cvd_direction_pct' not in df_out.columns:
        # The existing `cvd` column is the cumulative line — not the
        # direction ratio expected by the event gate. We expose BOTH:
        # keep `cvd` as-is (cumulative), add `cvd_cumulative` as alias
        # so downstream code can pick the right one. The proper
        # `cvd_direction_pct` per-bar feature is engineered in Phase 2.
        df_out['cvd_cumulative'] = df_out['cvd']
        # Don't drop or rename `cvd` yet — Phase 2 builds proper variants.
    if _phase0_renames:
        df_out = df_out.rename(columns=_phase0_renames)
        print(f"   🔧 Phase 0 renames applied: {_phase0_renames}")

    df_out.to_parquet(out_path, index=False)
    print(f"\n💾 Dataset محفوظ: {out_path}")
    print(f"   Rows: {len(df_out):,} | Columns: {len(df_out.columns)}")

    # ── C2: dataset_meta.json sidecar — schema source of truth ─────────
    # Lets external consumers introspect the parquet's label-side schema
    # (which targets present, which masks they pair with, leakage classes)
    # without importing the refinery. See modules/dataset_schema.py.
    try:
        from modules.dataset_schema import write_dataset_meta
        meta_extra = {
            "freq": str(freq),
            "horizon_bars": int(horizon_bars),
            "tp_atr_mult": float(tp_atr_mult),
            "sl_atr_mult": float(sl_atr_mult),
            "session_profile": str(session_profile),
            "apply_event_direction_veto": bool(apply_event_direction_veto),
        }
        meta_path = write_dataset_meta(df_out, out_path, extra=meta_extra)
        print(f"   📋 dataset_meta.json: {meta_path}")
    except Exception as e:
        print(f"   ⚠️  dataset_meta sidecar skipped: {e!r}")

    mbp_bar_cov_stats = _coverage_stats(df_out.get('mbp_bar_coverage', 0.0), low_threshold=0.30)
    mbp_roll_cov_stats = _coverage_stats(df_out.get('mbp_roll_lob_coverage', 0.0), low_threshold=0.50)
    mbo_bar_cov_stats = _coverage_stats(df_out.get('mbo_bar_coverage', 0.0), low_threshold=0.50)
    print(
        "   📊 MBP coverage: "
        f"bar_mean={mbp_bar_cov_stats['mean']:.1%}, bar_low(<30%)={mbp_bar_cov_stats['low_ratio']:.1%} | "
        f"roll_mean={mbp_roll_cov_stats['mean']:.1%}, roll_low(<50%)={mbp_roll_cov_stats['low_ratio']:.1%}"
    )
    print(
        "   📊 MBO tape density (mbo_bar_coverage): "
        f"mean={mbo_bar_cov_stats['mean']:.1%}, low(<50%)={mbo_bar_cov_stats['low_ratio']:.1%}"
    )

    # فلتر التدريب — للتحقق (لا يُحذف، train_v19 يُفلتر بنفسه)
    n_train_events = int(df_out['train_event_flag'].sum())
    n_long  = int((df_out['bias_label'] == 0).sum())
    n_short = int((df_out['bias_label'] == 1).sum())
    print(f"\n   📊 Training pool:")
    print(f"      train_event_flag = True : {n_train_events:,} rows ({n_train_events/max(len(df_out),1):.1%})")
    print(f"      LONG  : {n_long:,} | SHORT : {n_short:,}")
    if 'regime_label' in df_out.columns:
        for reg in ['trending', 'ranging', 'volatile']:
            m = (df_out['regime_label'] == reg) & (df_out['train_event_flag'] == 1)
            print(f"      {reg:8s} events: {int(m.sum()):,}")

    # ── 7.5 Event-gate auto-diagnostic ─────────────────────────────
    # Surfaces gate starvation / component domination / warm-up bias
    # so high-NEUTRAL pipelines tell us WHY before we ask. Safe wrapper:
    # any failure prints a warning and continues — never breaks the run.
    _run_event_gate_audit_safe(out_path, output_dir)

    # ── 8. Manifest ────────────────────────────────────────────────
    regime_ev_counts: dict = {}
    if 'regime_label' in df_out.columns:
        for reg in ['trending', 'ranging', 'volatile']:
            m = (df_out['regime_label'] == reg) & (df_out['train_event_flag'] == 1)
            regime_ev_counts[reg] = int(m.sum())
    neutral_counts = {
        str(k): int(v) for k, v in pd.to_numeric(df_out.get('neutral_reason', 0), errors='coerce')
        .fillna(0).astype(int).value_counts().to_dict().items()
    }
    event_direction_counts = {
        str(k): int(v) for k, v in pd.to_numeric(df_out.get('event_direction', 0), errors='coerce')
        .fillna(0).astype(int).value_counts().to_dict().items()
    }
    contract = _build_daytrade_contract(
        df=df_out,
        mbo_path=mbo_path_contract or os.path.abspath(str(preflight_parquet or "")),
        mbp_path=mbp_path,
        output_dir=output_dir,
        freq=freq,
        horizon_bars=horizon_bars,
        session_profile=session_profile,
        split_meta=split_meta,
    )

    manifest = {
        'mode'                        : 'day_trading',
        'version'                     : 'v19-event-gate',
        'freq'                        : freq,
        'sanitize_max_bar_return_pct' : sanitize_max_bar_return_pct,
        'sanitize_input_ticks'        : bool(sanitize_input_ticks),
        'horizon_bars_default'        : horizon_bars,
        'fwd_ret_clean_bars'          : n_fwd_clean,
        'tp_atr_mult_default'         : tp_atr_mult,
        'sl_atr_mult_default'         : sl_atr_mult,
        'regime_tp_sl'                : {k: list(v) for k, v in REGIME_TP_SL.items()},
        'regime_max_bars'             : REGIME_MAX_BARS,
        'regime_event_threshold'      : REGIME_EVENT_THRESHOLD,
        'event_score_weights'         : {k: float(v) for k, v in EVENT_SCORE_WEIGHTS.items()},
        'event_lob_coverage_cut'      : float(EVENT_LOB_COVERAGE_CUT),
        'event_mbo_coverage_cut'      : float(EVENT_MBO_COVERAGE_CUT),
        'event_threshold_scale'       : float(event_threshold_scale),
        'event_threshold_shift'       : float(event_threshold_shift),
        'min_event_rate'              : float(min_event_rate),
        'kalman_event_floor'          : float(kalman_event_floor),
        'event_score_tier_labels'     : (
            bool(DAYTRADE_EVENT_SCORE_TIER_LABELS)
            if event_score_tier_labels is None
            else bool(event_score_tier_labels)
        ),
        'event_label_score_strong_min': float(EVENT_LABEL_SCORE_STRONG_MIN),
        'event_label_score_mid_min'   : float(EVENT_LABEL_SCORE_MID_MIN),
        'event_label_tier_strong_tpl' : list(EVENT_LABEL_TIER_STRONG),
        'event_label_tier_mid_tpl'    : list(EVENT_LABEL_TIER_MID),
        'event_label_tier_weak_tpl'   : list(EVENT_LABEL_TIER_WEAK),
        'weak_event_to_directional'   : bool(weak_event_to_directional),
        'weak_event_min_move_atr'     : float(weak_event_min_move_atr),
        'sl_to_opposite'              : bool(sl_to_opposite),
        'include_weak_directional_in_train': bool(include_weak_directional_in_train),
        'soft_label_mode'             : str(soft_label_mode).lower(),
        'soft_label_n_scenarios'      : int(max(1, soft_label_n_scenarios)),
        'soft_label_horizon_std'      : float(max(0.0, soft_label_horizon_std)),
        'soft_label_tp_std'           : float(max(0.0, soft_label_tp_std)),
        'soft_label_sl_std'           : float(max(0.0, soft_label_sl_std)),
        'soft_label_random_seed'      : int(soft_label_random_seed),
        'preflight_parquet'           : None if not preflight_parquet else os.path.abspath(str(preflight_parquet)),
        'reuse_lob_tensors_dir'       : None if not reuse_lob_tensors_dir else os.path.abspath(str(reuse_lob_tensors_dir)),
        'save_preflight_emitted'      : bool(save_preflight and not preflight_parquet),
        'rows'                        : len(df_out),
        'label_distribution'          : {str(k): int(v) for k, v in label_dist.items()},
        'event_rate_tp_wins'          : float(event_strict),
        'train_event_pool_rate'       : float(train_pool),
        'train_event_count'           : n_train_events,
        'bias_long_short_counts'      : {'LONG': nc0, 'SHORT': nc1},
        'long_short_imbalance'        : round(imb, 3),
        'regime_train_event_counts'   : regime_ev_counts,
        'neutral_reason_counts'       : neutral_counts,
        'event_direction_counts'      : event_direction_counts,
        'catboost_surface_n'          : len(CATBOOST_ADVISOR_FEATURES_DT),
        'catboost_advisor_features'   : CATBOOST_ADVISOR_FEATURES_DT,
        'original_features'           : list(ORIGINAL_FEATURES),
        'original_features_export_only': True,
        'day_trading_context_features': DAY_TRADING_FEATURES,
        'lob_tensors_built'           : bool(build_lob_tensors),
        'lob_roll_lookback_bars'      : DEEPLOB_TIME_STEPS_DEFAULT,
        'lob_tensor_layout'           : 'rolling_bars_time_x_20_levels_x_3ch_peak_mbp_when_available',
        'mbp_bar_coverage_stats'      : mbp_bar_cov_stats,
        'mbp_roll_lob_coverage_stats' : mbp_roll_cov_stats,
        'mbo_bar_coverage_stats'      : mbo_bar_cov_stats,
        'ts_min'                      : str(df_out['ts_event'].min()),
        'ts_max'                      : str(df_out['ts_event'].max()),
        'session_profile'             : str(session_profile),
        'session_windows_utc'         : {k: list(v) for k, v in SESSIONS.items()},
        'dataset_split'               : split_meta,
        'artifact_contract'           : {
            'dataset_id': contract.get('dataset_id'),
            'schema_version': contract.get('schema_version'),
            'label_mode': contract.get('label_mode'),
        },
        'label_logic'                 : (
            'Event Gate (hawkes+absorption+kyle+cvd+LOB roll coverage+MBO tape density) → '
            'Kalman Trend Filter (weak counter-trend veto) → '
            'First Barrier Hit (Regime TP/SL unless event_score tiers on is_event: '
            'strong≥event_label_score_strong_min / mid≥event_label_score_mid_min else weak) '
            '+ Session Timeout → SL outcomes → NEUTRAL'
        ),
    }
    with open(os.path.join(output_dir, manifest_fn), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    contract.update({
        'split_meta': split_meta,
        'artifact_filenames': artifact_filenames,
        'final_feature_shards': [{
            'name': os.path.basename(out_path),
            'path': os.path.abspath(out_path),
            'rows': int(len(df_out)),
            'ts_min': str(df_out['ts_event'].min()),
            'ts_max': str(df_out['ts_event'].max()),
        }],
    })
    artifact_manifest_path = write_manifest(
        output_dir,
        kind='day_trading_artifact_v19',
        config={
            'freq': freq,
            'horizon': int(horizon_bars),
            'tp_mult': float(tp_atr_mult),
            'sl_mult': float(sl_atr_mult),
            'session_profile': str(session_profile),
            'min_event_rate': float(min_event_rate),
            'soft_label_mode': str(soft_label_mode).lower(),
            'sanitize_input_ticks': bool(sanitize_input_ticks),
        },
        inputs={
            'mbo': mbo_path_contract or "",
            'mbp': None if not mbp_path else os.path.abspath(str(mbp_path)),
            'preflight_parquet': None if not preflight_parquet else os.path.abspath(str(preflight_parquet)),
        },
        metrics={
            'rows': int(len(df_out)),
            'train_rows': int(split_meta['train_rows']),
            'holdout_rows': int(split_meta['holdout_rows']),
            'purged_rows': int(split_meta['purged_rows']),
        },
        extra=contract,
        filename='artifact_manifest.json',
    )
    print(f"  ✅ Artifact manifest: {artifact_manifest_path}")
    print(f"  ✅ Split metadata: {split_path}")

    print("\n" + "="*65)
    print("✅ Day Trading Refinery اكتمل")
    print(f"   الخطوة التالية:")
    print(f"   py -3.13 train_v19.py --data {out_path} --output outputs_dt")
    print("="*65)

    return out_path


def _cli_validate_freq(value: str) -> str:
    """يقبل أي offset زمني يفهمه pandas (مثل 1min، 5min، 90s، 4h)."""
    s = str(value).strip()
    if not s:
        raise argparse.ArgumentTypeError('freq فارغ')
    try:
        td = pd.to_timedelta(s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"freq غير صالح {value!r}; أمثلة: 1min 5min 15min 30min 1h 4h"
        ) from exc
    if bool(pd.isna(td)):
        raise argparse.ArgumentTypeError(f"freq غير صالح {value!r}")
    sec = float(td.total_seconds())
    if sec <= 0:
        raise argparse.ArgumentTypeError('freq يجب أن يكون مدة موجبة')
    return s


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description='Day Trading Refinery — QuantSystem V19')
    p.add_argument(
        '--mbo',
        default=None,
        help='مسار مجلد/ملف MBO parquet (مطلوب إلا مع --preflight-parquet وبدون إعادة بناء LOB من التيكات)',
    )
    p.add_argument('--mbp',     default=None, help='اختياري: مسار ملف/مجلد MBP10 (csv/parquet) لاستخراج ميزات book قوية')
    p.add_argument('--output',  default='pipeline_day_trading/features', help='مسار الـ output')
    p.add_argument(
        '--inspect-parquet',
        default=None,
        metavar='PATH',
        help=(
            'طباعة ملخص بنية parquet (صفوف، أعمدة، تغطية ORIGINAL_FEATURES) ثم الخروج؛ '
            'لا يحمّل كامل الجدول إن وُجد pyarrow. لا يُشغّل المصفاة.'
        ),
    )
    p.add_argument(
        '--freq',
        type=_cli_validate_freq,
        default=DAY_TRADE_DEFAULT_BAR_FREQ,
        help=f'شمع أساس pandas (مثل 1min، 5min، 15min، 30min، 1h، 4h…)، الافتراضي={DAY_TRADE_DEFAULT_BAR_FREQ}',
    )
    p.add_argument(
        '--horizon',
        type=int,
        default=6,
        help=(
            'عدد شموع أفق الليبل/الحدود؛ المدة التقويمية = horizon × مدة الشمعة. '
            'مع --freq 1min القيمة 60 تعني تقريباً ساعة أماماً، والقيمة 6 تعني 6 دقائق.'
        ),
    )
    p.add_argument('--tp_mult', type=float, default=1.5, help='TP = tp_mult * ATR')
    p.add_argument('--sl_mult', type=float, default=1.0, help='SL = sl_mult × ATR')
    p.add_argument('--no_lob',  action='store_true', help='تخطي بناء LOB tensors')
    p.add_argument(
        '--session_profile',
        '--session-profile',
        choices=sorted(SESSION_PROFILES.keys()),
        default='daytrade_default',
        dest='session_profile',
        help='تعريف نوافذ الجلسات UTC. استخدم stage1_like لمطابقة المصفاة الأساسية.',
    )
    p.add_argument(
        '--event_threshold_scale',
        type=float,
        default=1.0,
        help='scale لعَتبات Event Gate (أقل من 1.0 = إشارات أكثر، default=1.0)',
    )
    p.add_argument(
        '--event_threshold_shift',
        type=float,
        default=0.0,
        help='إزاحة لعَتبات Event Gate بعد الـ scale (قيمة سالبة = إشارات أكثر، default=0.0)',
    )
    p.add_argument(
        '--min_event_rate',
        type=float,
        default=0.12,
        help=(
            'floor مستهدف لنسبة event_flag (يُقصّ ضمن [0, 60%%])؛ '
            'أعلى من الـ default يوسّع التدريب عند تضيّق البوابة، default=0.12'
        ),
    )
    p.add_argument(
        '--kalman_event_floor',
        type=float,
        default=float(_KALMAN_EVENT_FLOOR),
        help='عتبة event_score قبل Kalman counter-trend veto (أقل = veto أقل، default=0.70)',
    )
    p.add_argument(
        '--no-event-score-tier-labels',
        action='store_true',
        help=(
            'عطّل شرائح TP/SL/horizon حسب event_score على صفوف الحدث؛ '
            'يعود السلوك لـ REGIME_TP_SL و REGIME_MAX_BARS فقط (افتراضي: من regime_config، عادة مفعّل).'
        ),
    )
    p.add_argument(
        '--weak_event_to_directional',
        action='store_true',
        help='حوّل weak-event rows إلى LONG/SHORT إذا حركة الأفق تخطت حد ATR (تقليل قوي للـ NEUTRAL).',
    )
    p.add_argument(
        '--weak_event_min_move_atr',
        type=float,
        default=0.35,
        help='الحد الأدنى لحركة weak-event (بوحدة ATR) قبل تحويلها لاتجاهي، default=0.35',
    )
    p.add_argument(
        '--sl_to_opposite',
        action='store_true',
        help='حوّل long_sl->SHORT و short_sl->LONG بدل NEUTRAL (وضع aggressive).',
    )
    p.add_argument(
        '--include_weak_directional_in_train',
        action='store_true',
        help='أدخل الاتجاهات الناتجة من weak-event ضمن train_event_flag.',
    )
    p.add_argument(
        '--strict_train_pool',
        action='store_true',
        help='يضيق train_event_flag: جلسات نشطة فقط + ATR >= 0.5x الوسيط (اتجاهي = فوز TP فقط)',
    )
    p.add_argument(
        '--soft_label_mode',
        choices=['analytical', 'monte_carlo'],
        default='analytical',
        help='طريقة soft labels داخل DayTrade (default=analytical)',
    )
    p.add_argument(
        '--soft_label_n_scenarios',
        type=int,
        default=200,
        help='عدد سيناريوهات Monte Carlo عند --soft_label_mode monte_carlo',
    )
    p.add_argument(
        '--soft_label_horizon_std',
        type=float,
        default=0.30,
        help='std لاضطراب horizon في Monte Carlo soft labels',
    )
    p.add_argument(
        '--soft_label_tp_std',
        type=float,
        default=0.20,
        help='std لاضطراب TP في Monte Carlo soft labels',
    )
    p.add_argument(
        '--soft_label_sl_std',
        type=float,
        default=0.20,
        help='std لاضطراب SL في Monte Carlo soft labels',
    )
    p.add_argument(
        '--soft_label_random_seed',
        type=int,
        default=42,
        help='random seed لسيناريوهات Monte Carlo soft labels',
    )
    p.add_argument(
        '--artifact-tag',
        default=None,
        help=(
            "Safe suffix for daytrade outputs (e.g. v19_tensorfix): "
            "day_trading_features_<tag>.parquet, lob_tensors_<tag>.npy; "
            "recorded in artifact_manifest for train_v19."
        ),
    )
    p.add_argument(
        '--no-save-preflight',
        action='store_true',
        help=(
            'لا تحفظ day_trading_preflight_<tag>.parquet (الافتراضي: يُحفظ بعد بناء الميزات وقبل Event Gate '
            'للتجارب السريعة على العتبات عبر --preflight-parquet).'
        ),
    )
    p.add_argument(
        '--slim-output',
        action='store_true',
        help=(
            'مخرجات أقل في مجلد النتيجة: لا حفظ preflight ولا ملفات LOB (*.npy)؛ يفرض سلوكاً مطابقاً لـ --no-lob '
            '(لا يُجمع مع --reuse-lob-tensors-dir). يبقى parquet الرئيسي + manifest + split.'
        ),
    )
    p.add_argument(
        '--sanitize-max-bar-return',
        dest='sanitize_max_bar_return',
        type=float,
        default=None,
        metavar='PCT',
        help=(
            'إسقاط الشمعات حيث |(close−open)/open| أكبر من هذا الحد (كسر مطلق، مثل 0.05 ≈ 5%%). '
            'افتراضي: بدون إسقاط.'
        ),
    )
    p.add_argument(
        '--preflight-parquet',
        default=None,
        help=(
            'تحميل شموع ما قبل البوابة من parquet (تخطي تحميل MBO والتجميع وبناء الميزات). '
            'لازم نفس --freq و --session_profile كما عند الحفظ. مع LOB: مرّر --mbo أو --reuse-lob-tensors-dir.'
        ),
    )
    p.add_argument(
        '--reuse-lob-tensors-dir',
        default=None,
        help=(
            'مجلد يحتوي lob_tensors_<tag>.npy و lob_tensor_timestamps_<tag>.npy (نفس --artifact-tag) '
            'لنسخها بدون إعادة حساب (سريع مع --preflight-parquet).'
        ),
    )
    p.add_argument(
        '--fwd-ret-clean-bars',
        type=int,
        default=None,
        help=(
            'يضيف عمود fwd_ret_clean = close.shift(-N)/close-1 لتشخيص IC منفصل عن مسار الليبل. '
            'بدون تمرير: N = نفس --horizon. مرّر -1 لتعطيل الإضافة.'
        ),
    )
    p.add_argument(
        '--no-sanitize-input-ticks',
        action='store_true',
        help=(
            'تعطيل تنظيف تيكات الإدخال (فرز ثابت مع إزالة التكرار الكامل/المفتاح على MBO، '
            'وإسقاط BBO المتقاطع على MBP). الافتراضي: التنظيف مفعّل.'
        ),
    )
    p.add_argument(
        '--enrich-v19-2',
        action='store_true',
        help=(
            'Phase 3 من تقرير الدمج: يضيف V19.2 features '
            '(sim_*, bp_*, zone_full, ghz_*) على الـ output النهائي. '
            'الافتراضي: معطّل (backward compat).'
        ),
    )
    p.add_argument(
        '--timeout-mfe-mae-ratio',
        type=float,
        default=2.0,
        help=(
            'Sprint 19: MFE/MAE ratio لاتخاذ قرار اتجاهي في حالات timeout/SL. '
            'الافتراضي=2.0 (الفائز يجب أن يكون ضعف الخاسر). '
            'لبيانات low volatility، جرّب 1.3-1.5 لرفع directional yield.'
        ),
    )
    p.add_argument(
        '--timeout-mfe-min-move-atr',
        type=float,
        default=1.0,
        help=(
            'Sprint 19: الحد الأدنى لحركة MFE/MAE بوحدة ATR. الافتراضي=1.0. '
            'لبيانات low volatility (مثل 6B في فترات هادئة)، جرّب 0.3-0.5 '
            'لرفع directional labels من ~2%% إلى ~30-50%%.'
        ),
    )
    p.add_argument(
        '--apply-event-direction-veto',
        action='store_true',
        help=(
            'A2: يُفعّل (opt-in) فيتو event_direction في الـ labeling. مُعطَّل '
            'افتراضياً. تفعيله يسمح لـ event_direction voting (CVD+OBI+Kalman) '
            'بمنع اتجاه في الـ scan والـ rescue. تحذير: الـ voting يخطئ ~70%% على '
            'real data، فتفعيله يُسقط ~60%% من الـ directional signal ويعيد فرض '
            'انحياز اتجاهي فوق الـ symmetric scan (A1). للتجارب فقط؛ event_direction '
            'يبقى عموداً (feature) يتعلّمه النموذج في الوضع الافتراضي.'
        ),
    )
    p.add_argument(
        '--no-multitask-diagnostics',
        action='store_true',
        help=(
            'يعطّل Multi-task diagnostic columns (Phase 1 من Multi-task Plan). '
            'الافتراضي: مُفعَّل — يضيف 10 أعمدة (mfe/mae/tradability_label/'
            'neutral_type/market_state_label/net_expectancy_proxy/...) بدون كسر '
            'bias_label القديم. تُستخدم في Phase 2 لتدريب multi-head MetaLearner.'
        ),
    )
    p.add_argument(
        '--no-seasonal',
        action='store_true',
        help=(
            'يعطّل إضافة الـ 21 ميزة من modules/seasonal_map.py '
            '(session_phase، dow، month_end، quarter_end، …). افتراضي: مُفعَّل.'
        ),
    )
    p.add_argument(
        '--no-cycle-features',
        action='store_true',
        help=(
            'يعطّل إضافة الـ 12 ميزة بنيوية من modules/price_cycle (PR #18): '
            'Wyckoff phases + swing structure + Hurst + fractal alignment. '
            'افتراضي: مُفعَّل.'
        ),
    )
    p.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help=(
            'عدد worker processes لـ enrich_mbo loop. None=auto (cpu_count-1)، '
            '1=sequential (مطابق بايت-لبايت للسلوك القديم). يمكن override بـ env DD_N_WORKERS. '
            'الانقسام بالـ day؛ المحركات stateful بتاخد reset في حدود اليوم في النسخة المتوازية '
            '(~92-97%% من ticks متطابقة مع sequential، الفرق في warmup transient أول كل يوم).'
        ),
    )
    args = p.parse_args()

    if args.inspect_parquet:
        inspect_daytrade_parquet_schema(args.inspect_parquet)
        sys.exit(0)

    slim = bool(args.slim_output)
    save_preflight_run = (not args.no_save_preflight) and (not slim)
    build_lob_run = (not args.no_lob) and (not slim)

    if slim and args.reuse_lob_tensors_dir:
        p.error('--slim-output لا يُجمع مع --reuse-lob-tensors-dir')

    if not args.preflight_parquet and args.mbo is None:
        p.error('--mbo مطلوب ما لم يُمرَّر --preflight-parquet')
    if args.reuse_lob_tensors_dir and args.no_lob:
        p.error('لا يمكن الجمع بين --reuse-lob-tensors-dir و --no-lob')
    if args.preflight_parquet and (not args.no_lob) and (not args.reuse_lob_tensors_dir) and args.mbo is None:
        p.error(
            'مع --preflight-parquet: مرّر --mbo لبناء LOB من التيكات، أو '
            '--reuse-lob-tensors-dir لنسخ tensors جاهزة، أو --no-lob'
        )

    run_day_trading_refinery(
        mbo_dir=args.mbo,
        mbp_path=args.mbp,
        output_dir=args.output,
        freq=args.freq,
        horizon_bars=args.horizon,
        tp_atr_mult=args.tp_mult,
        sl_atr_mult=args.sl_mult,
        build_lob_tensors=build_lob_run,
        session_profile=args.session_profile,
        strict_train_pool=args.strict_train_pool,
        event_threshold_scale=args.event_threshold_scale,
        event_threshold_shift=args.event_threshold_shift,
        min_event_rate=args.min_event_rate,
        kalman_event_floor=args.kalman_event_floor,
        weak_event_to_directional=args.weak_event_to_directional,
        weak_event_min_move_atr=args.weak_event_min_move_atr,
        sl_to_opposite=args.sl_to_opposite,
        include_weak_directional_in_train=args.include_weak_directional_in_train,
        soft_label_mode=args.soft_label_mode,
        soft_label_n_scenarios=args.soft_label_n_scenarios,
        soft_label_horizon_std=args.soft_label_horizon_std,
        soft_label_tp_std=args.soft_label_tp_std,
        soft_label_sl_std=args.soft_label_sl_std,
        soft_label_random_seed=args.soft_label_random_seed,
        artifact_tag=args.artifact_tag,
        save_preflight=save_preflight_run,
        preflight_parquet=args.preflight_parquet,
        reuse_lob_tensors_dir=args.reuse_lob_tensors_dir,
        event_score_tier_labels=(False if args.no_event_score_tier_labels else None),
        fwd_ret_clean_bars=args.fwd_ret_clean_bars,
        sanitize_max_bar_return_pct=args.sanitize_max_bar_return,
        sanitize_input_ticks=(not args.no_sanitize_input_ticks),
        enrich_v19_2=args.enrich_v19_2,
        n_workers=args.n_workers,
        add_seasonal_features_flag=(not args.no_seasonal),
        add_cycle_features_flag=(not args.no_cycle_features),
        timeout_mfe_mae_ratio=args.timeout_mfe_mae_ratio,
        timeout_mfe_min_move_atr=args.timeout_mfe_min_move_atr,
        apply_event_direction_veto=args.apply_event_direction_veto,
        add_multitask_diagnostics=(not args.no_multitask_diagnostics),
    )
