# تصنيف الميزات بالوسوم — 130 ميزة

| # | الميزة | الوسم | الملاحظة |
|---|---|---|---|
| 1 | `cvd` | 📊 day_trade/هيكل/label | أصل — صافي ضغط الشراء/البيع |
| 2 | `session_cvd` | 🔗 مشتّق | مشتّق من cvd (إعادة تصفير بالجلسة) |
| 3 | `absorption_intensity` | 🌊 عمق السوق فقط | شدّة الامتصاص (AII) — للعمق |
| 4 | `cancel_ratio` | 🌊 عمق السوق فقط | نسبة الإلغاء — للعمق |
| 5 | `spoofing_ratio` | 🌊 عمق السوق فقط | نسبة spoofing — للعمق |
| 6 | `spoofing_duration` | 🔁 مكرّر | ≈ cancel_ratio (IC متطابق) — كلاهما سلوك إلغاء |
| 7 | `liquidity_trap` | 🌊 عمق السوق فقط | فخّ سيولة — للعمق |
| 8 | `kyle_lambda` | 🌊 عمق السوق فقط | أثر السعر/الحجم — سيولة (عمق) |
| 9 | `hawkes_intensity` | 📊 day_trade/هيكل/label | STRONG (intrabar) — تجمّع الأحداث |
| 10 | `vnet` | 🌊 عمق السوق فقط | صافي الحجم الموقّع — عمق/تدفّق |
| 11 | `hawkes_intrabar_sum` | 📊 day_trade/هيكل/label | STRONG — مجموع كثافة هوكس |
| 12 | `kyle_lambda_intrabar_mean` | 🌊 عمق السوق فقط | مشتّق kyle داخل الشمعة — عمق |
| 13 | `vnet_intrabar_last` | 🔗 مشتّق | مشتّق vnet (آخر قيمة) |
| 14 | `micro_atr` | 🔗 مشتّق | مشتّق atr (نطاق دقيق) |
| 15 | `volume_burst` | 📊 day_trade/هيكل/label | MODERATE — انفجار حجم |
| 16 | `inter_event_time` | 🌊 عمق السوق فقط | سرعة تنفيذ الأوامر — قراءة عمق (MODERATE) |
| 17 | `fisher_signal` | 📊 day_trade/هيكل/label | كشف تطرّف السعر |
| 18 | `anomaly` | 📊 day_trade/هيكل/label | علم شذوذ (WEAK) |
| 19 | `cvd_momentum` | 🔗 مشتّق | مشتّق من cvd (مشتقّته) |
| 20 | `cvd_price_divergence` | 🔗 مشتّق | مشتّق من cvd + السعر |
| 21 | `trend_strength` | 📊 day_trade/هيكل/label | قوة الاتجاه |
| 22 | `correction_depth` | 🌊 عمق السوق فقط | عمق التصحيح — للعمق |
| 23 | `liquidity_sweep` | 🌊 عمق السوق فقط | كنس السيولة — للعمق |
| 24 | `buy_absorption_approx` | 🌊 عمق السوق فقط | تقدير امتصاص شراء (IC=0) — للعمق |
| 25 | `sell_absorption_approx` | 🌊 عمق السوق فقط | تقدير امتصاص بيع (IC=0) — للعمق |
| 26 | `obi` | 🌊 عمق السوق فقط | عدم توازن الدفتر — قراءة عمق |
| 27 | `bid_wall_strength` | 🌊 عمق السوق فقط | جدار شراء (proxy هشّ) — للعمق |
| 28 | `ask_wall_strength` | 🌊 عمق السوق فقط | جدار بيع (proxy هشّ) — للعمق |
| 29 | `distance_to_wall` | 🌊 عمق السوق فقط | مسافة لأقرب جدار — للعمق |
| 30 | `gap_size` | 🌊 عمق السوق فقط | حجم الفجوة — للعمق |
| 31 | `liquidity_density` | 🌊 عمق السوق فقط | كثافة السيولة (MODERATE) — للعمق |
| 32 | `micro_price` | 🌊 عمق السوق فقط | سعر مرجّح بأحجام الكتاب — عمق |
| 33 | `current_vwap` | 📊 day_trade/هيكل/label | STRONG — مرجع القيمة العادلة |
| 34 | `vwap_z_score` | 📊 day_trade/هيكل/label | تأكيد compass للـ gate |
| 35 | `order_flow_imbalance` | 📊 day_trade/هيكل/label | OFI — تأكيد compass للـ gate |
| 36 | `pdh` | 📊 day_trade/هيكل/label | قمة الأمس — مستوى |
| 37 | `pdl` | 📊 day_trade/هيكل/label | قاع الأمس — مستوى |
| 38 | `dist_to_pdh` | 🔗 مشتّق | مشتّق pdh (مسافة) |
| 39 | `price_position` | 📊 day_trade/هيكل/label | موضع السعر في مدى اليوم |
| 40 | `pwh` | 📊 day_trade/هيكل/label | قمة الأسبوع — مستوى |
| 41 | `pwl` | 📊 day_trade/هيكل/label | قاع الأسبوع — مستوى |
| 42 | `weekly_price_position` | 📊 day_trade/هيكل/label | موضع السعر في مدى الأسبوع |
| 43 | `london_sess_high` | 📊 day_trade/هيكل/label | STRONG — قمة لندن الجارية (أقوى إشارة) |
| 44 | `london_sess_low` | 📊 day_trade/هيكل/label | MODERATE — قاع لندن الجاري |
| 45 | `dist_to_london_high_atr` | 🔗 مشتّق | مشتّق london_high ÷ atr |
| 46 | `dist_to_london_low_atr` | 🔗 مشتّق | مشتّق london_low ÷ atr |
| 47 | `tick_count` | 📊 day_trade/هيكل/label | STRONG — نشاط التداول |
| 48 | `mbp_bar_coverage` | 📊 day_trade/هيكل/label | تغطية MBP (جودة) |
| 49 | `mbo_bar_coverage` | 📊 day_trade/هيكل/label | تغطية MBO (جودة) |
| 50 | `mbp_roll_lob_coverage` | 📊 day_trade/هيكل/label | تغطية LOB متحرّكة (جودة) |
| 51 | `liquidity_gaps` | 🌊 عمق السوق فقط | فجوات السيولة (NOISE) — للعمق |
| 52 | `is_london` | 📊 day_trade/هيكل/label | علم لندن |
| 53 | `is_overlap` | 📊 day_trade/هيكل/label | علم التداخل |
| 54 | `is_ny` | 📊 day_trade/هيكل/label | علم نيويورك |
| 55 | `atr_14` | 📊 day_trade/هيكل/label | التقلّب — مرجع التطبيع |
| 56 | `is_session_break` | 📊 day_trade/هيكل/label | فاصل الجلسة |
| 57 | `bar_range` | 📊 day_trade/هيكل/label | مدى الشمعة |
| 58 | `body_ratio` | 📊 day_trade/هيكل/label | نسبة جسم الشمعة |
| 59 | `rsi_14` | 📊 day_trade/هيكل/label | القوة النسبية |
| 60 | `macd_hist` | 📊 day_trade/هيكل/label | هيستوغرام الماكد |
| 61 | `vwap_dist` | 🔗 مشتّق | مشتّق من vwap (مسافة) |
| 62 | `vwap_dist_1h_roll` | 🔗 مشتّق | مشتّق من vwap (نافذة ساعة) |
| 63 | `lob_imbalance` | 🌊 عمق السوق فقط | عدم توازن الكتاب — للعمق |
| 64 | `return_6b` | 📊 day_trade/هيكل/label | عائد 6 شموع |
| 65 | `return_1h` | 📊 day_trade/هيكل/label | عائد ساعة |
| 66 | `return_4h` | 📊 day_trade/هيكل/label | عائد 4 ساعات |
| 67 | `volume_ratio_6b` | 📊 day_trade/هيكل/label | نسبة حجم 6 شموع |
| 68 | `volume_ratio_1h` | 📊 day_trade/هيكل/label | نسبة حجم ساعة |
| 69 | `volume_ratio_4h` | 📊 day_trade/هيكل/label | نسبة حجم 4 ساعات |
| 70 | `cvd_slope_6b` | 🔗 مشتّق | مشتّق من cvd (ميل 6 شموع) |
| 71 | `cvd_slope_1h` | 🔗 مشتّق | مشتّق من cvd (ميل ساعة) |
| 72 | `cvd_slope_4h` | 🔗 مشتّق | مشتّق من cvd (ميل 4 ساعات) |
| 73 | `session_phase` | 📊 day_trade/هيكل/label | طور الجلسة |
| 74 | `time_since_london_open_min` | 📊 day_trade/هيكل/label | دقائق منذ فتح لندن |
| 75 | `time_to_london_close_min` | 📊 day_trade/هيكل/label | دقائق حتى إغلاق لندن |
| 76 | `time_since_ny_open_min` | 📊 day_trade/هيكل/label | دقائق منذ فتح نيويورك |
| 77 | `time_to_ny_close_min` | 📊 day_trade/هيكل/label | STRONG — دقائق حتى إغلاق نيويورك |
| 78 | `dow_sin` | 📊 day_trade/هيكل/label | يوم الأسبوع جيب |
| 79 | `dow_cos` | 📊 day_trade/هيكل/label | يوم الأسبوع جتا |
| 80 | `is_monday` | 📊 day_trade/هيكل/label | الإثنين |
| 81 | `is_friday` | 📊 day_trade/هيكل/label | الجمعة |
| 82 | `dom` | 📊 day_trade/هيكل/label | يوم الشهر |
| 83 | `dom_sin` | 🔗 مشتّق | مشتّق dom (جيب) |
| 84 | `dom_cos` | 🔗 مشتّق | مشتّق dom (جتا) |
| 85 | `is_month_end` | 📊 day_trade/هيكل/label | نهاية الشهر |
| 86 | `is_month_start` | 📊 day_trade/هيكل/label | بداية الشهر |
| 87 | `is_quarter_end` | 📊 day_trade/هيكل/label | نهاية الربع |
| 88 | `is_year_end` | 📊 day_trade/هيكل/label | نهاية السنة |
| 89 | `woy_sin` | 📊 day_trade/هيكل/label | أسبوع السنة جيب |
| 90 | `woy_cos` | 🔗 مشتّق | مشتّق woy (جتا) |
| 91 | `is_first_week_of_year` | 📊 day_trade/هيكل/label | أول أسبوع |
| 92 | `is_dst_transition_week` | 📊 day_trade/هيكل/label | أسبوع DST |
| 93 | `is_event_window` | 📊 day_trade/هيكل/label | ⚠️ أصفار فارغة (placeholder) |
| 94 | `cycle_structure_score` | 📊 day_trade/هيكل/label | قوة HH/HL/LH/LL (سببي) |
| 95 | `cycle_bars_since_swing` | 📊 day_trade/هيكل/label | شموع منذ آخر swing |
| 96 | `cycle_trend_maturity` | 📊 day_trade/هيكل/label | نضج الترند |
| 97 | `cycle_momentum_decay` | 📊 day_trade/هيكل/label | تآكل الزخم |
| 98 | `cycle_phase_acc_prob` | 🌊 عمق السوق فقط | طور Wyckoff (قاعدة ضعيفة) → هدف SSL |
| 99 | `cycle_phase_markup_prob` | 🌊 عمق السوق فقط | طور Wyckoff (قاعدة ضعيفة) → هدف SSL |
| 100 | `cycle_phase_dist_prob` | 🌊 عمق السوق فقط | طور Wyckoff (قاعدة ضعيفة) → هدف SSL |
| 101 | `cycle_phase_markdown_prob` | 🌊 عمق السوق فقط | طور Wyckoff (قاعدة ضعيفة) → هدف SSL |
| 102 | `cycle_position` | 🌊 عمق السوق فقط | ⚠️ 4 قيم فقط (ضعيف) → هدف SSL |
| 103 | `cycle_hurst` | 📊 day_trade/هيكل/label | أُسّ هرست |
| 104 | `cycle_fractal_dim` | 📊 day_trade/هيكل/label | البُعد الفركتالي |
| 105 | `cycle_mtf_alignment` | 📊 day_trade/هيكل/label | توافق متعدّد الأطر |
| 106 | `event_score_continuous` | 📊 day_trade/هيكل/label | درجة الحدث المستمرّة (gate ناعم) |
| 107 | `event_score_binary` | 📊 day_trade/هيكل/label | درجة الحدث الثنائية (gate صلب) |
| 108 | `hawkes_z_raw` | 🔗 مشتّق | مشتّق hawkes (z خام) |
| 109 | `absorb_z_raw` | 🌊 عمق السوق فقط | امتصاص z خام — مدخل عمق |
| 110 | `kyle_z_raw` | 🔗 مشتّق | مشتّق kyle (z خام) |
| 111 | `bar_cvd_delta` | 🔗 مشتّق | مشتّق من cvd (دلتا الشمعة) |
| 112 | `cvd_bar_5m` | 🔗 مشتّق | مشتّق من cvd (تجميع 5د) |
| 113 | `cvd_direction_ratio_5m` | 🔗 مشتّق | مشتّق من cvd (نسبة اتجاه) |
| 114 | `cvd_intensity_vs_atr` | 🔗 مشتّق | مشتّق من cvd ÷ atr |
| 115 | `cvd_divergence_at_level` | 📊 day_trade/هيكل/label | تأكيد compass للـ gate — يبقى |
| 116 | `cvd_consecutive_imbalance` | 🔗 مشتّق | مشتّق من cvd (تتالٍ) |
| 117 | `obi_net` | 🔁 مكرّر | = obi (IC متطابق) |
| 118 | `cvd_cumulative` | 🔁 مكرّر | = cvd (IC متطابق) |
| 119 | `dist_to_session_high_atr` | 🔗 مشتّق | مشتّق session_high ÷ atr |
| 120 | `dist_to_vwap_atr` | 🔗 مشتّق | مشتّق من vwap ÷ atr |
| 121 | `dist_to_pdh_atr` | 🔗 مشتّق | مشتّق pdh ÷ atr |
| 122 | `dist_to_pdl_atr` | 🔗 مشتّق | مشتّق pdl ÷ atr |
| 123 | `dist_to_pwh_atr` | 🔗 مشتّق | مشتّق pwh ÷ atr |
| 124 | `dist_to_pwl_atr` | 🔗 مشتّق | مشتّق pwl ÷ atr |
| 125 | `regime_label_grouped` | 📊 day_trade/هيكل/label | تصنيف النظام السوقي |
| 126 | `is_warmup` | 📊 day_trade/هيكل/label | فترة الإحماء (استبعاد) |
| 127 | `hawkes_x_session_phase` | 🔗 مشتّق | تفاعل hawkes × طور الجلسة |
| 128 | `tick_count_x_session_phase` | 🔗 مشتّق | تفاعل tick_count × طور الجلسة |
| 129 | `iceberg_count_5m` | 🌊 عمق السوق فقط | عدد iceberg (IC=0) — للعمق |
| 130 | `iceberg_total_volume_5m` | 🌊 عمق السوق فقط | حجم iceberg (IC=0) — للعمق |

## الإحصاء

- 📊 day_trade/هيكل/label: **67**
- 🔗 مشتّق: **31**
- 🌊 عمق السوق فقط: **29**
- 🔁 مكرّر: **3**
- المجموع: 130

غير مصنّف: لا شيء — كل الـ 130 مصنّفة ✅
