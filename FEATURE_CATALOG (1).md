# كتالوج الميزات الكامل — QuantSystem

> إجمالي الميزات المُصدَّرة: 130 | مُقاسة في تدقيق الـ IC: 80

> الأعمدة: الميزة | العائلة | المصدر | الوظيفة | حُكم IC (|IC|/verdict) | قرار الـ workflow


## التدفّق / المايكروستركتشر (38)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `cvd` | MBO/trades | دلتا الحجم التراكمي — صافي ضغط الشراء مقابل البيع | 0.014/NONLINEAR_EDGE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `session_cvd` | MBO/trades | CVD مُعاد تصفيره عند بداية كل جلسة | 0.011/NONLINEAR_EDGE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_momentum` | MBO/trades | تسارع/تباطؤ الـ CVD (مشتقّته) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_price_divergence` | MBO/trades | تباعد CVD عن السعر — إشارة انعكاس محتملة | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_slope_6b` | MBO/trades | ميل CVD على 6 شموع (اتجاه قصير) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_slope_1h` | MBO/trades | ميل CVD على ساعة | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_slope_4h` | MBO/trades | ميل CVD على 4 ساعات (اتجاه أكبر) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `bar_cvd_delta` | MBO/trades | صافي دلتا CVD داخل الشمعة | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_bar_5m` | MBO/trades | CVD المجمّع على شمعة 5 دقائق | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_direction_ratio_5m` | MBO/trades | نسبة اتجاه CVD (شراء/بيع) على 5د | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_intensity_vs_atr` | MBO/trades | شدّة CVD منسوبة للتقلّب (ATR) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_divergence_at_level` | MBO/trades | تباعد CVD عند مستوى سعري — تأكيد compass للـ gate | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_consecutive_imbalance` | MBO/trades | عدم توازن CVD متتالٍ (ضغط مستمرّ) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `cvd_cumulative` | MBO/trades | CVD تراكمي عبر الزمن | 0.014/NONLINEAR_EDGE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `obi` | MBP-10 | عدم توازن دفتر الأوامر — ضغط bid مقابل ask | 0.013/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `obi_net` | MBP-10 | صافي OBI (نسخة مطبّعة) | 0.013/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `order_flow_imbalance` | MBP-10 | OFI — عدم توازن تدفّق الأوامر (تأكيد compass) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `micro_price` | MBP-10 | السعر المرجّح بأحجام أفضل bid/ask | 0.023/WEAK | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `current_vwap` | trades | السعر المرجّح بالحجم — مرجع القيمة العادلة | 0.102/STRONG | المجموعة 1 → day_trade (مفيد فردياً) |
| `vwap_z_score` | trades | انحراف السعر عن VWAP بوحدات z (تأكيد compass) | 0.012/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `vwap_dist` | trades | مسافة السعر عن VWAP | 0.013/NONLINEAR_EDGE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `vwap_dist_1h_roll` | trades | مسافة VWAP على نافذة ساعة متحرّكة | 0.015/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `dist_to_vwap_atr` | trades | مسافة VWAP منسوبة لـ ATR | 0.013/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `kyle_lambda` | MBO/trades | لامبدا كايل — أثر السعر لكل وحدة حجم (سيولة) | 0.009/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `kyle_lambda_intrabar_mean` | MBO/trades | متوسط لامبدا كايل داخل الشمعة | 0.063/MODERATE | المجموعة 1 → day_trade (مفيد فردياً) |
| `kyle_z_raw` | MBO/trades | لامبدا كايل بوحدات z خام | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `hawkes_intensity` | MBO/trades | كثافة هوكس — تجمّع الأحداث (نشاط ذاتي التحفيز) | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `hawkes_intrabar_sum` | MBO/trades | مجموع كثافة هوكس داخل الشمعة | 0.106/STRONG | المجموعة 1 → day_trade (مفيد فردياً) |
| `hawkes_z_raw` | MBO/trades | كثافة هوكس بوحدات z خام | — | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `tick_count` | MBO/trades | عدد التيكات في الشمعة — نشاط التداول | 0.106/STRONG | المجموعة 1 → day_trade (مفيد فردياً) |
| `vnet` | MBO/trades | صافي الحجم الموقّع (volume net) | 0.015/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `vnet_intrabar_last` | MBO/trades | آخر قيمة vnet داخل الشمعة | 0.008/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `volume_burst` | trades | انفجار حجم مفاجئ مقابل المعتاد | 0.065/MODERATE | المجموعة 1 → day_trade (مفيد فردياً) |
| `inter_event_time` | MBO | الزمن بين الأحداث — سرعة تنفيذ الأوامر | 0.093/MODERATE | المجموعة 1 → day_trade (مفيد فردياً) |
| `fisher_signal` | trades | تحويل فيشر للسعر — كشف نقاط التطرّف | 0.013/NOISE | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `anomaly` | trades | علم شذوذ إحصائي في النشاط | 0.044/WEAK | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `hawkes_x_session_phase` | تفاعل | تفاعل كثافة هوكس مع طور الجلسة | 0.028/WEAK | المجموعة 1 (ضعيف فردياً) أو دعم العمق |
| `tick_count_x_session_phase` | تفاعل | تفاعل عدد التيكات مع طور الجلسة | 0.029/WEAK | المجموعة 1 (ضعيف فردياً) أو دعم العمق |

## العمق / دفتر الأوامر (يدوي) (9)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `bid_wall_strength` | MBP-10 | قوة جدار الشراء (proxy هشّ من buy_ratio) | — | المجموعة 2 → العمق الخام (SSL/CNN) |
| `ask_wall_strength` | MBP-10 | قوة جدار البيع (proxy هشّ) | — | المجموعة 2 → العمق الخام (SSL/CNN) |
| `distance_to_wall` | MBP-10 | مسافة السعر عن أقرب جدار سيولة | — | المجموعة 2 → العمق الخام (SSL/CNN) |
| `liquidity_density` | MBP-10 | كثافة السيولة عبر المستويات | 0.094/MODERATE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `liquidity_gaps` | MBP-10 | فجوات السيولة في الكتاب | 0.0/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `liquidity_sweep` | MBP-10 | كنس السيولة — اختراق مستوى مع إزالة جدار | 0.027/WEAK | المجموعة 2 → العمق الخام (SSL/CNN) |
| `lob_imbalance` | MBP-10 | عدم توازن دفتر الأوامر الكلّي | — | المجموعة 2 → العمق الخام (SSL/CNN) |
| `correction_depth` | trades | عمق التصحيح السعري | — | المجموعة 2 → العمق الخام (SSL/CNN) |
| `gap_size` | MBP-10 | حجم الفجوة السعرية | 0.024/WEAK | المجموعة 2 → العمق الخام (SSL/CNN) |

## مستوى الأمر (MBO) (9)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `absorption_intensity` | MBO | شدّة الامتصاص — حجم موقّع عالٍ مع سعر ثابت (AII) | 0.037/WEAK | المجموعة 2 → العمق الخام (SSL/CNN) |
| `buy_absorption_approx` | MBO | تقدير امتصاص الشراء | 0.0/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `sell_absorption_approx` | MBO | تقدير امتصاص البيع | 0.0/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `cancel_ratio` | MBO | نسبة الإلغاء — أوامر تُلغى قبل التنفيذ | 0.009/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `spoofing_ratio` | MBO | نسبة الـ spoofing — أوامر وهمية | 0.015/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `spoofing_duration` | MBO | مدّة بقاء الأوامر الوهمية | 0.009/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `liquidity_trap` | MBO | فخّ سيولة — جدار يجذب ثم يختفي | 0.016/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `iceberg_count_5m` | MBO | عدد أوامر iceberg المكتشفة في 5د | 0.0/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |
| `iceberg_total_volume_5m` | MBO | الحجم الكلّي لأوامر iceberg في 5د | 0.0/NOISE | المجموعة 2 → العمق الخام (SSL/CNN) |

## الهيكل / المستويات (16)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `pdh` | OHLCV | قمة اليوم السابق (مستوى مقاومة) | 0.079/MODERATE | المجموعة 1 → day_trade |
| `pdl` | OHLCV | قاع اليوم السابق (مستوى دعم) | 0.066/MODERATE | المجموعة 1 → day_trade |
| `dist_to_pdh` | OHLCV | مسافة السعر عن قمة الأمس | 0.011/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `dist_to_pdh_atr` | OHLCV | مسافة قمة الأمس منسوبة لـ ATR | 0.014/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `dist_to_pdl_atr` | OHLCV | مسافة قاع الأمس منسوبة لـ ATR | — | المجموعة 1 → day_trade |
| `price_position` | OHLCV | موضع السعر في مدى اليوم [0,1] | 0.021/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `pwh` | OHLCV | قمة الأسبوع السابق | — | المجموعة 1 → day_trade |
| `pwl` | OHLCV | قاع الأسبوع السابق | — | المجموعة 1 → day_trade |
| `dist_to_pwh_atr` | OHLCV | مسافة قمة الأسبوع منسوبة لـ ATR | — | المجموعة 1 → day_trade |
| `dist_to_pwl_atr` | OHLCV | مسافة قاع الأسبوع منسوبة لـ ATR | — | المجموعة 1 → day_trade |
| `weekly_price_position` | OHLCV | موضع السعر في مدى الأسبوع [0,1] | — | المجموعة 1 → day_trade |
| `london_sess_high` | OHLCV | قمة جلسة لندن الجارية (سببي) — أقوى إشارة IC | 0.11/STRONG | المجموعة 1 → day_trade |
| `london_sess_low` | OHLCV | قاع جلسة لندن الجاري (سببي) | 0.08/MODERATE | المجموعة 1 → day_trade |
| `dist_to_london_high_atr` | OHLCV | مسافة قمة لندن منسوبة لـ ATR | — | المجموعة 1 → day_trade |
| `dist_to_london_low_atr` | OHLCV | مسافة قاع لندن منسوبة لـ ATR | 0.07/MODERATE | المجموعة 1 → day_trade |
| `dist_to_session_high_atr` | OHLCV | مسافة قمة الجلسة منسوبة لـ ATR | 0.067/UNSTABLE_WALKFORWARD | المجموعة 1 → day_trade |

## الموسمية (زمني/تقويمي) (21)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `session_phase` | timestamp | طور الجلسة (افتتاح/منتصف/إغلاق/خارج) | 0.093/MODERATE | المجموعة 1 → day_trade |
| `time_since_london_open_min` | timestamp | دقائق منذ فتح لندن | — | المجموعة 1 → day_trade |
| `time_to_london_close_min` | timestamp | دقائق حتى إغلاق لندن | — | المجموعة 1 → day_trade |
| `time_since_ny_open_min` | timestamp | دقائق منذ فتح نيويورك | — | المجموعة 1 → day_trade |
| `time_to_ny_close_min` | timestamp | دقائق حتى إغلاق نيويورك — STRONG في IC | 0.106/STRONG | المجموعة 1 → day_trade |
| `dow_sin` | timestamp | يوم الأسبوع (جيب — دوري) | — | المجموعة 1 → day_trade |
| `dow_cos` | timestamp | يوم الأسبوع (جيب تمام — دوري) | 0.018/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `is_monday` | timestamp | علم الإثنين | — | المجموعة 1 → day_trade |
| `is_friday` | timestamp | علم الجمعة | — | المجموعة 1 → day_trade |
| `dom` | timestamp | يوم الشهر | — | المجموعة 1 → day_trade |
| `dom_sin` | timestamp | يوم الشهر (جيب — دوري) | — | المجموعة 1 → day_trade |
| `dom_cos` | timestamp | يوم الشهر (جيب تمام — دوري) | — | المجموعة 1 → day_trade |
| `is_month_end` | timestamp | علم نهاية الشهر (تدفّقات مؤسسية) | — | المجموعة 1 → day_trade |
| `is_month_start` | timestamp | علم بداية الشهر | — | المجموعة 1 → day_trade |
| `is_quarter_end` | timestamp | علم نهاية الربع | — | المجموعة 1 → day_trade |
| `is_year_end` | timestamp | علم نهاية السنة | — | المجموعة 1 → day_trade |
| `woy_sin` | timestamp | أسبوع السنة (جيب) | — | المجموعة 1 → day_trade |
| `woy_cos` | timestamp | أسبوع السنة (جيب تمام) | — | المجموعة 1 → day_trade |
| `is_first_week_of_year` | timestamp | علم أول أسبوع في السنة | — | المجموعة 1 → day_trade |
| `is_dst_transition_week` | timestamp | علم أسبوع تحوّل التوقيت الصيفي | — | المجموعة 1 → day_trade |
| `is_event_window` | timestamp | نافذة حدث اقتصادي — ⚠️ أصفار فارغة (placeholder) | — | المجموعة 1 → day_trade |

## الدورة السعرية (12)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `cycle_structure_score` | OHLCV | قوة نمط HH/HL/LH/LL [-1,1] (سببي) | 0.03/WEAK | المجموعة 1 → day_trade (دورة بنيوية سببية) |
| `cycle_bars_since_swing` | OHLCV | عدد الشموع منذ آخر swing مؤكَّد | 0.014/NOISE | المجموعة 1 → day_trade (دورة بنيوية سببية) |
| `cycle_trend_maturity` | OHLCV | نضج الترند (young/mature/exhausted) | 0.009/NOISE | المجموعة 1 → day_trade (دورة بنيوية سببية) |
| `cycle_momentum_decay` | OHLCV | تآكل الزخم (0=قوي، 1=متآكل) | 0.013/NOISE | المجموعة 1 → day_trade (دورة بنيوية سببية) |
| `cycle_phase_acc_prob` | OHLCV | احتمال طور التجميع (Wyckoff، قاعدة) | 0.014/NOISE | feature ضعيف + هدف SSL (انعكاس حقيقي) |
| `cycle_phase_markup_prob` | OHLCV | احتمال طور الصعود (قاعدة) | 0.018/NOISE | feature ضعيف + هدف SSL (انعكاس حقيقي) |
| `cycle_phase_dist_prob` | OHLCV | احتمال طور التوزيع (قاعدة) | 0.011/NOISE | feature ضعيف + هدف SSL (انعكاس حقيقي) |
| `cycle_phase_markdown_prob` | OHLCV | احتمال طور الهبوط (قاعدة) | 0.018/NOISE | feature ضعيف + هدف SSL (انعكاس حقيقي) |
| `cycle_position` | OHLCV | موضع الدورة — ⚠️ 4 قيم فقط (أضعف من اسمه) | 0.01/NONLINEAR_EDGE | feature ضعيف + هدف SSL (انعكاس حقيقي) |
| `cycle_hurst` | OHLCV | أُسّ هرست (>0.5 ترند، <0.5 ارتداد) | — | المجموعة 1 → day_trade (دورة بنيوية سببية) |
| `cycle_fractal_dim` | OHLCV | البُعد الفركتالي للسعر | — | المجموعة 1 → day_trade (دورة بنيوية سببية) |
| `cycle_mtf_alignment` | OHLCV | توافق متعدّد الأطر (1×/4×/16×) | 0.007/NONLINEAR_EDGE | المجموعة 1 → day_trade (دورة بنيوية سببية) |

## الزخم / التقلّب الكلاسيكي (13)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `trend_strength` | OHLCV | قوة الاتجاه | 0.017/NOISE | المجموعة 1 → day_trade |
| `atr_14` | OHLCV | المدى الحقيقي المتوسط (تقلّب) | 0.063/MODERATE | المجموعة 1 → day_trade |
| `micro_atr` | OHLCV | ATR على نطاق دقيق | 0.03/WEAK | المجموعة 1 → day_trade |
| `bar_range` | OHLCV | مدى الشمعة (أعلى-أدنى) | — | المجموعة 1 → day_trade |
| `body_ratio` | OHLCV | نسبة جسم الشمعة لمداها | 0.01/NOISE | المجموعة 1 → day_trade |
| `rsi_14` | OHLCV | مؤشر القوة النسبية | 0.023/NOISE | المجموعة 1 → day_trade |
| `macd_hist` | OHLCV | هيستوغرام الماكد | 0.02/NOISE | المجموعة 1 → day_trade |
| `return_6b` | OHLCV | العائد على 6 شموع | 0.011/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `return_1h` | OHLCV | العائد على ساعة | 0.012/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `return_4h` | OHLCV | العائد على 4 ساعات | 0.008/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `volume_ratio_6b` | trades | نسبة الحجم على 6 شموع | 0.019/NOISE | المجموعة 1 → day_trade |
| `volume_ratio_1h` | trades | نسبة الحجم على ساعة | 0.022/NOISE | المجموعة 1 → day_trade |
| `volume_ratio_4h` | trades | نسبة الحجم على 4 ساعات | 0.037/WEAK | المجموعة 1 → day_trade |

## أعلام الجلسات (4)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `is_london` | timestamp | علم جلسة لندن | 0.018/NONLINEAR_EDGE | المجموعة 1 → day_trade |
| `is_overlap` | timestamp | علم تداخل لندن/نيويورك | — | المجموعة 1 → day_trade |
| `is_ny` | timestamp | علم جلسة نيويورك | 0.078/MODERATE | المجموعة 1 → day_trade |
| `is_session_break` | timestamp | علم فاصل الجلسة (منع تجاوز النوافذ) | — | المجموعة 1 → day_trade |

## التغطية / الجودة (4)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `mbp_bar_coverage` | MBP-10 | تغطية بيانات MBP في الشمعة | 0.037/WARMUP_RIDER | المجموعة 1 → day_trade |
| `mbo_bar_coverage` | MBO | تغطية بيانات MBO في الشمعة | 0.072/MODERATE | المجموعة 1 → day_trade |
| `mbp_roll_lob_coverage` | MBP-10 | تغطية LOB المتحرّكة | — | المجموعة 1 → day_trade |
| `is_warmup` | derived | علم فترة الإحماء (استبعاد من التدريب) | — | المجموعة 1 → day_trade |

## الحدث (3)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `event_score_continuous` | derived | درجة الحدث المستمرّة [0,1] (gate ناعم) | — | المجموعة 1 → day_trade |
| `event_score_binary` | derived | درجة الحدث الثنائية (gate صلب) | — | المجموعة 1 → day_trade |
| `absorb_z_raw` | MBO | امتصاص بوحدات z خام (مدخل event score) | — | المجموعة 1 → day_trade |

## النظام السوقي (1)

| الميزة | المصدر | الوظيفة | IC | القرار |
|---|---|---|---|---|
| `regime_label_grouped` | derived | تصنيف النظام السوقي (trending/ranging/volatile) | — | المجموعة 1 → day_trade |

---

*ملاحظة: الميزات بـ IC='—' لم تكن ضمن الـ 80 المُقاسة (تفاعلات/مشتقّات/أعلام)، أو تُقاس عبر ميزتها الأمّ.*

---

## خلاصة القرارات (تقسيم الميزات الـ 130)

### المجموعة 1 — تبقى في day_trade (الهيكل + التدفّق منخفض الأبعاد)
- **مفيدة فردياً (أبقها بثقة):** `current_vwap`, `hawkes_intrabar_sum`, `tick_count` (STRONG) + `liquidity_density`, `inter_event_time`, `volume_burst`, `kyle_lambda_intrabar_mean`, `time_to_ny_close_min` (MODERATE)
- **الهيكل (16):** كل مستويات pdh/pdl/pwh/pwl/london + dist_to_*_atr — أقوى عائلة في الـ IC
- **الموسمية (21):** من timestamp — كلها سببية
- **الدورة البنيوية (7):** structure_score, bars_since_swing, trend_maturity, momentum_decay, hurst, fractal_dim, mtf_alignment
- **الزخم الكلاسيكي (13)** + أعلام الجلسات (4) + الجودة (4) + الحدث/regime

### المجموعة 2 — تنتقل للعمق الخام (SSL/CNN) — تُحذف من parquet
- **العمق اليدوي (9):** walls, liquidity_density/gaps/sweep, lob_imbalance, distance_to_wall, correction_depth, gap_size
- **مستوى الأمر (9):** absorption_intensity, buy/sell_absorption, cancel_ratio, spoofing_ratio/duration, liquidity_trap, iceberg_count/volume

### حالة خاصة — الدورة (طور Wyckoff)
`cycle_phase_*_prob` و`cycle_position`: feature ضعيف (قاعدة) + **هدف SSL** (يتعلّم انعكاساً حقيقياً من السعر، لا يقلّد القاعدة)

---

## أهمّ 3 نتائج من الكتالوج

1. **العمق ومستوى-الأمر = NOISE فردياً** — بما فيها iceberg/absorption (0.000). لكن هذا يقوّي نقلها للعميق (قيمتها مكانية تفاعلية، لا فردية).
2. **التدفّق مختلط** — `hawkes`/`tick_count`/`vwap` قوية، لكن `obi`/`cvd` المفردة ضعيفة (قيمتها في التفاعل).
3. **الهيكل والموسمية أقوى العائلات فردياً** — المستويات و`time_to_ny_close` هي العمود الفقري المُثبَت.

> الميزات بـ IC='—': تفاعلات/مشتقّات/أعلام لم تُقَس مفردةً، أو تُقاس عبر ميزتها الأمّ (مثلاً cvd_slope عبر cvd).
