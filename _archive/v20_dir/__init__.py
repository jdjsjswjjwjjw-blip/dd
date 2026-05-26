"""
v20 — حزمة تشغيل QuantSystem V19/V20.

`v20/prepare_day_trading.py` نسخة مطابقة لملف الجذر `prepare_day_trading.py`
(يتضمن إصلاح Sprint 19: كل حالة non-TP — timeout / long_sl / short_sl —
تُقرَّر بمنطق MFE/MAE بدل NEUTRAL التلقائي، فينخفض الـ NEUTRAL جذرياً).

التشغيل:
    python -m v20.prepare_day_trading --mbo <path> ...

المرجع الوحيد للحقيقة هو ملف الجذر؛ هذه الحزمة تبقيه قابلاً للاستدعاء
كـ `v20.prepare_day_trading`.
"""
