"""
self_supervised/explain_patterns.py
═══════════════════════════════════════════════════════════════════════════
Phase F: Interpretability & Pattern Discovery — بعد SSL Validation.

يستخرج إجابات على أسئلة محورية بعد التدريب:
    ❓ "إيه الـ patterns اللي اكتشفها النموذج؟"
    ❓ "إيه الدورات السعرية اللي اتعلمها؟"
    ❓ "في أي regime/session/cycle الأنماط بتظهر؟"
    ❓ "أي features الأكثر تأثيراً على القرارات؟"

ينتج:
    1. interpretability_report.md — تقرير قابل للقراءة
    2. patterns_analysis.json — بيانات تفصيلية
    3. cluster_statistics.csv — لكل cluster، characteristics

الاستخدام (بعد ما الـ pipeline يخلص):
    python self_supervised/explain_patterns.py \\
        --features pipeline/day_trading_features.parquet \\
        --embeddings checkpoints/ssl/embeddings/embeddings.npy \\
        --direction-head checkpoints/ssl/direction/best_direction_head.pt \\
        --output checkpoints/ssl/interpretability/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from self_supervised.fine_tune_direction import DirectionHead


def kmeans_clustering(X: np.ndarray, n_clusters: int = 8,
                      max_iter: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """K-Means من scratch (بدون sklearn dependency للسلامة)."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    # K-means++ initialization
    centers = [X[rng.randint(n)]]
    for _ in range(n_clusters - 1):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        idx = rng.choice(n, p=probs)
        centers.append(X[idx])
    centers = np.array(centers)

    labels = np.zeros(n, dtype=np.int32)
    for it in range(max_iter):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
        new_labels = dists.argmin(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(axis=0)
    return labels, centers


def analyze_clusters(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray,
    direction_preds: np.ndarray | None = None,
    direction_probs: np.ndarray | None = None,
) -> list[dict]:
    """يحلل خصائص كل cluster."""
    n_clusters = int(labels.max()) + 1
    cluster_stats = []
    for k in range(n_clusters):
        mask = labels == k
        n_in_cluster = int(mask.sum())
        if n_in_cluster < 5:
            continue

        sub = df[mask].copy()

        # Regime distribution
        regime_dist = {}
        if 'regime_label' in sub.columns:
            counts = sub['regime_label'].value_counts(normalize=True).to_dict()
            regime_dist = {k: float(v) for k, v in counts.items()}

        # Session phase distribution
        session_dist = {}
        if 'session_phase' in sub.columns:
            counts = sub['session_phase'].value_counts(normalize=True).to_dict()
            phase_names = {0: 'opening', 1: 'middle', 2: 'closing', 3: 'outside'}
            session_dist = {phase_names.get(int(k), f'phase_{k}'): float(v) for k, v in counts.items()}

        # Wyckoff phase distribution (from cycle features)
        wyckoff_dist = {}
        for phase_name in ('acc', 'markup', 'dist', 'markdown'):
            col = f'cycle_phase_{phase_name}_prob'
            if col in sub.columns:
                wyckoff_dist[phase_name] = float(sub[col].mean())

        # Day-of-week distribution
        dow_dist = {}
        if 'is_monday' in sub.columns and 'is_friday' in sub.columns:
            dow_dist['monday_pct'] = float(sub['is_monday'].mean())
            dow_dist['friday_pct'] = float(sub['is_friday'].mean())

        # Cycle position
        cycle_pos = {}
        if 'cycle_position' in sub.columns:
            cycle_pos['mean'] = float(sub['cycle_position'].mean())
            cycle_pos['std'] = float(sub['cycle_position'].std())

        # Direction prediction analysis
        direction_analysis = {}
        if direction_preds is not None:
            sub_preds = direction_preds[mask]
            direction_analysis = {
                'n_predictions': int(len(sub_preds)),
                'long_pct': float((sub_preds == 0).mean()),
                'short_pct': float((sub_preds == 1).mean()),
            }
            if direction_probs is not None:
                sub_probs = direction_probs[mask]
                direction_analysis['mean_confidence'] = float(sub_probs.max(axis=1).mean())

        # Market metrics
        market_metrics = {}
        if 'atr_14' in sub.columns:
            market_metrics['atr_mean'] = float(sub['atr_14'].mean())
            market_metrics['atr_std'] = float(sub['atr_14'].std())
        if 'hawkes_intensity' in sub.columns:
            market_metrics['hawkes_mean'] = float(sub['hawkes_intensity'].mean())
        if 'cvd' in sub.columns:
            market_metrics['cvd_std'] = float(sub['cvd'].std())

        # Discover dominant characteristic
        dominant_regime = max(regime_dist.items(), key=lambda x: x[1])[0] if regime_dist else 'unknown'
        dominant_session = max(session_dist.items(), key=lambda x: x[1])[0] if session_dist else 'unknown'
        dominant_wyckoff = max(wyckoff_dist.items(), key=lambda x: x[1])[0] if wyckoff_dist else 'unknown'

        cluster_stats.append({
            'cluster_id': k,
            'n_samples': n_in_cluster,
            'pct_of_total': float(n_in_cluster / len(labels) * 100),
            'dominant_regime': dominant_regime,
            'dominant_session': dominant_session,
            'dominant_wyckoff': dominant_wyckoff,
            'regime_distribution': regime_dist,
            'session_distribution': session_dist,
            'wyckoff_distribution': wyckoff_dist,
            'day_of_week': dow_dist,
            'cycle_position': cycle_pos,
            'direction': direction_analysis,
            'market_metrics': market_metrics,
        })

    cluster_stats.sort(key=lambda x: x['n_samples'], reverse=True)
    return cluster_stats


def feature_importance_via_perturbation(
    direction_head: DirectionHead,
    embeddings: np.ndarray,
    labels: np.ndarray,
    feature_groups: dict[str, slice],
    mu: np.ndarray, sigma: np.ndarray,
    device: torch.device,
) -> dict:
    """يقيس importance لكل feature group بـ perturbation:
       perturb group → قياس انخفاض الـ accuracy.
    """
    direction_head.eval()
    importance = {}

    # Baseline
    X_norm = (embeddings - mu) / sigma
    X_t = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    with torch.no_grad():
        baseline_logits = direction_head(X_t)
        baseline_preds = baseline_logits.argmax(dim=-1).cpu().numpy()
    baseline_acc = float((baseline_preds == labels).mean())

    # Per-group: shuffle that group's values, measure drop
    rng = np.random.RandomState(42)
    for group_name, group_slice in feature_groups.items():
        X_shuffled = X_norm.copy()
        n = X_shuffled.shape[0]
        perm = rng.permutation(n)
        X_shuffled[:, group_slice] = X_shuffled[perm, group_slice]
        X_t = torch.from_numpy(X_shuffled.astype(np.float32)).to(device)
        with torch.no_grad():
            shuffled_logits = direction_head(X_t)
            shuffled_preds = shuffled_logits.argmax(dim=-1).cpu().numpy()
        shuffled_acc = float((shuffled_preds == labels).mean())
        importance[group_name] = {
            'accuracy_drop': baseline_acc - shuffled_acc,
            'baseline_acc': baseline_acc,
            'perturbed_acc': shuffled_acc,
            'importance_pct': float((baseline_acc - shuffled_acc) / max(baseline_acc, 1e-6) * 100),
        }

    return importance


def analyze_seasonal_patterns(
    df: pd.DataFrame,
    direction_preds: np.ndarray,
    direction_probs: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """تحليل dedicated للأنماط الموسمية — accuracy + bias حسب:
        - hour of day (via session features)
        - day of week
        - session phase (opening/middle/closing/outside)
        - month-end / quarter-end / year-end
        - DST transition weeks
        - calendar event windows
    """
    n = len(df)
    seasonal = {
        'total_samples': int(n),
        'overall_accuracy': float((direction_preds == true_labels).mean()),
        'overall_confidence': float(direction_probs.max(axis=1).mean()),
    }

    # ── Helper for per-bucket accuracy ──
    def _bucket_stats(mask: np.ndarray, name: str) -> dict:
        n_sub = int(mask.sum())
        if n_sub < 3:
            return {'n_samples': n_sub, 'accuracy': None, 'confidence': None,
                    'long_pct': None, 'short_pct': None}
        preds_sub = direction_preds[mask]
        labels_sub = true_labels[mask]
        probs_sub = direction_probs[mask]
        return {
            'n_samples': n_sub,
            'accuracy': float((preds_sub == labels_sub).mean()),
            'confidence': float(probs_sub.max(axis=1).mean()),
            'long_pct': float((preds_sub == 0).mean()),
            'short_pct': float((preds_sub == 1).mean()),
            'pct_of_total': float(n_sub / n * 100),
        }

    # ── 1. Session Phase Analysis ──
    if 'session_phase' in df.columns:
        phase_names = {0: 'opening', 1: 'middle', 2: 'closing', 3: 'outside'}
        seasonal['session_phase'] = {}
        for code, name in phase_names.items():
            mask = (df['session_phase'] == code).to_numpy()
            seasonal['session_phase'][name] = _bucket_stats(mask, name)

    # ── 2. Day-of-week Analysis ──
    if 'is_monday' in df.columns and 'is_friday' in df.columns:
        seasonal['day_of_week'] = {
            'monday': _bucket_stats((df['is_monday'] == 1).to_numpy(), 'mon'),
            'friday': _bucket_stats((df['is_friday'] == 1).to_numpy(), 'fri'),
            'rest_of_week': _bucket_stats(
                ((df['is_monday'] == 0) & (df['is_friday'] == 0)).to_numpy(), 'rest'
            ),
        }

    # ── 3. Time within London/NY session ──
    if 'time_since_london_open_min' in df.columns:
        london_active = (df['time_since_london_open_min'] > 0).to_numpy()
        seasonal['london_session'] = {
            'active': _bucket_stats(london_active, 'london_active'),
            'first_hour': _bucket_stats(
                ((df['time_since_london_open_min'] > 0) &
                 (df['time_since_london_open_min'] <= 60)).to_numpy(), 'london_first_hr'
            ),
            'last_hour': _bucket_stats(
                ((df['time_to_london_close_min'] > 0) &
                 (df['time_to_london_close_min'] <= 60)).to_numpy(), 'london_last_hr'
            ),
        }

    if 'time_since_ny_open_min' in df.columns:
        seasonal['ny_session'] = {
            'active': _bucket_stats(
                (df['time_since_ny_open_min'] > 0).to_numpy(), 'ny_active'
            ),
            'first_hour': _bucket_stats(
                ((df['time_since_ny_open_min'] > 0) &
                 (df['time_since_ny_open_min'] <= 60)).to_numpy(), 'ny_first_hr'
            ),
            'last_hour': _bucket_stats(
                ((df['time_to_ny_close_min'] > 0) &
                 (df['time_to_ny_close_min'] <= 60)).to_numpy(), 'ny_last_hr'
            ),
        }

    # ── 4. Calendar effects ──
    seasonal['calendar_effects'] = {}
    for col_name, label in [
        ('is_month_end', 'month_end'),
        ('is_month_start', 'month_start'),
        ('is_quarter_end', 'quarter_end'),
        ('is_year_end', 'year_end'),
        ('is_dst_transition_week', 'dst_transition'),
        ('is_first_week_of_year', 'first_week_year'),
        ('is_event_window', 'event_window'),
    ]:
        if col_name in df.columns:
            seasonal['calendar_effects'][label] = _bucket_stats(
                (df[col_name] == 1).to_numpy(), label,
            )

    # ── 5. Time-of-day breakdown (hour bins via cyclical features) ──
    if 'time_since_london_open_min' in df.columns:
        # Use minutes since London open as proxy for time-of-day
        minutes = df['time_since_london_open_min'].to_numpy()
        seasonal['time_buckets'] = {}
        for label, lo, hi in [
            ('pre_london', 0, 0),  # mask differently
            ('london_0-2h', 1, 120),
            ('london_2-4h', 120, 240),
            ('london_4-6h', 240, 360),
            ('london_6-8h', 360, 480),
            ('london_8h+', 480, 999999),
        ]:
            if label == 'pre_london':
                ny_min = df['time_since_ny_open_min'].to_numpy()
                mask = (minutes == 0) & (ny_min == 0)
            else:
                mask = (minutes >= lo) & (minutes < hi)
            seasonal['time_buckets'][label] = _bucket_stats(mask, label)

    # ── 6. Best/Worst seasonal windows ──
    all_buckets = []
    for cat, items in seasonal.items():
        if cat in ('total_samples', 'overall_accuracy', 'overall_confidence'):
            continue
        if isinstance(items, dict):
            for sub_name, stats in items.items():
                if isinstance(stats, dict) and stats.get('accuracy') is not None and stats.get('n_samples', 0) >= 10:
                    all_buckets.append({
                        'category': cat,
                        'bucket': sub_name,
                        'accuracy': stats['accuracy'],
                        'n_samples': stats['n_samples'],
                        'confidence': stats.get('confidence'),
                    })
    all_buckets.sort(key=lambda x: x['accuracy'], reverse=True)
    seasonal['top_5_windows'] = all_buckets[:5]
    seasonal['bottom_5_windows'] = all_buckets[-5:][::-1] if len(all_buckets) >= 5 else []

    return seasonal


def generate_markdown_report(
    cluster_stats: list[dict],
    feature_importance: dict,
    overall_stats: dict,
    output_path: str,
    seasonal_stats: dict | None = None,
) -> None:
    """يولّد تقرير قابل للقراءة بالعربي + English."""
    lines = [
        '# SSL Pattern Discovery Report',
        '',
        '## ما تعلّمه النموذج من البيانات',
        '',
        f"**Total samples analyzed:** {overall_stats['n_samples']:,}",
        f"**Number of patterns discovered (clusters):** {overall_stats['n_clusters']}",
        f"**Directional rows in dataset:** {overall_stats['n_directional']:,}",
        f"**Holdout accuracy:** {overall_stats.get('accuracy', 'N/A')}",
        '',
        '---',
        '',
        '## 🔍 الأنماط المكتشفة (Discovered Patterns)',
        '',
    ]

    for stat in cluster_stats:
        cid = stat['cluster_id']
        lines.append(f'### Pattern #{cid} — {stat["dominant_regime"]} regime')
        lines.append('')
        lines.append(f'- **Size:** {stat["n_samples"]:,} samples ({stat["pct_of_total"]:.1f}% of total)')
        lines.append(f'- **Dominant regime:** `{stat["dominant_regime"]}`')
        lines.append(f'- **Dominant session phase:** `{stat["dominant_session"]}`')
        lines.append(f'- **Dominant Wyckoff phase:** `{stat["dominant_wyckoff"]}`')

        if stat['direction']:
            d = stat['direction']
            lines.append(f'- **Direction predictions:** LONG={d["long_pct"]*100:.1f}% '
                        f'SHORT={d["short_pct"]*100:.1f}%')
            if 'mean_confidence' in d:
                lines.append(f'- **Mean confidence:** {d["mean_confidence"]:.3f}')

        if stat['cycle_position']:
            lines.append(f'- **Cycle position:** mean={stat["cycle_position"]["mean"]:.3f}, '
                        f'std={stat["cycle_position"]["std"]:.3f}')

        if stat['market_metrics']:
            m = stat['market_metrics']
            metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in m.items())
            lines.append(f'- **Market metrics:** {metrics_str}')

        # Regime breakdown
        if len(stat['regime_distribution']) > 1:
            lines.append(f'')
            lines.append(f'  **Regime breakdown:**')
            for reg, pct in sorted(stat['regime_distribution'].items(), key=lambda x: -x[1]):
                lines.append(f'    - {reg}: {pct*100:.1f}%')

        # Wyckoff breakdown
        if stat['wyckoff_distribution']:
            lines.append(f'')
            lines.append(f'  **Wyckoff phase distribution:**')
            for phase, prob in sorted(stat['wyckoff_distribution'].items(), key=lambda x: -x[1]):
                lines.append(f'    - {phase}: {prob:.3f}')

        lines.append('')

    lines.extend([
        '---',
        '',
        '## 📊 Feature Importance (أي features الأكثر تأثيراً)',
        '',
        '| Feature Group | Baseline Acc | After Shuffling | Drop | Importance % |',
        '|---|---|---|---|---|',
    ])
    for group, imp in sorted(feature_importance.items(), key=lambda x: -x[1]['accuracy_drop']):
        lines.append(
            f'| `{group}` | {imp["baseline_acc"]:.3f} | {imp["perturbed_acc"]:.3f} | '
            f'{imp["accuracy_drop"]:.3f} | {imp["importance_pct"]:.1f}% |'
        )

    # ── Seasonal Patterns Section ──
    if seasonal_stats:
        lines.extend(['', '---', '', '## 🗓️ أنماط Seasonal Map (Temporal Patterns)', ''])
        lines.append(f"**Overall Accuracy:** {seasonal_stats['overall_accuracy']:.3f} | "
                    f"**Overall Confidence:** {seasonal_stats['overall_confidence']:.3f}")
        lines.append('')

        def _print_bucket_table(title: str, items: dict) -> list[str]:
            out = [f'### {title}', '',
                   '| Window | Samples | Accuracy | Confidence | LONG% | SHORT% |',
                   '|---|---|---|---|---|---|']
            for name, stats in items.items():
                if not isinstance(stats, dict) or stats.get('n_samples', 0) < 3:
                    continue
                acc = stats.get('accuracy')
                conf = stats.get('confidence')
                long_pct = stats.get('long_pct')
                short_pct = stats.get('short_pct')
                out.append(
                    f"| `{name}` | {stats['n_samples']} | "
                    f"{acc:.3f if acc is not None else 'N/A'} | "
                    f"{conf:.3f if conf is not None else 'N/A'} | "
                    f"{(long_pct or 0)*100:.1f}% | {(short_pct or 0)*100:.1f}% |"
                )
            out.append('')
            return out

        if 'session_phase' in seasonal_stats:
            lines.extend(_print_bucket_table(
                '### Session Phase (opening/middle/closing/outside)',
                seasonal_stats['session_phase'],
            ))
        if 'day_of_week' in seasonal_stats:
            lines.extend(_print_bucket_table(
                'Day of Week (Monday / Friday / Rest)',
                seasonal_stats['day_of_week'],
            ))
        if 'london_session' in seasonal_stats:
            lines.extend(_print_bucket_table(
                'London Session',
                seasonal_stats['london_session'],
            ))
        if 'ny_session' in seasonal_stats:
            lines.extend(_print_bucket_table(
                'NY Session',
                seasonal_stats['ny_session'],
            ))
        if 'time_buckets' in seasonal_stats:
            lines.extend(_print_bucket_table(
                'Time-of-Day Buckets',
                seasonal_stats['time_buckets'],
            ))
        if 'calendar_effects' in seasonal_stats:
            lines.extend(_print_bucket_table(
                'Calendar Effects (month-end / quarter-end / DST / events)',
                seasonal_stats['calendar_effects'],
            ))

        # Top/Bottom windows
        if seasonal_stats.get('top_5_windows'):
            lines.append('### 🏆 Top 5 Profitable Windows')
            lines.append('')
            lines.append('| Rank | Category | Window | Accuracy | Samples |')
            lines.append('|---|---|---|---|---|')
            for i, w in enumerate(seasonal_stats['top_5_windows'], 1):
                lines.append(f"| {i} | {w['category']} | `{w['bucket']}` | "
                           f"{w['accuracy']:.3f} | {w['n_samples']} |")
            lines.append('')

        if seasonal_stats.get('bottom_5_windows'):
            lines.append('### ⚠️ Bottom 5 Risky Windows')
            lines.append('')
            lines.append('| Rank | Category | Window | Accuracy | Samples |')
            lines.append('|---|---|---|---|---|')
            for i, w in enumerate(seasonal_stats['bottom_5_windows'], 1):
                lines.append(f"| {i} | {w['category']} | `{w['bucket']}` | "
                           f"{w['accuracy']:.3f} | {w['n_samples']} |")
            lines.append('')

    lines.extend([
        '',
        '---',
        '',
        '## 🎯 الـ Insights الرئيسية',
        '',
    ])

    # Generate insights
    insights = []
    most_important = max(feature_importance.items(), key=lambda x: x[1]['accuracy_drop']) if feature_importance else None
    if most_important:
        insights.append(
            f'1. **أكثر مجموعة features تأثيراً:** `{most_important[0]}` '
            f'(accuracy drop = {most_important[1]["accuracy_drop"]:.3f})'
        )

    largest_cluster = max(cluster_stats, key=lambda x: x['n_samples']) if cluster_stats else None
    if largest_cluster:
        insights.append(
            f'2. **أكبر pattern:** Cluster #{largest_cluster["cluster_id"]} '
            f'بـ {largest_cluster["n_samples"]:,} sample ({largest_cluster["pct_of_total"]:.1f}%) — '
            f'سائدة في `{largest_cluster["dominant_regime"]}` regime'
        )

    # Wyckoff distribution insight
    wyckoff_clusters = {}
    for stat in cluster_stats:
        wp = stat['dominant_wyckoff']
        wyckoff_clusters[wp] = wyckoff_clusters.get(wp, 0) + stat['n_samples']
    if wyckoff_clusters:
        dominant = max(wyckoff_clusters.items(), key=lambda x: x[1])
        insights.append(
            f'3. **Wyckoff phase الأكثر شيوعاً:** `{dominant[0]}` '
            f'بـ {dominant[1]:,} sample'
        )

    # Session phase insight
    session_clusters = {}
    for stat in cluster_stats:
        sp = stat['dominant_session']
        session_clusters[sp] = session_clusters.get(sp, 0) + stat['n_samples']
    if session_clusters:
        dominant = max(session_clusters.items(), key=lambda x: x[1])
        insights.append(
            f'4. **Session phase الأكثر شيوعاً:** `{dominant[0]}` '
            f'بـ {dominant[1]:,} sample'
        )

    for ins in insights:
        lines.append(f'- {ins}')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 📁 ملفات إضافية')
    lines.append('')
    lines.append('- `patterns_analysis.json` — البيانات الخام بتفصيل')
    lines.append('- `cluster_statistics.csv` — إحصائيات لكل cluster (للاستيراد في Excel)')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='day_trading_features.parquet')
    p.add_argument('--embeddings', required=True, help='embeddings.npy')
    p.add_argument('--direction-head', required=True, help='best_direction_head.pt')
    p.add_argument('--output', default='checkpoints/interpretability')
    p.add_argument('--n-clusters', type=int, default=8)
    p.add_argument('--max-samples', type=int, default=20000,
                  help='subsample if dataset too large')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"═══ Pattern Discovery & Interpretability ═══")
    print(f"Device: {device}")

    # Load
    print(f"\n📂 Loading data...")
    df = pd.read_parquet(args.features)
    embeddings = np.load(args.embeddings)
    ckpt = torch.load(args.direction_head, map_location=device, weights_only=False)
    print(f"   df: {len(df)} rows, embeddings: {embeddings.shape}")

    # Direction head
    direction_head = DirectionHead(ckpt['input_dim']).to(device)
    direction_head.load_state_dict(ckpt['model_state_dict'])
    mu, sigma = ckpt['mu'], ckpt['sigma']

    # ── Filter to directional rows ──
    bias = df['bias_label'].to_numpy()
    directional_mask = (bias == 0) | (bias == 1)
    n_directional = int(directional_mask.sum())
    print(f"   directional rows: {n_directional}")

    if n_directional < 50:
        print(f"⚠️  Too few directional rows for meaningful pattern analysis")
        return

    X = embeddings[directional_mask]
    y = bias[directional_mask]
    df_dir = df[directional_mask].reset_index(drop=True)

    # Subsample if too large
    if len(X) > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=args.max_samples, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]
        df_dir = df_dir.iloc[idx].reset_index(drop=True)
        print(f"   subsampled to {len(X)}")

    # Get direction predictions
    print(f"\n🔮 Getting direction predictions...")
    X_norm = (X - mu) / sigma
    X_t = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = direction_head(X_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
    acc = float((preds == y).mean())
    print(f"   Accuracy: {acc:.3f}")

    # ── Cluster embeddings ──
    print(f"\n🔬 K-Means clustering (n_clusters={args.n_clusters})...")
    cluster_labels, _ = kmeans_clustering(X, n_clusters=args.n_clusters)
    cluster_sizes = Counter(cluster_labels.tolist())
    print(f"   Cluster sizes: {dict(sorted(cluster_sizes.items()))}")

    # ── Analyze clusters ──
    print(f"\n📊 Analyzing cluster characteristics...")
    cluster_stats = analyze_clusters(
        X, df_dir, cluster_labels,
        direction_preds=preds, direction_probs=probs,
    )
    print(f"   Found {len(cluster_stats)} meaningful clusters (>=5 samples)")

    # ── Feature importance via perturbation ──
    print(f"\n🎯 Feature importance via perturbation...")
    # Use metadata if available
    meta_path = Path(args.embeddings).parent / 'metadata.json'
    micro_dim = 64
    macro_dim = 64
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        micro_dim = meta.get('micro_dim', 64)
        macro_dim = meta.get('macro_dim', 64)
        seasonal_cols = meta.get('seasonal_cols', [])
        cycle_struct_cols = meta.get('cycle_structural_cols', [])
        seasonal_dim = len(seasonal_cols)
        cycle_struct_dim = len(cycle_struct_cols)
    else:
        seasonal_dim = max(0, X.shape[1] - micro_dim - macro_dim - 12)
        cycle_struct_dim = X.shape[1] - micro_dim - macro_dim - seasonal_dim

    feature_groups = {
        'micro_emb (LOB Transformer)': slice(0, micro_dim),
        'macro_emb (Price Cycle)': slice(micro_dim, micro_dim + macro_dim),
        'seasonal_features': slice(micro_dim + macro_dim, micro_dim + macro_dim + seasonal_dim),
        'cycle_structural': slice(micro_dim + macro_dim + seasonal_dim, X.shape[1]),
    }
    feature_importance = feature_importance_via_perturbation(
        direction_head, X, y, feature_groups, mu, sigma, device,
    )
    for group, imp in sorted(feature_importance.items(), key=lambda x: -x[1]['accuracy_drop']):
        print(f"   {group}: drop={imp['accuracy_drop']:.3f} ({imp['importance_pct']:.1f}%)")

    # ── Generate reports ──
    print(f"\n📝 Generating reports...")
    overall_stats = {
        'n_samples': int(len(X)),
        'n_clusters': len(cluster_stats),
        'n_directional': n_directional,
        'accuracy': f'{acc:.3f}',
    }

    # ── Seasonal Pattern Analysis ──
    print(f"\n🗓️  Analyzing seasonal patterns (time/day/calendar)...")
    seasonal_stats = analyze_seasonal_patterns(
        df_dir, preds, probs, y,
    )
    print(f"   Overall accuracy: {seasonal_stats['overall_accuracy']:.3f}")
    if seasonal_stats.get('top_5_windows'):
        print(f"   Best window: {seasonal_stats['top_5_windows'][0]['bucket']} "
              f"(acc={seasonal_stats['top_5_windows'][0]['accuracy']:.3f})")

    # JSON
    json_path = Path(args.output) / 'patterns_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall': overall_stats,
            'clusters': cluster_stats,
            'feature_importance': feature_importance,
            'seasonal': seasonal_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✅ {json_path}")

    # Markdown
    md_path = Path(args.output) / 'interpretability_report.md'
    generate_markdown_report(
        cluster_stats, feature_importance, overall_stats,
        str(md_path), seasonal_stats=seasonal_stats,
    )
    print(f"   ✅ {md_path}")

    # CSV summary
    csv_path = Path(args.output) / 'cluster_statistics.csv'
    rows = []
    for s in cluster_stats:
        row = {
            'cluster_id': s['cluster_id'],
            'n_samples': s['n_samples'],
            'pct_of_total': s['pct_of_total'],
            'dominant_regime': s['dominant_regime'],
            'dominant_session': s['dominant_session'],
            'dominant_wyckoff': s['dominant_wyckoff'],
        }
        if s['direction']:
            row['long_pct'] = s['direction'].get('long_pct', 0)
            row['short_pct'] = s['direction'].get('short_pct', 0)
            row['mean_confidence'] = s['direction'].get('mean_confidence', 0)
        if s['cycle_position']:
            row['cycle_pos_mean'] = s['cycle_position']['mean']
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"   ✅ {csv_path}")

    # Print final summary
    print(f"\n{'═' * 60}")
    print(f"  PATTERN DISCOVERY COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Patterns discovered: {len(cluster_stats)}")
    print(f"  Top influential features:")
    for group, imp in sorted(feature_importance.items(), key=lambda x: -x[1]['accuracy_drop'])[:3]:
        print(f"    - {group}: {imp['importance_pct']:.1f}% importance")
    print(f"\n  → Read full report: {md_path}")


if __name__ == '__main__':
    main()
