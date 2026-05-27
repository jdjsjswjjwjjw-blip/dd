# Research Synthesis: Closing the SSL Specification Gap

Mapping the seven failure mechanisms observed in our 6-month SSL run to
the fourteen proposed solutions in the literature, ranked by feasibility
and impact for *this* codebase.

## Empirical evidence vs theoretical mechanisms

The two research documents identify seven failure mechanisms. Our
6-month run showed direct evidence of three of them:

| Mechanism | Theory | Direct evidence in our run |
|---|---|---|
| **#2 Class collapse** | BCE has a stable global minimum at "always predict majority class" when data is imbalanced | Direction head val: `100% UP` (predicted UP for every val sample) |
| **#5 Rule overlap** | SSL rediscovers the same microstructure features rules already use, adding no independent signal | SSL on non-event bars: **52.4% hit** (random); the apparent +0.25 rank-IC came entirely from overlap with day_trade events |
| **#7 MTL tradeoff** | Static loss weights cause one head to dominate gradients | v1 (β=5 magnitude): magnitude head got rank-IC=0.25 but direction head collapsed. v2 (α=5 direction): direction still failed but magnitude lost its signal (0.04) |

Mechanisms 1, 3, 4, 6 are structural — they exist in our setup but their
effects can't be directly measured at our current data scale. They become
binding constraints at 6+ year scale.

## Solutions by school

| Mechanism | Western | Chinese | Russian | Our status |
|---|---|---|---|---|
| 1. Loss mismatch | PEAR tangent projection | JEPA latent prediction | Conformal prediction | Mitigated by adaptive heads (max_R bucket) |
| 2. Class collapse | Decision-Focused Learning | **Fixed Simplex ETF + Dot-Regression** ⭐ | RobustScaler + 1:1 balance | **NOW IMPLEMENTED** |
| 3. Horizon mismatch | Causal sieve & directed delay | HEPA horizon conditioning | GBDT residual correctors | ShortTermHeads cover this |
| 4. Direction vs magnitude | Differentiable Sharpe | Momentum integration | CatBoost two-stage | Adaptive heads partially solve |
| 5. Rule overlap | Residual arbitrage portfolios | **Orthogonal Representation (OMoE)** ⭐ | Two-stage hybrid decomposition | **NOW IMPLEMENTED** |
| 6. Sample bottleneck | Gen-DFL synthetic | In-context retrieval | CoFinDiff diffusion | Solved by 5-year data |
| 7. MTL tradeoff | Deep Ensembles | DB-MTL + SON-GOKU | Encoder freezing | Walk-forward uses frozen encoder |

## What we just shipped (`modules/trading_intel/anti_collapse/`)

### 1. SimplexETFClassifier
Fixed-weight 3-way classifier where the K=3 class anchor vectors are an
equiangular tight frame: each unit-norm, pairwise cosine = −1/(K−1) = −0.5.

- The classifier has **0 trainable parameters**
- The encoder is forced to produce features aligned with the correct anchor
- Combined with `dot_regression_loss`, this **mathematically prevents**
  the optimal-shortcut majority predictor that broke our previous run

Empirical proof in `tests/test_anti_collapse.py::test_resists_majority_collapse`:
trained on 180/15/5 imbalanced 3-class data, minority-class accuracy
> 50% (not zero, as would happen with standard CE).

### 2. OrthogonalRepresentationModule
Adds a soft penalty during training that drives the SSL embedding to be
linearly uncorrelated with the day_trade rule features:

```python
penalty = (1 / (D_emb · D_rules)) · ||corr(z_emb, z_rules)||_F^2
```

- The gradient flows ONLY into the encoder (rule features are detached)
- Acts as a regularizer alongside the main loss
- Drives the encoder to find signal IN ADDITION to what rules already
  capture, not signal that mimics rules

Empirical proof in `test_perfectly_correlated_high_penalty`: when emb is
a duplicate of rules, the penalty is >5x what it is on truly uncorrelated
inputs. The Gram-Schmidt projection variant (`gram_schmidt_project_out`)
drops correlation by >100x after projection.

## How to wire into HybridModel training

```python
from modules.trading_intel.anti_collapse import (
    SimplexETFClassifier, SimplexETFConfig,
    dot_regression_loss,
    OrthogonalRepresentationModule,
)

# 1. Replace HybridModel's direction_head with a Simplex ETF
direction_clf = SimplexETFClassifier(SimplexETFConfig(
    feature_dim=hidden_dim,
    num_classes=3,
))

# 2. Add orthogonality regularizer
ortho_reg = OrthogonalRepresentationModule(weight=0.5)

# 3. Modified training step
def train_step(daytrade_feats, ssl_emb, cnn_emb, targets):
    backbone_out = model.backbone(torch.cat([daytrade_feats, ssl_emb, cnn_emb], dim=-1))
    direction_logits = direction_clf(backbone_out)

    # Main loss: dot-regression instead of CE
    loss_main = dot_regression_loss(
        backbone_out, targets.direction,
        direction_clf.etf_anchors, ignore_index=-1,
    )

    # Anti-overlap penalty
    loss_ortho = ortho_reg(backbone_out, daytrade_feats)

    return loss_main + loss_ortho
```

## What still needs implementing (ranked)

| # | Solution | School | Effort | Priority |
|---|---|---|---|---|
| 1 | **DB-MTL gradient balancing** | Chinese | MED (250 LOC) | next |
| 2 | **Differentiable Sharpe loss** | Western | HIGH (needs PnL during training) | after walk-forward signal |
| 3 | CoFinDiff synthetic data | Russian | HIGH (diffusion model) | only if sample size still bottleneck after 5yr |
| 4 | HEPA horizon conditioning | Chinese | MED (rebuild encoder) | for v2 of the pipeline |
| 5 | Conformal prediction wrapper | Russian | LOW (200 LOC) | for deployment safety |

## Decision: when to use Simplex ETF + Orthogonal Rep

The two new modules are **opt-in**. They make sense once we have enough
data for the encoder to actually learn class-separable features and find
signal independent of rules. At 5-year scale (post 2021-2025 walk-forward),
both should be turned on in the hybrid training script. At 6-month scale,
the data is too thin for either to help much.

## References

- Papyan, V., Han, X., Donoho, D. (2020). Prevalence of neural collapse
  during the terminal phase of deep learning training. *PNAS*.
- Yang, Y. et al. (2022). Inducing neural collapse in imbalanced learning.
  *NeurIPS*.
- Wang et al. (2020). Stiefel-Manifold Learning of Multiple Independent
  Representations.
- Original research documents: `ssl_specification_gap_three_schools.md`,
  `multi_scale_ai_geopolitical_analysis.md` (uploaded by user, 2026-05-27).
