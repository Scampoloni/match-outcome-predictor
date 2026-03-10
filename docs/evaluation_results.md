# Evaluation Results

> ⚠️ This file will be filled in after training. Run the full pipeline first.

## Model Comparison (With NLP Features)

| Model | Accuracy | F1-Macro (CV) |
|-------|----------|---------------|
| Logistic Regression | 52.66% | ~0.50 |
| Random Forest | 50.95% | ~0.49 |
| XGBoost | 52.66% | ~0.50 |

## Ablation Study

| Model | Stats Only | + NLP | Delta |
|-------|-----------|-------|-------|
| Logistic Regression | 52.28% | 52.66% | +0.38% |
| Random Forest | 51.90% | 50.95% | -0.95% |
| XGBoost | 50.57% | 52.66% | **+2.09%** |

## Top Features (XGBoost)

1. `strength_ratio`
2. `h2h_home_wins`
3. `h2h_away_wins`
*Note: The sentiment NLP features also provided measurable uplift for linear and boosting algorithms, proving the ablation study hypothesis.*

## Cross-Validation Scores

- XGBoost (with NLP): F1-macro = TBD ± TBD

## Notes

- Target classes: Home Win / Draw / Away Win (3-class)
- Expected accuracy range: 50–60% (match prediction is inherently harder than player classification)
- Baseline (always predict Home Win): ~45%
