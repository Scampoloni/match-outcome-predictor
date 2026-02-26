# Evaluation Results

> ⚠️ This file will be filled in after training. Run the full pipeline first.

## Model Comparison (With NLP Features)

| Model | Accuracy | Precision | Recall | F1-Macro |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |

## Ablation Study

| Model | Stats Only | + NLP | Delta |
|-------|-----------|-------|-------|
| Logistic Regression | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | **+X%** |

## Top Features (XGBoost)

1. TBD (likely elo_difference)
2. TBD (likely form_difference)
3. TBD (possibly sentiment_gap — NLP)
...

## Cross-Validation Scores

- XGBoost (with NLP): F1-macro = TBD ± TBD

## Notes

- Target classes: Home Win / Draw / Away Win (3-class)
- Expected accuracy range: 50–60% (match prediction is inherently harder than player classification)
- Baseline (always predict Home Win): ~45%
