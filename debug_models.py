"""Debug: Compare model predictions for a strong-vs-weak matchup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models" / "ml_classification"))

import joblib
import pandas as pd
import numpy as np

lr_b = joblib.load("models/ml_classification/saved_models/logistic_regression_with_nlp.pkl")
xgb_b = joblib.load("models/ml_classification/saved_models/xgboost_with_nlp.pkl")
rf_b = joblib.load("models/ml_classification/saved_models/random_forest_with_nlp.pkl")

# Check feature order
print("LR features:", lr_b['features'])
print()
print("XGB features:", xgb_b['features'])
print()

# Check if different
diff = [i for i, (a, b) in enumerate(zip(lr_b['features'], xgb_b['features'])) if a != b]
if diff:
    print(f"FEATURE ORDER DIFFERS at positions: {diff}")
else:
    print("Feature order IDENTICAL")
print()

# Simulate Bayern vs Gladbach
features = xgb_b['features']
fake = {f: 0.0 for f in features}
fake['elo_difference'] = 300.0
fake['form_difference'] = 1.5
fake['goals_per_game_home'] = 2.8
fake['goals_per_game_away'] = 1.2
fake['goals_conceded_per_game_home'] = 0.6
fake['goals_conceded_per_game_away'] = 1.5
fake['home_advantage_score'] = 1.0
fake['strength_ratio'] = 1.2
fake['league_position_home'] = 1.0
fake['league_position_away'] = 10.0
fake['h2h_home_wins'] = 2.0
fake['h2h_draws'] = 1.0
fake['goal_difference_delta'] = 2.0
fake['days_since_last_match_home'] = 7.0
fake['days_since_last_match_away'] = 7.0

df_fake = pd.DataFrame([fake])

for name, bundle in [("XGBoost", xgb_b), ("RandomForest", rf_b), ("LogReg", lr_b)]:
    model = bundle['model']
    mf = bundle['features']
    X = df_fake[mf].fillna(0)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = bundle['label_encoder'].inverse_transform([pred_idx])[0]
    classes = list(bundle['label_encoder'].classes_)
    probs_dict = {c: f"{p*100:.1f}%" for c, p in zip(classes, proba)}
    print(f"{name}: {probs_dict} => {pred_label}")

# Check LR scaler stats
print()
print("LR Pipeline steps:", [s[0] for s in lr_b['model'].steps])
scaler = lr_b['model'].named_steps['scaler']
print("Scaler means (key features):")
for f, m, s in zip(lr_b['features'], scaler.mean_, scaler.scale_):
    if f in ['elo_difference', 'form_difference', 'league_position_home', 'league_position_away', 'strength_ratio']:
        print(f"  {f}: mean={m:.3f}, scale={s:.3f}")

# Check LR coefficients
clf = lr_b['model'].named_steps['clf']
print()
print("LR Coefficients per class:")
for i, cls in enumerate(lr_b['label_encoder'].classes_):
    print(f"  {cls}:")
    coefs = clf.coef_[i]
    for f, c in sorted(zip(lr_b['features'], coefs), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"    {f}: {c:.4f}")
