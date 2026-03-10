"""
Model evaluation script.

Loads trained models and evaluates them on the held-out test split.
Outputs classification reports, confusion matrices, and feature importance plots.

Usage:
    python models/ml_classification/evaluate.py
    python models/ml_classification/evaluate.py --suffix no_nlp
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
LABEL_ORDER = ["Away Win", "Draw", "Home Win"]


def load_test_split(suffix: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = SAVED_MODELS_DIR / f"test_split_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}. Run train.py first.")
    df = pd.read_csv(path)
    y = df.pop("label").values
    return df, y


def evaluate_model(name: str, suffix: str, X_test: pd.DataFrame, y_test: np.ndarray):
    model_path = SAVED_MODELS_DIR / f"{name}_{suffix}.pkl"
    if not model_path.exists():
        log.warning("Model not found: %s", model_path)
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    X = X_test[features].fillna(X_test[features].median(numeric_only=True))
    y_pred = model.predict(X)

    report = classification_report(y_test, y_pred, target_names=LABEL_ORDER, output_dict=True)
    log.info("\n%s [%s]\n%s", name, suffix, classification_report(y_test, y_pred, target_names=LABEL_ORDER))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name} ({suffix})")
    plt.tight_layout()
    fig.savefig(SAVED_MODELS_DIR / f"confusion_{name}_{suffix}.png", dpi=150)
    plt.close(fig)

    # Feature importance (tree-based models only)
    try:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            importances.head(20).plot(kind="bar", ax=ax2, color="steelblue")
            ax2.set_title(f"Feature Importance — {name} ({suffix})")
            ax2.set_ylabel("Importance")
            plt.tight_layout()
            fig2.savefig(SAVED_MODELS_DIR / f"importance_{name}_{suffix}.png", dpi=150)
            plt.close(fig2)
        elif hasattr(model, "named_steps") and "clf" in model.named_steps:
             clf = model.named_steps["clf"]
             if hasattr(clf, "feature_importances_"):
                 # Handle pipeline wrapper if present
                 pass
    except Exception as e:
        log.warning(f"Could not compute feature importance for {name}: {e}")

    return report


def print_ablation_summary(reports_with: dict, reports_without: dict):
    """Print a side-by-side ablation comparison table."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY: NLP Feature Impact")
    print("=" * 60)
    print(f"{'Model':<25} {'No NLP':>10} {'With NLP':>10} {'Delta':>10}")
    print("-" * 60)
    for name in reports_with:
        if name not in reports_without:
            continue
        acc_with = reports_with[name]["accuracy"]
        acc_without = reports_without[name]["accuracy"]
        delta = acc_with - acc_without
        print(f"{name:<25} {acc_without:>10.4f} {acc_with:>10.4f} {delta:>+10.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classifiers.")
    parser.add_argument("--suffix", default=None, help="Evaluate a specific suffix (with_nlp or no_nlp)")
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = ["logistic_regression", "random_forest", "xgboost"]

    if args.suffix:
        suffixes = [args.suffix]
    else:
        suffixes = ["with_nlp", "no_nlp"]

    all_reports: dict[str, dict] = {}
    for suffix in suffixes:
        try:
            X_test, y_test = load_test_split(suffix)
        except FileNotFoundError as e:
            log.error(e)
            continue
        all_reports[suffix] = {}
        for name in model_names:
            report = evaluate_model(name, suffix, X_test, y_test)
            if report:
                all_reports[suffix][name] = report

    if "with_nlp" in all_reports and "no_nlp" in all_reports:
        print_ablation_summary(all_reports["with_nlp"], all_reports["no_nlp"])


if __name__ == "__main__":
    main()
