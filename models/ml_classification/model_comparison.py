"""
Side-by-side model comparison across all trained variants.

Generates a summary table and bar chart for the ablation study.

Usage:
    python models/ml_classification/model_comparison.py
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
LABEL_ORDER = ["Away Win", "Draw", "Home Win"]


def evaluate_on_test(name: str, suffix: str) -> dict | None:
    model_path = SAVED_MODELS_DIR / f"{name}_{suffix}.pkl"
    test_path = SAVED_MODELS_DIR / f"test_split_{suffix}.csv"
    if not model_path.exists() or not test_path.exists():
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(test_path)
    y_test = df.pop("label").values
    X = df[features].fillna(df[features].median(numeric_only=True))

    y_pred = model.predict(X)
    acc = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred, average="macro")
    return {"model": name, "suffix": suffix, "accuracy": acc, "f1_macro": f1}


def main():
    rows = []
    for suffix in ["no_nlp", "with_nlp"]:
        for name in MODEL_NAMES:
            result = evaluate_on_test(name, suffix)
            if result:
                rows.append(result)

    if not rows:
        log.error("No evaluation results found. Run train.py first.")
        return

    df = pd.DataFrame(rows)
    df["label"] = df["model"].str.replace("_", " ").str.title() + "\n(" + df["suffix"] + ")"
    log.info("\n%s", df[["model", "suffix", "accuracy", "f1_macro"]].to_string(index=False))

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["accuracy", "f1_macro"]):
        pivot = df.pivot(index="model", columns="suffix", values=metric)
        pivot.plot(kind="bar", ax=ax, color=["#e07070", "#70a8e0"])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.legend(title="Features")
        ax.tick_params(axis="x", rotation=20)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)

    plt.suptitle("Ablation Study: NLP Feature Impact on Model Performance", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = SAVED_MODELS_DIR / "ablation_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Ablation chart saved to %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
