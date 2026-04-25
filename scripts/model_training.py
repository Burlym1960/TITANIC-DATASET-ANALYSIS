from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def evaluate_model(name: str, model, x_train, x_valid, y_train, y_valid) -> dict:
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    proba = model.predict_proba(x_valid)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_valid, preds),
        "roc_auc": roc_auc_score(y_valid, proba) if proba is not None else None,
        "classification_report": classification_report(y_valid, preds),
    }
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    selected_path = root / "data" / "train_selected.csv"
    metrics_path = root / "data" / "model_metrics.txt"

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing file: {selected_path}. Run scripts/feature_selection.py first."
        )

    df = pd.read_csv(selected_path)
    if "Survived" not in df.columns:
        raise ValueError("Expected target column 'Survived' in selected dataset.")

    x = df.drop(columns=["Survived"])
    y = df["Survived"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
    rf = RandomForestClassifier(n_estimators=400, random_state=42)

    results = [
        evaluate_model("LogisticRegression", lr, x_train, x_valid, y_train, y_valid),
        evaluate_model("RandomForestClassifier", rf, x_train, x_valid, y_train, y_valid),
    ]

    with metrics_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(f"Model: {result['model']}\n")
            handle.write(f"Accuracy: {result['accuracy']:.4f}\n")
            if result["roc_auc"] is not None:
                handle.write(f"ROC-AUC: {result['roc_auc']:.4f}\n")
            handle.write("Classification Report:\n")
            handle.write(result["classification_report"])
            handle.write("\n" + "-" * 60 + "\n")

    print("Model training complete.")
    print(f"Saved metrics report to: {metrics_path}")
    for result in results:
        roc_auc_text = (
            f"{result['roc_auc']:.4f}" if result["roc_auc"] is not None else "N/A"
        )
        print(
            f"{result['model']}: accuracy={result['accuracy']:.4f}, "
            f"roc_auc={roc_auc_text}"
        )


if __name__ == "__main__":
    main()
