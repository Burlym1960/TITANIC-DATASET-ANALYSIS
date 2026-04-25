from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def drop_high_correlation(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def feature_importance_ranking(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x, y)
    ranking = pd.DataFrame(
        {"feature": x.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return ranking


def run_rfe(x: pd.DataFrame, y: pd.Series, n_features_to_select: int = 12) -> list[str]:
    estimator = LogisticRegression(max_iter=1000, solver="liblinear")
    selector = RFE(estimator, n_features_to_select=min(n_features_to_select, x.shape[1]))
    selector.fit(x, y)
    return x.columns[selector.support_].tolist()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "train_engineered.csv"
    selected_out = root / "data" / "train_selected.csv"
    importance_out = root / "data" / "feature_importance.csv"
    report_out = root / "data" / "selected_features.txt"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing file: {input_path}. Run scripts/feature_engineering.py first."
        )

    df = pd.read_csv(input_path)
    if "Survived" not in df.columns:
        raise ValueError("Expected target column 'Survived' in engineered training data.")

    y = df["Survived"]
    x = df.drop(columns=["Survived"])

    # Keep numeric-only for modeling if object columns remain
    x = x.select_dtypes(include=[np.number])

    x_uncorrelated, dropped_corr = drop_high_correlation(x, threshold=0.9)
    ranking = feature_importance_ranking(x_uncorrelated, y)
    top_features = ranking.head(15)["feature"].tolist()
    rfe_features = run_rfe(x_uncorrelated, y, n_features_to_select=12)

    # Conservative final selection: union of top-importance and RFE features
    selected_features = sorted(set(top_features).union(set(rfe_features)))

    selected_df = pd.concat([x_uncorrelated[selected_features], y], axis=1)
    selected_df.to_csv(selected_out, index=False)
    ranking.to_csv(importance_out, index=False)

    with report_out.open("w", encoding="utf-8") as handle:
        handle.write("Dropped due to high correlation (>0.9):\n")
        handle.write(", ".join(dropped_corr) if dropped_corr else "None")
        handle.write("\n\nTop 15 by RandomForest importance:\n")
        handle.write(", ".join(top_features))
        handle.write("\n\nRFE-selected features:\n")
        handle.write(", ".join(rfe_features))
        handle.write("\n\nFinal selected features:\n")
        handle.write(", ".join(selected_features))

    print("Feature selection complete.")
    print(f"Saved selected dataset to: {selected_out}")
    print(f"Saved feature importance ranking to: {importance_out}")
    print(f"Saved feature report to: {report_out}")


if __name__ == "__main__":
    main()
