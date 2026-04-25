from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_title(name: object) -> str:
    if pd.isna(name):
        return "Unknown"
    text = str(name)
    if "," in text and "." in text:
        return text.split(",")[1].split(".")[0].strip()
    return "Unknown"


def map_rare_titles(title: str) -> str:
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title if title in common else "Rare"


def extract_deck(cabin: object) -> str:
    if pd.isna(cabin):
        return "Unknown"
    text = str(cabin).strip()
    return text[0] if text else "Unknown"


def build_features(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler | None = None) -> Tuple[pd.DataFrame, StandardScaler]:
    out = df.copy()

    # Family-based features
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Title and deck extraction
    out["Title"] = out["Name"].apply(extract_title).apply(map_rare_titles)
    out["Deck"] = out["Cabin"].apply(extract_deck) if "Cabin" in out.columns else "Unknown"

    # Age groups
    bins = [0, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]
    out["AgeGroup"] = pd.cut(out["Age"], bins=bins, labels=labels, right=True, include_lowest=True).astype(str)

    # Fare per person
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)

    # Optional interaction features
    out["Pclass_Fare"] = out["Pclass"] * out["Fare"]

    title_factor = out["Title"].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}).fillna(5)
    out["Age_Title"] = out["Age"] * title_factor

    # Transform skewed features
    out["LogFare"] = np.log1p(out["Fare"])
    out["LogAge"] = np.log1p(out["Age"])

    # One-hot encode nominal features
    categorical_cols = [c for c in ["Sex", "Embarked", "Title", "Deck", "AgeGroup"] if c in out.columns]
    out = pd.get_dummies(out, columns=categorical_cols, drop_first=False)

    # Keep Pclass as ordinal value
    out["Pclass"] = out["Pclass"].astype(int)

    # Standardize continuous features
    cont_cols = [c for c in ["Age", "Fare", "FarePerPerson", "LogFare", "LogAge", "Pclass_Fare", "Age_Title"] if c in out.columns]
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        out[cont_cols] = scaler.fit_transform(out[cont_cols])
    else:
        out[cont_cols] = scaler.transform(out[cont_cols])

    return out, scaler


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "train_cleaned.csv"
    output_path = root / "data" / "train_engineered.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing file: {input_path}. Run scripts/data_cleaning.py first."
        )

    df = pd.read_csv(input_path)
    engineered_df, _ = build_features(df, fit_scaler=True)
    engineered_df.to_csv(output_path, index=False)

    print("Feature engineering complete.")
    print(f"Input shape: {df.shape}")
    print(f"Output shape: {engineered_df.shape}")
    print(f"Saved engineered data to: {output_path}")


if __name__ == "__main__":
    main()
