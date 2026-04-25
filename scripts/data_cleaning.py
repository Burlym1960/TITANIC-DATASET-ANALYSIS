from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(path):
    """Robust data loading with file existence and validation checks."""
    # Check if file exists first
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Check if file is empty
    if os.path.getsize(path) == 0:
        raise ValueError(f"File is empty: {path}")

    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV loaded but contains no rows.")
        return df

    except pd.errors.EmptyDataError:
        raise ValueError("No columns found. File may be corrupted or not a CSV.")


def _standardize_sex(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return text


def _cap_outliers_iqr(series: pd.Series, whisker_width: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whisker_width * iqr
    upper = q3 + whisker_width * iqr
    return series.clip(lower=lower, upper=upper)


def clean_titanic_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    decisions: dict = {"dropped_columns": [], "imputations": {}, "added_indicators": []}

    # Missing values diagnostics
    missing_ratio = cleaned.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.7].index.tolist()
    if high_missing_cols:
        cleaned = cleaned.drop(columns=high_missing_cols)
        decisions["dropped_columns"] = high_missing_cols

    # Add indicator columns before imputation to preserve missingness signal
    for col in ["Age", "Embarked", "Fare", "Cabin"]:
        if col in cleaned.columns and cleaned[col].isna().any():
            indicator_col = f"{col}_was_missing"
            cleaned[indicator_col] = cleaned[col].isna().astype(int)
            decisions["added_indicators"].append(indicator_col)

    # Imputation choices
    if "Age" in cleaned.columns and cleaned["Age"].isna().any():
        age_median = cleaned["Age"].median()
        cleaned["Age"] = cleaned["Age"].fillna(age_median)
        decisions["imputations"]["Age"] = f"median={age_median:.2f}"

    if "Fare" in cleaned.columns and cleaned["Fare"].isna().any():
        fare_median = cleaned["Fare"].median()
        cleaned["Fare"] = cleaned["Fare"].fillna(fare_median)
        decisions["imputations"]["Fare"] = f"median={fare_median:.2f}"

    if "Embarked" in cleaned.columns and cleaned["Embarked"].isna().any():
        embarked_mode = cleaned["Embarked"].mode(dropna=True)
        if not embarked_mode.empty:
            cleaned["Embarked"] = cleaned["Embarked"].fillna(embarked_mode.iloc[0])
            decisions["imputations"]["Embarked"] = f"mode={embarked_mode.iloc[0]}"

    if "Sex" in cleaned.columns:
        cleaned["Sex"] = cleaned["Sex"].map(_standardize_sex)

    # Duplicate records
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    decisions["removed_duplicates"] = int(before - len(cleaned))

    # Outlier capping for numeric columns specified in assignment
    for col in ["Age", "Fare"]:
        if col in cleaned.columns and np.issubdtype(cleaned[col].dtype, np.number):
            cleaned[col] = _cap_outliers_iqr(cleaned[col])
    decisions["outlier_strategy"] = "IQR capping for Age and Fare"

    return cleaned, decisions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "train.csv"
    output_path = root / "data" / "train_cleaned.csv"

    print("Loading file from:", input_path)

    # Use robust data loading
    train_df = load_data(input_path)

    cleaned_df, decisions = clean_titanic_data(train_df)
    cleaned_df.to_csv(output_path, index=False)

    print("Data cleaning complete.")
    print(f"Input shape: {train_df.shape}")
    print(f"Output shape: {cleaned_df.shape}")
    print(f"Saved cleaned data to: {output_path}")
    print("Decisions summary:", decisions)


if __name__ == "__main__":
    main()
