# Titanic Dataset Analysis - Assignment 2

This project builds a predictive modeling pipeline for Titanic survival, focusing on:
- Data cleaning
- Feature engineering
- Feature selection

## Project Structure

```text
dataset_analysis/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_cleaned.csv
│   ├── train_engineered.csv
│   ├── train_selected.csv
│   ├── feature_importance.csv
│   └── selected_features.txt
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   └── model_training.py
├── README.md
└── requirements.txt
```

## Approach

1. Clean the raw training dataset (`train.csv`) by handling missing values, checking consistency, capping outliers, and removing duplicates.
2. Engineer domain-informed features such as family-based features, title/deck extraction, age groups, and transformed numeric variables.
3. Select useful features using correlation filtering, Random Forest importance, and RFE.

## Data Cleaning Decisions

- **Missing values**
  - `Age`: median imputation + missing indicator
  - `Fare`: median imputation + missing indicator
  - `Embarked`: mode imputation + missing indicator
  - High-missing columns (>70%) are dropped
- **Outliers**
  - `Age` and `Fare` are capped using IQR-based clipping
- **Consistency**
  - `Sex` normalized to consistent labels (`male`, `female`)
  - duplicate rows removed

## Engineered Features

- `FamilySize = SibSp + Parch + 1`
- `IsAlone = 1 if FamilySize == 1 else 0`
- `Title` extracted from `Name`, rare titles grouped as `Rare`
- `Deck` extracted from `Cabin`
- `AgeGroup` bucketed into Child / Teen / Adult / Senior
- `FarePerPerson = Fare / FamilySize`
- Interactions: `Pclass_Fare`, `Age_Title`
- Transformations: `LogFare`, `LogAge`
- One-hot encoding for nominal categories (`Sex`, `Embarked`, `Title`, `Deck`, `AgeGroup`)
- Standardization for continuous features

## Feature Selection Strategy

- Remove highly correlated features (`|corr| > 0.9`)
- Rank features using `RandomForestClassifier` importance
- Run optional `RFE` with logistic regression
- Build final selected feature set from top-importance + RFE union

## How to Run

From `dataset_analysis/`:

```bash
pip install -r requirements.txt
python3 scripts/data_cleaning.py
python3 scripts/feature_engineering.py
python3 scripts/feature_selection.py
python3 scripts/model_training.py
```

## Notebook

Use `notebooks/Titanic_Feature_Engineering.ipynb` for exploration and visual explanation of:
- cleaning choices
- feature derivations
- transformation justifications
- feature selection insights

## Key Findings (to fill after running)

- Add your top features from `data/feature_importance.csv`
- Note which features were dropped and why
- Summarize model-ready feature set quality
