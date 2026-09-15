"""Cleans and enriches the raw Titanic dataset.

Reads data/raw/train.csv and writes data/processed/titanic_clean.csv with:
  - Age imputed with the median (per Pclass/Sex group)
  - Embarked imputed with the mode
  - Cabin dropped (too many missing values) but kept as `has_cabin` flag
  - normalized column names (snake_case)
  - a `family_size` column (SibSp + Parch + 1)
  - an `is_alone` flag (family_size == 1)
  - a `title` column extracted from the passenger's name
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "train.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "titanic_clean.csv"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]+)\.", name)
    title = match.group(1).strip() if match else "Unknown"
    rare = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
            "Sir", "Jonkheer", "Dona"}
    if title in rare:
        return "Rare"
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    df["has_cabin"] = df["cabin"].notna().astype(int)
    df = df.drop(columns=["cabin"])

    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda s: s.fillna(s.median())
    )

    df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])

    df["fare"] = df["fare"].fillna(df["fare"].median())

    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["title"] = df["name"].apply(extract_title)

    df["sex"] = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df["pclass"] = df["pclass"].astype("category")
    df["title"] = df["title"].astype("category")

    return df


def main() -> None:
    df = load_raw()
    clean_df = clean(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    print(f"Processed {len(clean_df)} rows -> {PROCESSED_PATH}")
    print("Missing values remaining:")
    print(clean_df.isna().sum()[clean_df.isna().sum() > 0])
    print("\nSurvival rate:", round(clean_df["survived"].mean(), 3))


if __name__ == "__main__":
    main()
