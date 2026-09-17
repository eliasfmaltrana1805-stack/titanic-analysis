"""Cleans and enriches the raw Titanic dataset.

Reads data/raw/train.csv and writes data/processed/titanic_clean.csv with:
  - Age imputed with the median of its (Pclass, Sex) group (177 missing
    values out of 891 -- ~19.9%). A per-group median was preferred over a
    single global median because age varies noticeably by class and sex
    in this dataset, so it keeps the imputed values closer to reality
    than a single flat number would.
  - Embarked imputed with the overall mode (only 2 missing values).
  - Cabin dropped (687 missing out of 891, ~77% -- too sparse to impute
    reliably) but kept as a binary `has_cabin` flag, since simply having
    a recorded cabin correlates with passenger class/fare.
  - Fare imputed with the median (0 missing in train.csv, but handled for
    robustness in case this script is reused on test.csv).
  - normalized column names (snake_case)
  - a `family_size` column (SibSp + Parch + 1)
  - an `is_alone` flag (family_size == 1)
  - a `title` column extracted from the passenger's name
  - an `age_group` column (Nino / Joven / Adulto / Adulto mayor -- ver
    docstring de `clean()` para los cortes exactos usados)
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

    # Categorias de edad (criterio propio, documentado en el README):
    # Nino <=12, Joven 13-18, Adulto 19-60, Adulto mayor >60.
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 60, 100],
        labels=["Nino", "Joven", "Adulto", "Adulto mayor"],
    )

    df["sex"] = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df["pclass"] = df["pclass"].astype("category")
    df["title"] = df["title"].astype("category")

    return df


def main() -> None:
    df = load_raw()

    print(f"Raw dataset: {len(df)} rows, {len(df.columns)} columns")
    print("\nVariables disponibles:")
    print(list(df.columns))
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nDuplicated rows:", df.duplicated().sum())
    print("\nMissing values in raw data:")
    missing = df.isna().sum()
    print(missing[missing > 0])
    print("\nEstadisticas descriptivas (variables numericas):")
    print(df.describe())

    clean_df = clean(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    print(f"\nProcessed {len(clean_df)} rows -> {PROCESSED_PATH}")
    print("Missing values remaining after cleaning:")
    remaining = clean_df.isna().sum()
    print(remaining[remaining > 0] if remaining.sum() else "None")
    print("\nSurvival rate:", round(clean_df["survived"].mean(), 3))


if __name__ == "__main__":
    main()
