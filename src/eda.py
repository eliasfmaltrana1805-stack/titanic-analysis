"""Exploratory data analysis on the cleaned Titanic dataset.

Reads data/processed/titanic_clean.csv (run src/preprocess.py first) and
writes summary figures to reports/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "titanic_clean.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")


def load_processed() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run 'python src/preprocess.py' first."
        )
    return pd.read_csv(PROCESSED_PATH)


def plot_survival_by_sex(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x="sex", y="survived", ax=ax, errorbar=None)
    ax.set_title("Survival rate by sex")
    ax.set_ylabel("Survival rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "survival_by_sex.png", dpi=150)
    plt.close(fig)


def plot_survival_by_class(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x="pclass", y="survived", ax=ax, errorbar=None)
    ax.set_title("Survival rate by passenger class")
    ax.set_ylabel("Survival rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "survival_by_class.png", dpi=150)
    plt.close(fig)


def plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue="survived", multiple="stack", bins=30, ax=ax)
    ax.set_title("Age distribution by survival")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "age_distribution.png", dpi=150)
    plt.close(fig)


def plot_survival_by_family_size(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="family_size", y="survived", ax=ax, errorbar=None)
    ax.set_title("Survival rate by family size")
    ax.set_ylabel("Survival rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "survival_by_family_size.png", dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("\nOverall survival rate:", round(df["survived"].mean(), 3))
    print("\nSurvival rate by sex:")
    print(df.groupby("sex")["survived"].mean())
    print("\nSurvival rate by class:")
    print(df.groupby("pclass")["survived"].mean())


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()

    print_summary(df)
    plot_survival_by_sex(df)
    plot_survival_by_class(df)
    plot_age_distribution(df)
    plot_survival_by_family_size(df)

    print(f"\nSaved 4 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
