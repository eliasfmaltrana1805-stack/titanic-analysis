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


def plot_survival_by_age_group(df: pd.DataFrame) -> None:
    order = ["Nino", "Joven", "Adulto", "Adulto mayor"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x="age_group", y="survived", order=order, ax=ax, errorbar=None)
    ax.set_title("Survival rate by age group")
    ax.set_ylabel("Survival rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "survival_by_age_group.png", dpi=150)
    plt.close(fig)


def plot_fare_by_survival(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="survived", y="fare", ax=ax)
    ax.set_title("Fare distribution by survival")
    ax.set_xlabel("Survived (0 = No, 1 = Yes)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fare_by_survival.png", dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("\n=== Analisis 1: Porcentaje general de supervivencia ===")
    print(f"{round(df['survived'].mean() * 100, 1)}% de los pasajeros sobrevivio "
          f"({df['survived'].sum()} de {len(df)}).")

    print("\n=== Analisis 2: Supervivencia por sexo ===")
    print((df.groupby("sex")["survived"].mean() * 100).round(1))

    print("\n=== Analisis 3: Supervivencia por clase de pasajero ===")
    print((df.groupby("pclass")["survived"].mean() * 100).round(1))

    print("\n=== Analisis 4: Supervivencia por grupo de edad ===")
    order = ["Nino", "Joven", "Adulto", "Adulto mayor"]
    print((df.groupby("age_group", observed=True)["survived"].mean().reindex(order) * 100).round(1))

    print("\n=== Analisis 5: Viajar solo vs. acompanado ===")
    alone_rate = df.groupby("is_alone")["survived"].mean() * 100
    print(f"Solo: {alone_rate[1]:.1f}%  |  Acompanado: {alone_rate[0]:.1f}%")

    print("\n=== Analisis 6: Tarifa pagada vs. supervivencia ===")
    fare_by_survival = df.groupby("survived")["fare"].mean().round(2)
    print(f"Tarifa promedio - No sobrevivio: {fare_by_survival[0]}  |  Sobrevivio: {fare_by_survival[1]}")
    print("Correlacion (fare, survived):", round(df["fare"].corr(df["survived"]), 3))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()

    print_summary(df)
    plot_survival_by_sex(df)
    plot_survival_by_class(df)
    plot_age_distribution(df)
    plot_survival_by_family_size(df)
    plot_survival_by_age_group(df)
    plot_fare_by_survival(df)

    print(f"\nSaved 6 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
