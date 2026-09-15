"""Documents and re-fetches the raw dataset for this project.

Source: Kaggle competition "Titanic - Machine Learning from Disaster"
https://www.kaggle.com/c/titanic/data

A copy is already committed at data/raw/train.csv so the project runs
without Kaggle credentials. To re-download it with the Kaggle CLI instead
(requires `pip install kaggle`, accepting the competition rules on Kaggle,
and a configured ~/.kaggle/kaggle.json API token):

    kaggle competitions download -c titanic -p data/raw
    unzip -o data/raw/titanic.zip -d data/raw
"""

import subprocess
import sys
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv"
KAGGLE_COMPETITION = "titanic"


def download() -> None:
    if RAW_PATH.exists():
        print(f"Dataset already present at {RAW_PATH}, skipping download.")
        return

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION,
         "-p", str(RAW_PATH.parent)],
        check=True,
    )
    subprocess.run(
        ["unzip", "-o", str(RAW_PATH.parent / "titanic.zip"), "-d", str(RAW_PATH.parent)],
        check=True,
    )
    print(f"Downloaded dataset to {RAW_PATH.parent}")


if __name__ == "__main__":
    try:
        download()
    except FileNotFoundError:
        sys.exit(
            "The 'kaggle' CLI is not installed. Install it with "
            "'pip install kaggle', accept the competition rules on Kaggle, "
            "and configure your API token, or use the dataset copy already "
            "committed in data/raw/."
        )
