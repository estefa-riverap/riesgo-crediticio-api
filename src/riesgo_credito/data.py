"""Descarga, carga y limpieza del dataset de riesgo crediticio."""

import shutil
from pathlib import Path

import pandas as pd

from riesgo_credito.config import DATA_PATH

KAGGLE_DATASET = "laotse/credit-risk-dataset"
CSV_NAME = "credit_risk_dataset.csv"

EDAD_MAXIMA = 100
ANTIGUEDAD_LABORAL_MAXIMA = 60


def descargar_datos(destino: Path = DATA_PATH) -> Path:
    """Descarga el dataset desde Kaggle y lo copia en la carpeta data/."""
    import kagglehub  # dependencia de desarrollo, solo se usa para descargar

    carpeta = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(carpeta / CSV_NAME, destino)
    return destino


def cargar_datos(ruta: Path = DATA_PATH) -> pd.DataFrame:
    """Lee el CSV del dataset."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecuta: uv run python -m riesgo_credito.data"
        )
    return pd.read_csv(ruta)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Quita duplicados y registros con edad o antigüedad laboral imposibles.

    Los valores faltantes de person_emp_length y loan_int_rate se conservan:
    el modelo los maneja directamente.
    """
    limpio = df.drop_duplicates()
    limpio = limpio[limpio["person_age"] <= EDAD_MAXIMA]
    antiguedad = limpio["person_emp_length"]
    limpio = limpio[antiguedad.isna() | (antiguedad <= ANTIGUEDAD_LABORAL_MAXIMA)]
    return limpio.reset_index(drop=True)


if __name__ == "__main__":
    ruta = descargar_datos()
    print(f"Datos guardados en {ruta}")
