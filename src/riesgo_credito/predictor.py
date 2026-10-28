"""Carga del modelo y cálculo de la predicción para una solicitud."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from riesgo_credito.features import construir_features
from riesgo_credito.schemas import ResultadoRiesgo, SolicitudCredito


def cargar_modelo(ruta: Path) -> Pipeline:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta}")
    return joblib.load(ruta)


def clasificar(probabilidad: float, umbral: float) -> ResultadoRiesgo:
    """Convierte la probabilidad en una decisión según el umbral."""
    prediccion = int(probabilidad >= umbral)
    return ResultadoRiesgo(
        probabilidad_incumplimiento=round(probabilidad, 4),
        prediccion=prediccion,
        nivel_riesgo="alto" if prediccion == 1 else "bajo",
        umbral=umbral,
    )


def predecir(
    modelo: Pipeline, solicitud: SolicitudCredito, umbral: float
) -> ResultadoRiesgo:
    entrada = construir_features(pd.DataFrame([solicitud.model_dump()]))
    probabilidad = float(modelo.predict_proba(entrada)[0, 1])
    return clasificar(probabilidad, umbral)
