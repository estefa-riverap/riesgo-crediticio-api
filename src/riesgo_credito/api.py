"""Servicio HTTP de predicción de riesgo crediticio."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from riesgo_credito.config import DECISION_THRESHOLD, MODEL_PATH
from riesgo_credito.predictor import cargar_modelo, predecir
from riesgo_credito.schemas import ResultadoRiesgo, SolicitudCredito

estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga una sola vez al iniciar el servicio
    estado["modelo"] = cargar_modelo(MODEL_PATH)
    yield
    estado.clear()


app = FastAPI(
    title="API de riesgo crediticio",
    description="Estima la probabilidad de que una solicitud de crédito "
    "termine en incumplimiento.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modelo_cargado": "modelo" in estado}


@app.post("/predict", response_model=ResultadoRiesgo)
def predict(solicitud: SolicitudCredito) -> ResultadoRiesgo:
    modelo = estado.get("modelo")
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    return predecir(modelo, solicitud, DECISION_THRESHOLD)
