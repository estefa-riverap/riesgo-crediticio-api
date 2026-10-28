"""Configuración leída desde variables de entorno, con valores por defecto."""

import os
from pathlib import Path

DATA_PATH = Path(os.getenv("DATA_PATH", "data/credit_risk_dataset.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/modelo_riesgo.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "models/metricas.json"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

TARGET = "loan_status"
TEST_SIZE = 0.2
RANDOM_STATE = 42
