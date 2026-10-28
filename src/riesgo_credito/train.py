"""Entrena el modelo de riesgo crediticio y lo guarda en models/."""

import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from riesgo_credito.config import (
    DECISION_THRESHOLD,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from riesgo_credito.data import cargar_datos, limpiar_datos
from riesgo_credito.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    construir_features,
)


def crear_modelo() -> Pipeline:
    """Preprocesamiento + clasificador en un solo objeto."""
    preprocesador = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clasificador = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline([("preprocesador", preprocesador), ("modelo", clasificador)])


def entrenar() -> dict:
    df = limpiar_datos(cargar_datos())
    X = construir_features(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    modelo = crear_modelo()
    modelo.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_test)[:, 1]
    predicciones = (probabilidades >= DECISION_THRESHOLD).astype(int)
    metricas = {
        "registros_entrenamiento": len(X_train),
        "registros_prueba": len(X_test),
        "umbral": DECISION_THRESHOLD,
        "roc_auc": round(roc_auc_score(y_test, probabilidades), 4),
        "precision": round(precision_score(y_test, predicciones), 4),
        "recall": round(recall_score(y_test, predicciones), 4),
        "f1": round(f1_score(y_test, predicciones), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    return metricas


if __name__ == "__main__":
    print(json.dumps(entrenar(), indent=2))
