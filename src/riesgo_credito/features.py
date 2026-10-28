"""Definición de las variables que usa el modelo.

Estas funciones se usan tanto en el entrenamiento como en la inferencia,
para que ambos procesos transformen los datos de la misma manera.
"""

import pandas as pd

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def calcular_porcentaje_ingreso(loan_amnt: float, person_income: float) -> float:
    """Proporción del ingreso anual que representa el monto del préstamo."""
    if person_income <= 0:
        raise ValueError("El ingreso debe ser mayor que cero")
    return round(loan_amnt / person_income, 2)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula loan_percent_income y devuelve las columnas en el orden del modelo.

    loan_percent_income se recalcula siempre a partir del monto y el ingreso,
    así el usuario de la API no tiene que enviarlo ni puede enviarlo inconsistente.
    """
    salida = df.copy()
    salida["loan_percent_income"] = [
        calcular_porcentaje_ingreso(monto, ingreso)
        for monto, ingreso in zip(
            salida["loan_amnt"], salida["person_income"], strict=True
        )
    ]
    return salida[FEATURES]
