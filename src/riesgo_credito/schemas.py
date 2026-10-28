"""Contrato de entrada y salida de la API."""

from typing import Literal

from pydantic import BaseModel, Field


class SolicitudCredito(BaseModel):
    person_age: int = Field(ge=18, le=100, description="Edad en años")
    person_income: float = Field(gt=0, description="Ingreso anual")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60, description="Años de empleo")
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0, description="Monto solicitado")
    loan_int_rate: float = Field(gt=0, le=100, description="Tasa de interés (%)")
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        description="Incumplimiento previo registrado"
    )
    cb_person_cred_hist_length: int = Field(
        ge=0, le=60, description="Años de historial crediticio"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 28,
                "person_income": 45000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 8000,
                "loan_int_rate": 11.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            }
        }
    }


class ResultadoRiesgo(BaseModel):
    probabilidad_incumplimiento: float
    prediccion: int = Field(description="1 = incumplimiento, 0 = no incumplimiento")
    nivel_riesgo: Literal["alto", "bajo"]
    umbral: float
