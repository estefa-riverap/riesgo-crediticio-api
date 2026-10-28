import pandas as pd
import pytest

from riesgo_credito.data import limpiar_datos
from riesgo_credito.features import (
    FEATURES,
    calcular_porcentaje_ingreso,
    construir_features,
)
from riesgo_credito.predictor import clasificar


def test_porcentaje_ingreso_se_calcula_desde_monto_e_ingreso():
    assert calcular_porcentaje_ingreso(10000, 50000) == 0.2


def test_porcentaje_ingreso_rechaza_ingreso_cero():
    with pytest.raises(ValueError):
        calcular_porcentaje_ingreso(10000, 0)


def test_construir_features_devuelve_columnas_del_modelo():
    df = pd.DataFrame(
        [
            {
                "person_age": 30,
                "person_income": 40000,
                "person_home_ownership": "RENT",
                "person_emp_length": 2,
                "loan_intent": "MEDICAL",
                "loan_grade": "C",
                "loan_amnt": 6000,
                "loan_int_rate": 13.0,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3,
            }
        ]
    )
    resultado = construir_features(df)
    assert list(resultado.columns) == FEATURES
    assert resultado.loc[0, "loan_percent_income"] == 0.15


def test_limpiar_datos_quita_duplicados_y_valores_imposibles():
    df = pd.DataFrame(
        {
            "person_age": [25, 25, 144, 40, 35],
            "person_emp_length": [2.0, 2.0, 5.0, 123.0, None],
        }
    )
    limpio = limpiar_datos(df)
    # queda una fila de los duplicados y la fila con antigüedad faltante
    assert len(limpio) == 2
    assert limpio["person_age"].max() <= 100


@pytest.mark.parametrize(
    ("probabilidad", "esperado"), [(0.49, "bajo"), (0.5, "alto"), (0.9, "alto")]
)
def test_clasificar_respeta_el_umbral(probabilidad, esperado):
    assert clasificar(probabilidad, umbral=0.5).nivel_riesgo == esperado
