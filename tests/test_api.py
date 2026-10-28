import pytest
from fastapi.testclient import TestClient

from riesgo_credito.api import app

SOLICITUD_RIESGO_BAJO = {
    "person_age": 40,
    "person_income": 120000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 12,
    "loan_intent": "HOMEIMPROVEMENT",
    "loan_grade": "A",
    "loan_amnt": 5000,
    "loan_int_rate": 7.5,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 15,
}

SOLICITUD_RIESGO_ALTO = {
    "person_age": 23,
    "person_income": 15000,
    "person_home_ownership": "RENT",
    "person_emp_length": 0,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "E",
    "loan_amnt": 9000,
    "loan_int_rate": 18.5,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 2,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_indica_modelo_cargado(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok", "modelo_cargado": True}


def test_predict_devuelve_probabilidad_valida(client):
    respuesta = client.post("/predict", json=SOLICITUD_RIESGO_BAJO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert 0 <= cuerpo["probabilidad_incumplimiento"] <= 1
    assert cuerpo["prediccion"] in (0, 1)


def test_perfil_mas_riesgoso_tiene_mayor_probabilidad(client):
    bajo = client.post("/predict", json=SOLICITUD_RIESGO_BAJO).json()
    alto = client.post("/predict", json=SOLICITUD_RIESGO_ALTO).json()
    assert alto["probabilidad_incumplimiento"] > bajo["probabilidad_incumplimiento"]
    assert alto["nivel_riesgo"] == "alto"
    assert bajo["nivel_riesgo"] == "bajo"


@pytest.mark.parametrize(
    "cambio",
    [
        {"person_age": 15},
        {"person_income": 0},
        {"loan_grade": "Z"},
        {"loan_amnt": "mucho"},
    ],
)
def test_predict_rechaza_entradas_invalidas(client, cambio):
    respuesta = client.post("/predict", json={**SOLICITUD_RIESGO_BAJO, **cambio})
    assert respuesta.status_code == 422


def test_predict_rechaza_campos_faltantes(client):
    incompleta = {k: v for k, v in SOLICITUD_RIESGO_BAJO.items() if k != "loan_grade"}
    assert client.post("/predict", json=incompleta).status_code == 422
