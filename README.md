# API de riesgo crediticio

Servicio que estima la probabilidad de que una solicitud de crédito termine en incumplimiento (`loan_status = 1`).

**Usuario:** un analista de crédito que, al revisar una solicitud, quiere una estimación rápida del riesgo del solicitante como apoyo para su decisión. La API no aprueba ni rechaza créditos, entrega una probabilidad y un nivel de riesgo.

## Qué hace

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Indica si el servicio está activo y si el modelo se cargó |
| `/predict` | POST | Recibe los datos de una solicitud y devuelve la probabilidad de incumplimiento |
| `/docs` | GET | Documentación interactiva para probar la API desde el navegador |

Ejemplo de entrada para `/predict`:

```json
{
  "person_age": 28,
  "person_income": 45000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 8000,
  "loan_int_rate": 11.5,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}
```

Respuesta:

```json
{
  "probabilidad_incumplimiento": 0.0715,
  "prediccion": 0,
  "nivel_riesgo": "bajo",
  "umbral": 0.5
}
```

`loan_percent_income` no se envía: la API lo calcula como `loan_amnt / person_income`, igual que en el entrenamiento.

## Estructura

```
├── src/riesgo_credito/
│   ├── config.py      # configuración desde variables de entorno
│   ├── data.py        # descarga, carga y limpieza de datos
│   ├── features.py    # variables del modelo (compartidas por entrenamiento e inferencia)
│   ├── train.py       # entrenamiento y evaluación
│   ├── schemas.py     # contrato de entrada y salida de la API
│   ├── predictor.py   # carga del modelo y predicción
│   └── api.py         # servicio FastAPI
├── tests/             # pruebas con pytest
├── models/            # modelo entrenado y métricas
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Docker (para ejecutar el contenedor)
- cloudflared (para la publicación temporal)

## Ejecutar desde cero

```bash
git clone <url-del-repositorio>
cd riesgo-crediticio-api
uv sync
```

`uv sync` instala la versión de Python indicada en `.python-version` y las versiones exactas registradas en `uv.lock`.

El modelo entrenado ya está en `models/`, así que la API se puede levantar directamente:

```bash
uv run uvicorn riesgo_credito.api:app --port 8000
```

Luego abrir `http://localhost:8000/docs`.

### Volver a entrenar (opcional)

Los datos no se versionan. Para descargarlos desde Kaggle ([laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)) y entrenar:

```bash
uv run python -m riesgo_credito.data
uv run python -m riesgo_credito.train
```

El entrenamiento limpia los datos (quita duplicados, edades mayores a 100 y antigüedad laboral mayor a 60 años), separa 80 % entrenamiento y 20 % prueba, y guarda el modelo y sus métricas en `models/`.

## Configuración y secretos

La aplicación no usa credenciales ni contraseñas. La configuración se hace con variables de entorno, todas con valor por defecto:

| Variable | Por defecto | Uso |
|---|---|---|
| `MODEL_PATH` | `models/modelo_riesgo.joblib` | Ruta del modelo |
| `DECISION_THRESHOLD` | `0.5` | Probabilidad a partir de la cual el riesgo es "alto" |
| `DATA_PATH` | `data/credit_risk_dataset.csv` | Ruta de los datos para entrenar |

Para cambiarlas, copiar `.env.example` como `.env` y ajustar los valores. El archivo `.env` está en `.gitignore` y no se sube al repositorio.

```bash
uv run --env-file .env uvicorn riesgo_credito.api:app --port 8000
```

Si en el futuro se necesita un secreto (por ejemplo, una credencial de Kaggle), debe ir en `.env` o como variable de entorno al ejecutar el contenedor, nunca en el código ni en el Dockerfile.

## Verificación

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Las pruebas cubren el cálculo de `loan_percent_income`, la limpieza de datos, la regla del umbral, el endpoint `/health`, la respuesta de `/predict`, que un perfil más riesgoso reciba mayor probabilidad y el rechazo de entradas inválidas o incompletas.

## Contenedor

```bash
docker build -t riesgo-credito-api .
docker run --rm -p 8000:8000 riesgo-credito-api
```

La imagen solo incluye las dependencias de ejecución (`uv sync --no-dev`), el código de `src/` y el modelo. No incluye datos, pruebas, notebooks ni herramientas de desarrollo.

Para cambiar el umbral al ejecutar el contenedor:

```bash
docker run --rm -p 8000:8000 -e DECISION_THRESHOLD=0.4 riesgo-credito-api
```

## Publicación temporal

Con el contenedor en ejecución:

```bash
cloudflared tunnel --url http://localhost:8000
```

cloudflared muestra una URL pública `https://<nombre>.trycloudflare.com`. Abriendo `<url>/docs` desde otro dispositivo se puede probar `/predict`. La URL deja de funcionar al cerrar el túnel.

## Métricas del modelo

Modelo: `HistGradientBoostingClassifier` de scikit-learn con codificación one-hot de las variables categóricas. Evaluado sobre el 20 % de prueba (6.482 registros):

| Métrica | Valor |
|---|---|
| ROC AUC | 0,949 |
| Precisión (clase incumplimiento) | 0,979 |
| Recall (clase incumplimiento) | 0,738 |
| F1 | 0,841 |

Con umbral 0,5 el modelo detecta cerca de 74 % de los incumplimientos. Bajar el umbral aumenta el recall a costa de más falsas alarmas.

`scikit-learn` tiene versión fija en `pyproject.toml` porque el modelo guardado debe cargarse con la misma versión con la que se entrenó.
