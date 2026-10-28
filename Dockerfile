FROM python:3.12-slim

# uv se copia desde su imagen oficial, con versión fija
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# 1. Dependencias de ejecución (sin el grupo dev: pytest, ruff, kagglehub)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Código y modelo entrenado
COPY src ./src
COPY models/modelo_riesgo.joblib ./models/modelo_riesgo.joblib
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/models/modelo_riesgo.joblib \
    DECISION_THRESHOLD=0.5

# Usuario sin privilegios
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "riesgo_credito.api:app", "--host", "0.0.0.0", "--port", "8000"]
