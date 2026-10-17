FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MISAN_DB_PATH=/data/misan.db \
    MISAN_DATA_DIR=/data \
    MISAN_MODELS_DIR=/data/models \
    MISAN_API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY misan ./misan
COPY dashboard ./dashboard
COPY config ./config

RUN pip install --upgrade pip && pip install .

# Bundle the verified initial model; startup copies it into persistent storage.
COPY data/models ./initial_models

RUN mkdir -p /data/raw /data/processed /data/reports /data/models

EXPOSE 8501

CMD ["sh", "-c", "if [ ! -f /data/models/latest.json ]; then cp -a /app/initial_models/. /data/models/; fi; uvicorn misan.api:app --host 127.0.0.1 --port 8000 & exec streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true"]
