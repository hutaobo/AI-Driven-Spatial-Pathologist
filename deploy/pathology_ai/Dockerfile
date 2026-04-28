FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN pip install --upgrade pip && pip install "pydantic>=2,<3"

COPY src/pathology_ai_service /app/src/pathology_ai_service

EXPOSE 8000

CMD ["python", "-m", "pathology_ai_service", "--host", "0.0.0.0", "--port", "8000"]
