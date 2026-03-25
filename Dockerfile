FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    APP_DATA_DIR=/home/username/app/project-vol \
    GRADIO_TEMP_DIR=/tmp/gradio

LABEL org.opencontainers.image.title="AI Driven Spatial Pathologist" \
      org.opencontainers.image.description="SciLifeLab Serve deployment wrapper for HistoSeg" \
      org.opencontainers.image.source="https://github.com/hutaobo/AI-Driven-Spatial-Pathologist"

WORKDIR /home/username/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .

RUN mkdir -p /home/username/app/project-vol /tmp/gradio

EXPOSE 7860

CMD ["python", "main.py"]
