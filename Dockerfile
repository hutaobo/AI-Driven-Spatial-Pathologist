FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    APP_DATA_DIR=/home/username/app/project-vol \
    GRADIO_TEMP_DIR=/tmp/gradio

LABEL org.opencontainers.image.title="Agentic Spatial Pathologist" \
      org.opencontainers.image.description="Agentic workflows for spatial pathology" \
      org.opencontainers.image.source="https://github.com/hutaobo/Agentic-Spatial-Pathologist"

WORKDIR /home/username/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .

RUN mkdir -p /home/username/app/project-vol

EXPOSE 7860

CMD ["python", "main.py"]
