FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')"

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs /app/voices /app/.cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "gradio_interface.py", "--host", "0.0.0.0", "--port", "7860"]
