FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    KOKORO_BASE_DIR=/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-lock.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')"

COPY . .

RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 7860

# Allow extra time on first start for model/voice downloads from Hugging Face
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=5)" || exit 1

CMD ["kokoro-tts-web", "--host", "0.0.0.0", "--port", "7860"]
