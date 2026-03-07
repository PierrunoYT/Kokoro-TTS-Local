FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# NOTE: requirements.txt uses unpinned dependencies for flexibility.
# For fully reproducible builds, generate a lock file:
#   pip install -r requirements.txt && pip freeze > requirements-lock.txt
# Then replace "requirements.txt" below with "requirements-lock.txt".

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')"

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs /app/voices /app/.cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Allow extra time on first start for model/voice downloads from Hugging Face
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "gradio_interface.py", "--host", "0.0.0.0", "--port", "7860"]
