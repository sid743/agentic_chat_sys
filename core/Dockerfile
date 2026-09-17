FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agentic_core ./agentic_core
COPY config ./config
COPY data ./data
COPY tests ./tests
COPY pyproject.toml .

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/var \
    && chown -R agent:agent /app
USER agent

EXPOSE 8088
VOLUME ["/app/var"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "agentic_core.main:app", "--host", "0.0.0.0", "--port", "8088"]
