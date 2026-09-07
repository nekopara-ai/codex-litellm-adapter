FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CODEX_ADAPTER_HOST=0.0.0.0 \
    CODEX_ADAPTER_REPLAY_DB=/state/replay.sqlite3

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir . \
    && useradd --system --uid 10001 --create-home adapter \
    && mkdir /state \
    && chown adapter /state

USER 10001
EXPOSE 4001
HEALTHCHECK --interval=30s --timeout=8s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4001/health', timeout=6).read()"
CMD ["codex-litellm-adapter"]
