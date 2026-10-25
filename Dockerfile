FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir ".[server]"

EXPOSE 8000

CMD ["python", "-m", "gemini_mcp_relay.server.main", "--host", "0.0.0.0", "--port", "8000"]
