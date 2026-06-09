FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .[server]

COPY . .

EXPOSE 8000

ENV GEMINI_BASE_URL="https://generativelanguage.googleapis.com"

CMD ["uvicorn", "gemini_mcp_relay.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
