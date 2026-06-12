FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/

COPY gemini_mcp_relay /app/gemini_mcp_relay/

RUN pip install --no-cache-dir .[server]

EXPOSE 8000

ENTRYPOINT ["gemini-mcp-relay", "--host", "0.0.0.0", "--port", "8000"]