# Gemini MCP Relay

A FastAPI proxy server for integrating the Model Context Protocol (MCP) with the Google Gemini API.

This server acts as a bridge between the client application and Google's servers. It intercepts the official Google request format, fetches tools from remote MCP servers, converts them into the Gemini format, and autonomously orchestrates the `function_calling` execution loop, streaming the final result back to the client.

## Installation and Setup

### Method 1. Locally (Python venv)
1. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```
The server will start at `http://0.0.0.0:8000`.

### Method 2. Via Docker (Recommended for production)
The proxy is fully ready for deployment in a Docker container.

1. Build the image:
```bash
docker build -t gemini-mcp-relay .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --name mcp-relay gemini-mcp-relay
```

## Configuration (Environment Variables)

When starting the server (both locally and in Docker), you can override the base URL to which the proxy will send requests.

To do this, set the `GEMINI_BASE_URL` environment variable:

**Docker example:**
```bash
docker run -d -p 8000:8000 -e GEMINI_BASE_URL="https://generativelanguage.googleapis.com" gemini-mcp-relay
```
*(If left empty, the proxy will automatically route requests to the standard Google API servers).*

## Usage from a Client Application

The server fully emulates the Google API. All you need to do on the client side is replace the `base_url` with the proxy's address and pass a Base64-encoded `X-MCP-Servers` header.

Python example (using the official `google.genai` SDK):

```python
import base64
import json
from google import genai

# 1. Build the MCP servers configuration
# You can connect multiple servers. 
# For private servers, you can pass an optional 'headers' dictionary with authorization tokens.
mcp_config = {
    "math_server": {
        "url": "https://mathematics.fastmcp.app/mcp"
    },
    "private_database": {
        "url": "https://api.mycompany.com/mcp",
        "headers": {
            "Authorization": "Bearer YOUR_SECRET_TOKEN"
        }
    }
}
mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")

# 2. Connect to our proxy
client = genai.Client(
    api_key="YOUR_GEMINI_API_KEY", # The key will be sent to the proxy, and the proxy will pass it to Google
    http_options={
        "base_url": "http://127.0.0.1:8000",
        "headers": {"x-mcp-servers": mcp_header}
    }
)

# 3. Make a standard request (The proxy will fetch tools from MCP, execute them in a loop, and return the response)
response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Calculate the square root of 144, and then multiply the result by 10. Write down the steps.",
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```