# Gemini MCP Relay

`gemini-mcp-relay` is a Python SDK Wrapper and a standalone proxy that seamlessly integrates the **Model Context Protocol (MCP)** with the official Google GenAI SDK.

It intercepts requests to Google's API, fetches tools from your remote MCP servers, passes them to the model, and autonomously orchestrates the function-calling loop.

## Installation

Install via pip. You can choose to install just the core library (for local SDK wrapping) or include the proxy server dependencies.

```bash
# 1. Install for local Python SDK usage
pip install gemini-mcp-relay
```

```bash
# Install with proxy server capabilities
pip install "gemini-mcp-relay[server]"
```

---

## 🔨 Usage Mode 1: Local SDK Wrapper

If you are writing a Python application, you don't need to run a separate server. You can use the `MCPClientWrapper` to wrap the official `google.genai.Client`.

The wrapper features a stateful connection pool, meaning it maintains persistent connections to your MCP servers via an `async with` block.

### Example: Using MCP servers locally

```python
import asyncio
from google import genai
from gemini_mcp_relay import MCPClientWrapper

async def main():
    # 1. Initialize the standard Google GenAI client
    base_client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
    
    # 2. Wrap it with our MCP Client
    client = MCPClientWrapper(base_client)
    
    # 3. Open the connection pool context
    async with client:
        # Dynamically add an MCP server (connection stays open)
        await client.mcp.add_server("math_server", {
            "url": "https://mathematics.fastmcp.app/mcp"
        })
        
        # --- You can use single-turn generation ---
        # response = await client.models.generate_content(
        #     model="gemini-3.5-flash",
        #     contents="What is 125 * 456? Use the calculation tool."
        # )
        # print("Response:", response.text)
        # -------------------------------------------
        
        # --- Or you can use the standard chats module ---
        chat = client.chats.create(model="gemini-3.5-flash")
        
        print("Model is thinking...")
        response = await chat.send_message("What is 125 * 456? Use the calculate tool.")
        print("Response:", response.text)
        
        # You can dynamically add or remove servers on the fly
        await client.mcp.remove_server("math_server")
        
        await client.mcp.add_server("search_server", {
            "url": "https://mcp.exa.ai/mcp",
            "type": "sse", # explicitly tell the proxy to use SSE transport
            "headers": {"Authorization": "Bearer YOUR_TOKEN"}
        })
        
        response = await chat.send_message("Now search the web for the latest news.")
        print("Response:", response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔨 Usage Mode 2: Standalone Proxy Server

If you are building a non-Python application (e.g., Node.js, Go) or simply prefer to run the MCP Relay as a microservice, you can start the built-in FastAPI server.

### 1. Start the Server
*(Requires the `[server]` installation extra)*

```bash
gemini-mcp-relay --port 8000
```
The server is now running and fully emulates the Google API.

### 2. Connect from any Client

On the client side, replace the Google `base_url` with the proxy's address (`http://127.0.0.1:8000`) and pass your MCP configuration via the `X-MCP-Servers` HTTP header (encoded in Base64).

**Python SDK Example:**
```python
import base64
import json
from google import genai

# Define your MCP servers
mcp_config = {
    "math_server": {
        "url": "https://mathematics.fastmcp.app/mcp"
    }
}

# Encode to Base64 to pass over HTTP
mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")

# Connect to the local Proxy Server
client = genai.Client(
    api_key="YOUR_GEMINI_API_KEY",
    http_options={
        "base_url": "http://127.0.0.1:8000",
        "headers": {"x-mcp-servers": mcp_header}
    }
)

# The server handles the MCP orchestration automatically
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Calculate 15 * 8"
)
print(response.text)
```

*(Note: The server proxy operates in a stateless manner. It will establish and close connections to the MCP servers on every single HTTP request).*

---

## Additional Features

### Disabling Specific Tools
You can manually prevent specific tools from being passed to the model.

**Locally:** Pass `excluded_tools` to the wrapper.
```python
client = MCPClientWrapper(base_client, excluded_tools=["unsafe_delete_tool"])
```

**Via Server:** Pass a Base64-encoded array to the `X-MCP-Excluded-Tools` header.

### Tracing Tool Execution
Whether you are using the **Local SDK Wrapper** or connecting via the **Standalone Proxy Server**, the tool execution lifecycle remains completely transparent. Because the library executes tools autonomously on your behalf, it automatically injects the intermediate `function_call` and `function_response` steps directly into the final response (or SSE stream chunks). 

This allows your application to perfectly trace and log exactly which MCP tools were used behind the scenes.

```python
for part in response.candidates[0].content.parts:
    if part.function_call:
        print(f"[🔧 Tool Executed: {part.function_call.name}]")
    elif part.function_response:
        print(f"[✅ Tool Result]: {part.function_response.response}")
    elif part.text:
        print(part.text)
```

### Retrieving Available Tools (Server Mode Only)
If you need to retrieve a list of all available tools and their JSON schemas from the configured MCP servers via HTTP, use the `GET /v1/mcp/tools` endpoint. 

```bash
curl http://127.0.0.1:8000/v1/mcp/tools \
  -H "X-MCP-Servers: <BASE64_ENCODED_CONFIG>"
```