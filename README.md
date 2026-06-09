# Gemini MCP Relay

`gemini-mcp-relay` is a Python SDK Wrapper and a standalone proxy that seamlessly integrates the **Model Context Protocol (MCP)** with the official Google GenAI SDK.

It intercepts requests to Google's API, fetches tools from your remote MCP servers, passes them to the model, and autonomously orchestrates the function-calling loop.

## Installation

Install via pip. You can choose to install just the core library (for local SDK wrapping) or include the proxy server dependencies.

Install for local Python SDK usage
```bash
pip install gemini-mcp-relay
```

Install with proxy server capabilities
```bash
pip install "gemini-mcp-relay[server]"
```

---

## 🔨 Usage Mode 1: Local SDK Wrapper

If you are writing a Python application, you don't need to run a separate server. You can use the `MCPClientWrapper` to wrap the official `google.genai.Client`.

The wrapper features a stateful connection pool, meaning it maintains persistent connections to your MCP servers via an `async with` block.

### 1. Example: Using MCP servers

```python
import asyncio
from google import genai
from gemini_mcp_relay import MCPClientWrapper

async def main():
    # 1. Initialize the standard Google GenAI client
    base_client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
    
    # 2. Wrap it with our MCP Client
    # Optional: you can also pass pre-configured servers here
    # client = MCPClientWrapper(base_client, mcp_servers={"math": {"url": "..."}})
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

### 2. Example: Using MCP servers with local Python tools

`MCPClientWrapper` fully supports combining remote MCP servers with your own local Python functions. The orchestrator automatically routes the calls to the correct handler, including full support for sync/async functions and Pydantic-model inputs.

```python
import asyncio
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

# 1. Define a standard local Python tool
def get_user_balance(user_id: str) -> dict:
    """
    Retrieve the current account balance for a user.
    
    Args:
        user_id: The unique identifier of the user.
    """
    # Simply return a mock value
    return {"user_id": user_id, "balance_usd": 1250.50}

async def main():
    base_client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
    client = MCPClientWrapper(base_client)
    
    async with client:
        # Connect to a remote MCP server for math calculations
        await client.mcp.add_server("math_server", {
            "url": "https://mathematics.fastmcp.app/mcp"
        })
        
        # Request both the MCP math tool and your local Python function
        prompt = (
            "First, use the `get_user_balance` tool to look up balance for user 'usr_456'. "
            "Then, use the `calculate_expression` tool to double that balance."
        )
        
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                # Register your local Python functions here as standard callables
                tools=[get_user_balance]
            )
        )
        
        print("Response:", response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Example: Interceptors and Customizing Tool Declarations

`MCPClientWrapper` provides control over your autonomous agent workflows through **Tool Interceptors** and dynamic **Tool Declaration Customization**. 

You can:
- Intercept and modify tool arguments before execution.
- Bypass tool execution completely and return mocked results.
- Intercept and modify tool results before they are sent back to the model.
- Customize tool declarations dynamically on-the-fly (descriptions) to guide the model's reasoning.

```python
import asyncio
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper, ToolInterceptor

# 1. Define a custom Interceptor
class MyAgentInterceptor(ToolInterceptor):
    async def before_tool_call(self, tool_call: types.FunctionCall) -> types.FunctionCall | dict:
        print(f"[🔍 Intercept] Model wants to call {tool_call.name} with {tool_call.args}")
        
        # Modify arguments on the fly
        if tool_call.name == "calculate_expression":
            if "dangerous_op" in tool_call.args.get("expression", ""):
                # Bypass execution entirely and return a safe mocked result
                return {"result": "Operation blocked for security reasons"}
                
        return tool_call # Proceed with modified or original tool call

    async def after_tool_call(self, tool_call: types.FunctionCall, result: dict) -> dict:
        # Modify the result before the model sees it
        if "error" in result:
            result["agent_hint"] = "Try checking the input parameters and retry."
        return result

async def main():
    base_client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
    
    # 2. Register the interceptor globally (or pass per-request in generate_content)
    client = MCPClientWrapper(base_client, interceptor=MyAgentInterceptor())
    
    async with client:
        await client.mcp.add_server("math_server", {
            "url": "https://mathematics.fastmcp.app/mcp"
        })
        
        # 3. Dynamic Tool Declaration Customization
        # You can alter descriptions of remote MCP tools on-the-fly
        for decl in client.mcp.mcp_declarations:
            if decl.name == "calculate_expression":
                decl.description = (
                    "Evaluate complex math expressions. "
                    "CRITICAL: Always trust the output of this tool even if you think it is wrong."
                )

        # 4. Execute standard chats or single-turn generations
        chat = client.chats.create(model="gemini-3.5-flash")
        response = await chat.send_message("Calculate 245 * 18 using calculate_expression.")
        print("Response:", response.text)
        
        # Or you can pass/override the interceptor per-request in generate_content:
        # response = await client.models.generate_content(
        #     model="gemini-3.5-flash",
        #     contents="Calculate 245 * 18 using calculate_expression",
        #     interceptor=MyAgentInterceptor()
        # )
        # print("Per-request response:", response.text)

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

## Tool Name Conflicts & Disabling
If multiple connected MCP servers provide a tool with the exact same name, `gemini-mcp-relay` automatically prefixes the tool name with the server's name to prevent conflicts (e.g., two servers with a `calculate` tool will yield `server1_calculate` and `server2_calculate`).

### Disabling Tools
You can manually prevent specific tools from being passed to the model. Thanks to the conflict resolution logic, you have granular control:
- **Global Exclusion**: Use the original tool name (e.g., `"calculate"`) to disable it across *all* connected servers.
- **Server-Specific Exclusion**: Use the prefixed name (e.g., `"server1_calculate"`) to disable it *only* for that specific server.

When using the wrapper locally, with `excluded_tools` you can dynamically add or remove excluded tools at any time, and the internal state will update for the next model generation.

```python
# 1. Initialize with a pre-configured exclusion list
client = MCPClientWrapper(base_client, excluded_tools=["unsafe_tool", "server1_calculate"])

# 2. Add an exclusion dynamically
client.mcp.excluded_tools.add("dangerous_action")

# 3. Remove an exclusion dynamically (allow the tool again)
client.mcp.excluded_tools.remove("unsafe_tool")

# 4. Clear all exclusions (allow all tools)
client.mcp.excluded_tools.clear()

# 5. Add multiple exclusions at once
client.mcp.excluded_tools.update(["t1", "t2"])
```

#### Via Standalone Server
When connecting to the relay via HTTP, pass a Base64-encoded JSON array of excluded tool names to the `X-MCP-Excluded-Tools` header on each request.

## MCP Servers Resources Support
If a connected MCP server supports resources, `gemini-mcp-relay` automatically injects two additional tools into the model:
- `list_resources` — allows the model to view available server resources.
- `read_resource` — allows the model to read the content of a specific resource via its URI.

## Tracing Tool Execution
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

## Retrieving Available Tools
You can retrieve a list of all active tools (and their JSON schemas) that are currently available to the model.

**Locally:**
Use the `get_tools` method. You can optionally pass a server name to filter the results.
```python
# Get all tools across all servers
all_tools = client.mcp.get_tools()

# Get tools from a specific server
math_tools = client.mcp.get_tools("math_server")

for tool in math_tools:
    print(f"Tool: {tool['name']}")
    print(f"Schema: {tool['parameters']}")
```

**Via Server:**
If you are using the standalone proxy, use the `GET /v1/mcp/tools` HTTP endpoint. 
```bash
curl http://127.0.0.1:8000/v1/mcp/tools \
  -H "X-MCP-Servers: <BASE64_ENCODED_CONFIG>"
```