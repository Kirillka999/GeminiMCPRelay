import json
import base64
import logging
from contextlib import AsyncExitStack
import httpx
from fastapi import HTTPException
from google.genai import types

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)

class MCPServerAdapter:
    def __init__(self, name: str, session: ClientSession):
        self.name = name
        self.session = session
        self.tool_mappings = {}  # mapped_name -> original_mcp_name

    def register_tool(self, original_name: str, mapped_name: str):
        self.tool_mappings[mapped_name] = original_name

    async def process_function_calls_as_parts(self, calls: list) -> list:
        parts = []
        for call in calls:
            gemini_name = call.name
            mcp_name = self.tool_mappings.get(gemini_name, gemini_name)
            
            try:
                result = await self.session.call_tool(mcp_name, call.args)
                
                # Check for structuredContent or join text content
                if getattr(result, "structuredContent", None):
                    final_val = result.structuredContent
                elif result.content:
                    texts = [c.text for c in result.content if getattr(c, "type", "") == "text" and getattr(c, "text", "")]
                    if texts:
                        combined_text = "\n".join(texts)
                        try:
                            final_val = json.loads(combined_text)
                        except json.JSONDecodeError:
                            final_val = combined_text
                    else:
                        final_val = "Empty or non-text response"
                else:
                    final_val = "Empty response"

                response_key = "error" if result.isError else "result"
                
                part = types.Part.from_function_response(
                    name=gemini_name,
                    response={response_key: final_val}
                )
                part.function_response.id = call.id
                parts.append(part)
                
            except Exception as e:
                logger.error(f"Error calling tool {mcp_name} on {self.name}: {e}", exc_info=True)
                part = types.Part.from_function_response(
                    name=gemini_name,
                    response={"error": str(e)}
                )
                part.function_response.id = call.id
                parts.append(part)
                
        return parts


class MCPConnectionManager:
    def __init__(self, mcp_header: str, excluded_tools_header: str = None):
        self.mcp_header = mcp_header
        self.excluded_tools_header = excluded_tools_header
        self.stack = AsyncExitStack()
        self.adapters_map = {}
        self.mcp_declarations = []
        self.raw_tools = []
        self._connected = False

    async def connect(self, fetch_raw_tools=False):
        if self._connected:
            return
            
        if not self.mcp_header:
            return
            
        excluded_tools = set()
        if self.excluded_tools_header:
            try:
                decoded = base64.b64decode(self.excluded_tools_header).decode("utf-8")
                excluded_tools = set(json.loads(decoded))
            except Exception as e:
                logger.warning(f"Failed to parse excluded tools header: {e}")
                
        try:
            decoded_json = base64.b64decode(self.mcp_header).decode("utf-8")
            connections = json.loads(decoded_json)
        except Exception as e:
            logger.error(f"Failed to parse X-MCP-Servers header: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse X-MCP-Servers: {str(e)}")

        # Keep track of global names to avoid collisions
        global_tool_names = set()

        for name, config in connections.items():
            if config.get("command") or config.get("transport") == "stdio":
                raise HTTPException(status_code=400, detail=f"Transport 'stdio' is not allowed for server '{name}'.")
                
            url = config.get("url")
            httpUrl = config.get("httpUrl")
            mcp_type = config.get("type")
            headers = config.get("headers", {})
            
            transport_ctx = None
            
            # Transport Selection Logic matching gemini-cli
            if httpUrl:
                logger.warning(f"MCP server '{name}': 'httpUrl' is deprecated. Please migrate to 'url' with 'type: \"http\"'.")
                transport_ctx = streamable_http_client(url=httpUrl, http_client=httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0)))
            elif url and mcp_type == "http":
                transport_ctx = streamable_http_client(url=url, http_client=httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0)))
            elif url and mcp_type == "sse":
                transport_ctx = sse_client(url=url, headers=headers)
            elif url:
                # Default to streamable_http_client
                transport_ctx = streamable_http_client(url=url, http_client=httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0)))
            else:
                continue

            try:
                # Initialize connection
                streams = await self.stack.enter_async_context(transport_ctx)
                if len(streams) == 3:
                    read_stream, write_stream, _ = streams
                else:
                    read_stream, write_stream = streams
                session = await self.stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                
                adapter = MCPServerAdapter(name, session)
                
                # Fetch tools
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    if tool.name in excluded_tools:
                        continue
                        
                    # Handle name collisions
                    mapped_name = tool.name
                    if mapped_name in global_tool_names:
                        mapped_name = f"{name}_{tool.name}"
                    
                    global_tool_names.add(mapped_name)
                    adapter.register_tool(tool.name, mapped_name)
                    self.adapters_map[mapped_name] = adapter
                    
                    # Prepare Gemini declaration
                    # Input schema may lack type="object", ensure it has it to appease Gemini SDK
                    input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else tool.input_schema
                    if "type" not in input_schema:
                        input_schema["type"] = "object"
                        
                    decl = types.FunctionDeclaration(
                        name=mapped_name,
                        description=tool.description or "",
                        parameters_json_schema=input_schema
                    )
                    self.mcp_declarations.append(decl)
                    
                    if fetch_raw_tools:
                        self.raw_tools.append({
                            "serverName": name,
                            "name": mapped_name,
                            "description": tool.description or "",
                            "parameters": input_schema
                        })
                        
            except Exception as e:
                logger.error(f"Failed to connect or fetch tools from MCP server '{name}': {e}", exc_info=True)
                raise HTTPException(status_code=502, detail=f"Failed to connect to MCP server '{name}': {str(e)}")
                
        self._connected = True

    async def close(self):
        await self.stack.aclose()


# Backward compatibility wrappers for non-streaming endpoints
async def get_mcp_adapters_and_tools(mcp_header: str, excluded_tools_header: str = None) -> tuple[dict, list[types.FunctionDeclaration]]:
    """
    Deprecated: Used only if keeping old interface. We will remove this 
    in api.py and use MCPConnectionManager directly.
    """
    manager = MCPConnectionManager(mcp_header, excluded_tools_header)
    await manager.connect()
    # Note: caller is responsible for calling manager.close() !
    return manager.adapters_map, manager.mcp_declarations

async def get_mcp_raw_tools(mcp_header: str) -> list[dict]:
    manager = MCPConnectionManager(mcp_header)
    try:
        await manager.connect(fetch_raw_tools=True)
        return manager.raw_tools
    finally:
        await manager.close()
