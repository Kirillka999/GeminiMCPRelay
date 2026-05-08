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
                final_val = self._extract_result_content(result)
                response_key = "error" if result.isError else "result"
                
                part = types.Part.from_function_response(
                    name=gemini_name,
                    response={response_key: final_val}
                )
                
            except Exception as e:
                logger.error(f"Error calling tool {mcp_name} on {self.name}: {e}", exc_info=True)
                part = types.Part.from_function_response(
                    name=gemini_name,
                    response={"error": str(e)}
                )
                
            part.function_response.id = call.id
            parts.append(part)
                
        return parts

    def _extract_result_content(self, result):
        if getattr(result, "structuredContent", None):
            return result.structuredContent
            
        if not result.content:
            return "Empty response"

        texts = [c.text for c in result.content if getattr(c, "type", "") == "text" and getattr(c, "text", "")]
        if not texts:
            return "Empty or non-text response"

        combined_text = "\n".join(texts)
        try:
            return json.loads(combined_text)
        except json.JSONDecodeError:
            return combined_text


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
        if self._connected or not self.mcp_header:
            return
            
        excluded_tools = self._parse_excluded_tools()
        connections = self._parse_connections()

        # Keep track of global names to avoid collisions
        global_tool_names = set()

        for name, config in connections.items():
            if config.get("command") or config.get("transport") == "stdio":
                raise HTTPException(status_code=400, detail=f"Transport 'stdio' is not allowed for server '{name}'.")
                
            transport_ctx = self._create_transport_context(name, config)
            if not transport_ctx:
                continue

            try:
                streams = await self.stack.enter_async_context(transport_ctx)
                read_stream, write_stream = streams[:2] if len(streams) >= 2 else streams
                
                session = await self.stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                
                adapter = MCPServerAdapter(name, session)
                await self._register_server_tools(
                    adapter, 
                    session, 
                    name, 
                    excluded_tools, 
                    global_tool_names, 
                    fetch_raw_tools
                )
                        
            except Exception as e:
                logger.error(f"Failed to connect or fetch tools from MCP server '{name}': {e}", exc_info=True)
                raise HTTPException(status_code=502, detail=f"Failed to connect to MCP server '{name}': {str(e)}")
                
        self._connected = True

    async def close(self):
        await self.stack.aclose()

    def _parse_excluded_tools(self) -> set:
        if not self.excluded_tools_header:
            return set()
        try:
            decoded = base64.b64decode(self.excluded_tools_header).decode("utf-8")
            return set(json.loads(decoded))
        except Exception as e:
            logger.warning(f"Failed to parse excluded tools header: {e}")
            return set()

    def _parse_connections(self) -> dict:
        try:
            decoded_json = base64.b64decode(self.mcp_header).decode("utf-8")
            return json.loads(decoded_json)
        except Exception as e:
            logger.error(f"Failed to parse X-MCP-Servers header: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse X-MCP-Servers: {str(e)}")

    def _create_transport_context(self, name: str, config: dict):
        url = config.get("url")
        httpUrl = config.get("httpUrl")
        mcp_type = config.get("type")
        headers = config.get("headers", {})

        if httpUrl:
            logger.warning(f"MCP server '{name}': 'httpUrl' is deprecated. Please migrate to 'url' with 'type: \"http\"'.")
            return streamable_http_client(
                url=httpUrl, 
                http_client=httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0))
            )
        
        if url and mcp_type == "sse":
            return sse_client(url=url, headers=headers)
        
        if url:
            # Default to streamable_http_client for "http" or undefined
            return streamable_http_client(
                url=url, 
                http_client=httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0))
            )
            
        return None

    async def _register_server_tools(
        self, 
        adapter: MCPServerAdapter, 
        session: ClientSession, 
        server_name: str, 
        excluded_tools: set, 
        global_tool_names: set, 
        fetch_raw_tools: bool
    ):
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            if tool.name in excluded_tools:
                continue
                
            mapped_name = tool.name
            if mapped_name in global_tool_names:
                mapped_name = f"{server_name}_{tool.name}"
            
            global_tool_names.add(mapped_name)
            adapter.register_tool(tool.name, mapped_name)
            self.adapters_map[mapped_name] = adapter
            
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
                    "serverName": server_name,
                    "name": mapped_name,
                    "description": tool.description or "",
                    "parameters": input_schema
                })
