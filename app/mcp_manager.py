import asyncio
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
        self.list_resources_tool_name = None
        self.read_resource_tool_name = None

    def register_tool(self, original_name: str, mapped_name: str):
        self.tool_mappings[mapped_name] = original_name

    def register_resource_tools(self, list_name: str, read_name: str):
        self.list_resources_tool_name = list_name
        self.read_resource_tool_name = read_name

    async def process_function_calls_as_parts(self, calls: list) -> list:
        parts = []
        for call in calls:
            gemini_name = call.name
            
            try:
                if gemini_name == self.list_resources_tool_name:
                    res = await asyncio.wait_for(self.session.list_resources(), timeout=30.0)
                    all_resources = []
                    for r in res.resources:
                        all_resources.append({
                            "uri": str(r.uri),
                            "name": getattr(r, "name", ""),
                            "description": r.description or "",
                            "mimeType": r.mimeType or ""
                        })
                    final_val = all_resources
                    response_key = "result"
                    
                elif gemini_name == self.read_resource_tool_name:
                    uri = call.args.get("uri")
                    if not uri:
                        raise ValueError("Missing 'uri' argument")
                    res = await asyncio.wait_for(self.session.read_resource(uri), timeout=120.0)
                    texts = []
                    for content in res.contents:
                        if hasattr(content, "text") and content.text:
                            texts.append(content.text)
                        elif hasattr(content, "blob") and content.blob:
                            texts.append(f"[Binary Blob: {getattr(content, 'mimeType', 'unknown')}]")
                    final_val = "\n".join(texts)
                    response_key = "result"
                    
                else:
                    mcp_name = self.tool_mappings.get(gemini_name, gemini_name)
                    result = await asyncio.wait_for(self.session.call_tool(mcp_name, call.args), timeout=300.0)
                    final_val = self._extract_result_content(result)
                    response_key = "error" if getattr(result, "isError", False) else "result"
                
                part = types.Part.from_function_response(
                    name=gemini_name,
                    response={response_key: final_val}
                )
                
            except Exception as e:
                logger.error(f"Error calling tool {gemini_name} on {self.name}: {e}", exc_info=True)
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
        self.server_stacks = []
        self.adapters_map = {}
        self.mcp_declarations = []
        self.raw_tools = []
        self._connected = False

    async def connect(self, fetch_raw_tools=False):
        if self._connected or not self.mcp_header:
            return
            
        excluded_tools = self._parse_excluded_tools()
        connections = self._parse_connections()

        server_data, tool_name_counts = await self._gather_servers_data(connections, excluded_tools, fetch_raw_tools)
        self._register_gathered_tools(server_data, tool_name_counts, excluded_tools, fetch_raw_tools)
                
        self._connected = True

    async def _gather_servers_data(self, connections: dict, excluded_tools: set, fetch_raw_tools: bool) -> tuple[list, dict]:
        server_data = []
        tool_name_counts = {}

        for name, config in connections.items():
            if config.get("command") or config.get("transport") == "stdio":
                raise HTTPException(status_code=400, detail=f"Transport 'stdio' is not allowed for server '{name}'.")
                
            transport_ctx = self._create_transport_context(name, config)
            if not transport_ctx:
                continue

            server_stack = AsyncExitStack()
            try:
                streams = await server_stack.enter_async_context(transport_ctx)
                read_stream, write_stream = streams[:2] if len(streams) >= 2 else streams
                
                session = await server_stack.enter_async_context(ClientSession(read_stream, write_stream))
                await asyncio.wait_for(session.initialize(), timeout=20.0)
                
                import re
                safe_server_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()
                adapter = MCPServerAdapter(name, session)
                
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=20.0)
                valid_tools = []
                for t in tools_response.tools:
                    if t.name in excluded_tools or f"{safe_server_name}_{t.name}" in excluded_tools:
                        continue
                    valid_tools.append(t)
                
                capabilities = session.get_server_capabilities()
                has_resources = bool(capabilities and getattr(capabilities, "resources", None))
                
                has_list = has_resources and "list_resources" not in excluded_tools and f"{safe_server_name}_list_resources" not in excluded_tools
                has_read = has_resources and "read_resource" not in excluded_tools and f"{safe_server_name}_read_resource" not in excluded_tools

                for t in valid_tools:
                    tool_name_counts[t.name] = tool_name_counts.get(t.name, 0) + 1
                    
                if has_list:
                    tool_name_counts["list_resources"] = tool_name_counts.get("list_resources", 0) + 1
                if has_read:
                    tool_name_counts["read_resource"] = tool_name_counts.get("read_resource", 0) + 1

                server_data.append({
                    "name": name,
                    "safe_name": safe_server_name,
                    "adapter": adapter,
                    "tools": valid_tools,
                    "has_list": has_list,
                    "has_read": has_read
                })
                
                self.server_stacks.append(server_stack)
                        
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                logger.error(f"Failed to connect or fetch tools from MCP server '{name}': {e}", exc_info=True)
                # Cleanup the failed isolated stack
                try:
                    await server_stack.aclose()
                except BaseException:
                    pass
                    
                if fetch_raw_tools:
                    continue
                raise HTTPException(status_code=502, detail=f"Failed to connect to MCP server '{name}': {str(e)}")
                
        return server_data, tool_name_counts

    def _register_gathered_tools(self, server_data: list, tool_name_counts: dict, excluded_tools: set, fetch_raw_tools: bool):
        for data in server_data:
            server_name = data["name"]
            safe_name = data["safe_name"]
            adapter = data["adapter"]
            
            for tool in data["tools"]:
                mapped_name = f"{safe_name}_{tool.name}" if tool_name_counts[tool.name] > 1 else tool.name
                
                adapter.register_tool(tool.name, mapped_name)
                self.adapters_map[mapped_name] = adapter
                
                input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else tool.input_schema
                if "type" not in input_schema:
                    input_schema["type"] = "object"
                    
                self._append_tool_declarations(
                    mapped_name=mapped_name,
                    description=tool.description or "",
                    schema=input_schema,
                    server_name=server_name,
                    fetch_raw_tools=fetch_raw_tools
                )

            list_name = f"{safe_name}_list_resources" if tool_name_counts.get("list_resources", 0) > 1 else "list_resources"
            read_name = f"{safe_name}_read_resource" if tool_name_counts.get("read_resource", 0) > 1 else "read_resource"

            # Always register with adapter to be safe, even if excluded from model
            adapter.register_resource_tools(list_name, read_name)

            if data.get("has_list"):
                self.adapters_map[list_name] = adapter
                self._append_tool_declarations(
                    mapped_name=list_name,
                    description=f"List available resources from the {server_name} server.",
                    schema={"type": "object", "properties": {}},
                    server_name=server_name,
                    fetch_raw_tools=fetch_raw_tools
                )

            if data.get("has_read"):
                self.adapters_map[read_name] = adapter
                self._append_tool_declarations(
                    mapped_name=read_name,
                    description=f"Read a specific resource from the {server_name} server using its URI.",
                    schema={
                        "type": "object",
                        "properties": {"uri": {"type": "string", "description": "The URI of the resource to read"}},
                        "required": ["uri"]
                    },
                    server_name=server_name,
                    fetch_raw_tools=fetch_raw_tools
                )

    def _append_tool_declarations(self, mapped_name: str, description: str, schema: dict, server_name: str, fetch_raw_tools: bool):
        decl = types.FunctionDeclaration(
            name=mapped_name,
            description=description,
            parameters_json_schema=schema
        )
        self.mcp_declarations.append(decl)
        
        if fetch_raw_tools:
            self.raw_tools.append({
                "serverName": server_name,
                "name": mapped_name,
                "description": description,
                "parameters": schema
            })

    async def close(self):
        for s in self.server_stacks:
            try:
                await s.aclose()
            except BaseException as e:
                logger.warning(f"Ignored error closing stack: {e}")

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
