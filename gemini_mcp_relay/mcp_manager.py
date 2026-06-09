import asyncio
import json
import base64
import logging
import re
from contextlib import AsyncExitStack
import httpx
from fastapi import HTTPException
from google.genai import types

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from gemini_mcp_relay.formatters import normalize_tool_schema

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

    async def call_tool_raw(self, call: types.FunctionCall) -> tuple[str, dict]:
        """
        Executes a single tool call and returns a tuple (response_key, response_dict).
        response_key can be 'result' or 'error'.
        """
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
                return "result", {"result": all_resources}
                
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
                return "result", {"result": final_val}
                
            else:
                mcp_name = self.tool_mappings.get(gemini_name, gemini_name)
                result = await asyncio.wait_for(self.session.call_tool(mcp_name, call.args), timeout=300.0)
                final_val = self._extract_result_content(result)
                response_key = "error" if getattr(result, "isError", False) else "result"
                return response_key, {response_key: final_val}
            
        except Exception as e:
            logger.error(f"Error calling tool {gemini_name} on {self.name}: {e}", exc_info=True)
            return "error", {"error": str(e)}

    async def process_function_calls_as_parts(self, calls: list) -> list:
        parts = []
        for call in calls:
            _, response_dict = await self.call_tool_raw(call)
            part = types.Part.from_function_response(
                name=call.name,
                response=response_dict
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

class ReactiveSet(set):
    def __init__(self, iterable=None, on_change=None):
        super().__init__(iterable or [])
        self.on_change = on_change

    def _notify(self):
        if self.on_change:
            self.on_change()

    def add(self, element):
        if element not in self:
            super().add(element)
            self._notify()

    def remove(self, element):
        super().remove(element)
        self._notify()

    def discard(self, element):
        if element in self:
            super().discard(element)
            self._notify()

    def pop(self):
        item = super().pop()
        self._notify()
        return item

    def clear(self):
        if len(self) > 0:
            super().clear()
            self._notify()

    def update(self, *others):
        initial_len = len(self)
        super().update(*others)
        if len(self) != initial_len:
            self._notify()

    def intersection_update(self, *others):
        super().intersection_update(*others)
        self._notify()

    def difference_update(self, *others):
        super().difference_update(*others)
        self._notify()

    def symmetric_difference_update(self, *others):
        super().symmetric_difference_update(*others)
        self._notify()


class ServerState:
    def __init__(self, stack: AsyncExitStack, session: ClientSession, adapter: MCPServerAdapter):
        self.stack = stack
        self.session = session
        self.adapter = adapter
        self.cached_tools = []
        self.has_list = False
        self.has_read = False

class MCPConnectionManager:
    def __init__(self, excluded_tools: list = None):
        self.excluded_tools = ReactiveSet(excluded_tools, on_change=self._on_excluded_tools_changed)
        self.servers = {}  # name -> ServerState
        self.adapters_map = {}
        self.mcp_declarations = []
        self.raw_tools = []
        
    def _on_excluded_tools_changed(self):
        self._rebuild_state()

    @classmethod
    def from_http_headers(cls, mcp_header: str, excluded_tools_header: str = None):
        server_config = {}
        excluded_tools = []
        
        if mcp_header:
            try:
                decoded_json = base64.b64decode(mcp_header).decode("utf-8")
                server_config = json.loads(decoded_json)
            except Exception as e:
                logger.error(f"Failed to parse X-MCP-Servers header: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to parse X-MCP-Servers: {str(e)}")
                
        if excluded_tools_header:
            try:
                decoded = base64.b64decode(excluded_tools_header).decode("utf-8")
                excluded_tools = json.loads(decoded)
            except Exception as e:
                logger.warning(f"Failed to parse excluded tools header: {e}")

        instance = cls(excluded_tools=excluded_tools)
        return instance, server_config

    async def connect_all_from_config(self, config_dict: dict, ignore_connection_errors: bool = False):
        """Helper for the FastAPI server to connect multiple servers at once."""
        for name, config in config_dict.items():
            await self.add_server(name, config, skip_rebuild=True, raise_on_error=not ignore_connection_errors)
        self._rebuild_state()

    async def add_server(self, name: str, config: dict, skip_rebuild=False, raise_on_error: bool = True):
        if name in self.servers:
            logger.warning(f"Server '{name}' is already connected.")
            return

        if config.get("command") or config.get("transport") == "stdio":
            raise HTTPException(status_code=400, detail=f"Transport 'stdio' is not allowed for server '{name}'.")
            
        transport_ctx, http_client = self._create_transport_context(name, config)
        if not transport_ctx:
            return

        server_stack = AsyncExitStack()
        try:
            try:
                if http_client:
                    await server_stack.enter_async_context(http_client)
                streams = await server_stack.enter_async_context(transport_ctx)
                read_stream, write_stream = streams[:2] if len(streams) >= 2 else streams
                
                session = await server_stack.enter_async_context(ClientSession(read_stream, write_stream))
                await asyncio.wait_for(session.initialize(), timeout=20.0)
            except BaseException as e:
                raise RuntimeError(f"Connection failed: {e}")

            adapter = MCPServerAdapter(name, session)
            state = ServerState(server_stack, session, adapter)
            self.servers[name] = state
            
            # Fetch tools and capabilities once and cache them
            tools_response = await asyncio.wait_for(session.list_tools(), timeout=20.0)
            state.cached_tools = tools_response.tools
            
            capabilities = session.get_server_capabilities()
            has_resources = bool(capabilities and getattr(capabilities, "resources", None))
            state.has_list = has_resources
            state.has_read = has_resources
            
            if not skip_rebuild:
                self._rebuild_state()
                    
        except asyncio.CancelledError:
            try:
                await server_stack.aclose()
            except BaseException:
                pass
            raise
        except BaseException as e:
            logger.error(f"Failed to connect or fetch tools from MCP server '{name}': {e}", exc_info=True)
            try:
                await server_stack.aclose()
            except BaseException:
                pass
                
            if raise_on_error:
                raise HTTPException(status_code=502, detail=f"Failed to connect to MCP server '{name}': {str(e)}")

    async def remove_server(self, name: str):
        if name not in self.servers:
            return
            
        state = self.servers.pop(name)
        try:
            await state.stack.aclose()
        except BaseException as e:
            logger.warning(f"Ignored error closing stack for '{name}': {e}")
            
        self._rebuild_state()

    def get_tools(self, server_name: str = None) -> list[dict]:
        """
        Returns a list of active tools, optionally filtered by a specific server.
        The tools are returned as dictionaries containing 'serverName', 'name', 
        'description', and 'parameters' (JSON schema).
        """
        if not server_name:
            return self.raw_tools
        return [t for t in self.raw_tools if t["serverName"] == server_name]

    def _rebuild_state(self):
        self.adapters_map.clear()
        self.mcp_declarations.clear()
        self.raw_tools.clear()
        
        server_data, tool_name_counts = self._gather_servers_data(self.excluded_tools)
        self._register_gathered_tools(server_data, tool_name_counts, self.excluded_tools)

    def _gather_servers_data(self, excluded_tools: set) -> tuple[list, dict]:
        server_data = []
        tool_name_counts = {}

        for name, state in self.servers.items():
            safe_server_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()
            
            try:
                valid_tools = []
                for t in state.cached_tools:
                    if t.name in excluded_tools or f"{safe_server_name}_{t.name}" in excluded_tools:
                        continue
                    valid_tools.append(t)
                
                has_list = state.has_list and "list_resources" not in excluded_tools and f"{safe_server_name}_list_resources" not in excluded_tools
                has_read = state.has_read and "read_resource" not in excluded_tools and f"{safe_server_name}_read_resource" not in excluded_tools

                for t in valid_tools:
                    tool_name_counts[t.name] = tool_name_counts.get(t.name, 0) + 1
                    
                if has_list:
                    tool_name_counts["list_resources"] = tool_name_counts.get("list_resources", 0) + 1
                if has_read:
                    tool_name_counts["read_resource"] = tool_name_counts.get("read_resource", 0) + 1

                server_data.append({
                    "name": name,
                    "safe_name": safe_server_name,
                    "adapter": state.adapter,
                    "tools": valid_tools,
                    "has_list": has_list,
                    "has_read": has_read
                })
            except Exception as e:
                logger.error(f"Error gathering tools from '{name}': {e}", exc_info=True)
            
        return server_data, tool_name_counts

    def _register_gathered_tools(self, server_data: list, tool_name_counts: dict, excluded_tools: set):
        for data in server_data:
            server_name = data["name"]
            safe_name = data["safe_name"]
            adapter = data["adapter"]
            
            for tool in data["tools"]:
                mapped_name = f"{safe_name}_{tool.name}" if tool_name_counts[tool.name] > 1 else tool.name
                
                adapter.register_tool(tool.name, mapped_name)
                self.adapters_map[mapped_name] = adapter
                
                raw_schema = getattr(tool, "inputSchema", getattr(tool, "input_schema", None))
                input_schema = normalize_tool_schema(raw_schema)

                self._append_tool_declarations(
                    mapped_name=mapped_name,
                    description=tool.description or "",
                    schema=input_schema,
                    server_name=server_name
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
                    server_name=server_name
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
                    server_name=server_name
                )

    def _append_tool_declarations(self, mapped_name: str, description: str, schema: dict, server_name: str):
        decl = types.FunctionDeclaration(
            name=mapped_name,
            description=description,
            parameters_json_schema=schema
        )
        self.mcp_declarations.append(decl)
        
        self.raw_tools.append({
            "serverName": server_name,
            "name": mapped_name,
            "description": description,
            "parameters": schema
        })

    async def close(self):
        for state in self.servers.values():
            try:
                await state.stack.aclose()
            except BaseException as e:
                logger.warning(f"Ignored error closing stack: {e}")
        self.servers.clear()

    def _create_transport_context(self, name: str, config: dict):
        url = config.get("url")
        httpUrl = config.get("httpUrl")
        mcp_type = config.get("type")
        headers = config.get("headers", {})

        if httpUrl:
            logger.warning(f"MCP server '{name}': 'httpUrl' is deprecated. Please migrate to 'url' with 'type: \"http\"'.")
            client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0))
            return streamable_http_client(url=httpUrl, http_client=client), client
        
        if url and mcp_type == "sse":
            return sse_client(url=url, headers=headers), None
        
        if url:
            # Default to streamable_http_client for "http" or undefined
            client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(5.0, read=300.0))
            return streamable_http_client(url=url, http_client=client), client
            
        return None, None
