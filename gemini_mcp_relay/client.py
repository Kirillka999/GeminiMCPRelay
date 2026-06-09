import logging
import asyncio
from google import genai
from google.genai import types
from google.genai.chats import AsyncChats

from gemini_mcp_relay.mcp_manager import MCPConnectionManager
from gemini_mcp_relay.orchestrator import generate_content_loop, stream_generate_content_loop

logger = logging.getLogger(__name__)

class AsyncModelsProxy:
    def __init__(self, wrapper):
        self._wrapper = wrapper

    async def _setup_and_run(self, loop_func, model: str, contents, config: types.GenerateContentConfig | None = None, **kwargs):
        manager = self._wrapper.mcp
        
        if config is None:
            config = types.GenerateContentConfig()
        elif isinstance(config, dict):
            config = types.GenerateContentConfig(**config)
            
        if manager.mcp_declarations:
            mcp_tool = types.Tool(function_declarations=manager.mcp_declarations)
            if config.tools is None:
                config.tools = []
            config.tools.append(mcp_tool)
        
        return await loop_func(
            client=self._wrapper.base_client,
            model_name=model,
            contents=contents,
            config=config,
            adapters_map=manager.adapters_map
        )

    async def generate_content(self, model: str, contents, config: types.GenerateContentConfig | None = None, **kwargs) -> types.GenerateContentResponse:
        return await self._setup_and_run(generate_content_loop, model, contents, config, **kwargs)

    async def generate_content_stream(self, model: str, contents, config: types.GenerateContentConfig | None = None, **kwargs):
        manager = self._wrapper.mcp
        
        if config is None:
            config = types.GenerateContentConfig()
        elif isinstance(config, dict):
            config = types.GenerateContentConfig(**config)
            
        if manager.mcp_declarations:
            mcp_tool = types.Tool(function_declarations=manager.mcp_declarations)
            if config.tools is None:
                config.tools = []
            config.tools.append(mcp_tool)

        async for chunk in stream_generate_content_loop(
            client=self._wrapper.base_client,
            model_name=model,
            contents=contents,
            config=config,
            adapters_map=manager.adapters_map
        ):
            yield chunk

class AioNamespace:
    def __init__(self, wrapper):
        self.models = AsyncModelsProxy(wrapper)
        self.chats = AsyncChats(modules=self.models)

class MCPClientWrapper:
    """
    A wrapper around google.genai.Client that intercepts generation requests,
    fetches tools from configured MCP servers, and autonomously executes the function calling loop locally.
    """
    def __init__(self, base_client: genai.Client, mcp_servers: dict = None, excluded_tools: list = None):
        self.base_client = base_client
        self._initial_mcp_servers = mcp_servers or {}
        
        self.mcp = MCPConnectionManager(excluded_tools=excluded_tools)
        self.aio = AioNamespace(self)
        self.models = self.aio.models
        self.chats = self.aio.chats

    async def __aenter__(self):
        if self._initial_mcp_servers:
            await self.mcp.connect_all_from_config(self._initial_mcp_servers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mcp.close()
