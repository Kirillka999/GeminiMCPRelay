import logging
from google import genai
from google.genai import types
from google.genai.chats import AsyncChats

from gemini_mcp_relay.mcp_manager import MCPConnectionManager
from gemini_mcp_relay.orchestrator import generate_content_loop, stream_generate_content_loop
from gemini_mcp_relay.formatters import unsquash_contents

logger = logging.getLogger(__name__)

class AsyncModelsProxy:
    def __init__(self, wrapper):
        self._wrapper = wrapper

    def _prepare_contents(self, contents):
        if not isinstance(contents, list):
            contents_list = [contents]
        else:
            contents_list = list(contents)
            
        formatted_contents = []
        for c in contents_list:
            if isinstance(c, types.Content):
                formatted_contents.append(c)
            elif isinstance(c, str):
                formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=c)]))
            else:
                formatted_contents.append(c)
                
        # Unsquash the history to prevent SDK sequence validation errors
        return unsquash_contents(formatted_contents)

    async def _setup_and_run(self, loop_func, model: str, contents, config: types.GenerateContentConfig | None = None, **kwargs):
        manager = self._wrapper.mcp
        
        # Extract interceptor if provided in method call, fallback to wrapper's global interceptor
        interceptor = kwargs.pop("interceptor", self._wrapper.interceptor)
        
        if config is None:
            config = types.GenerateContentConfig()
        elif isinstance(config, dict):
            config = types.GenerateContentConfig(**config)
            
        # Always disable Google's built-in Automated Function Calling (AFC)
        # to ensure our custom orchestrator manages the lifecycle and triggers interceptors.
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
            
        if manager.mcp_declarations:
            mcp_tool = types.Tool(function_declarations=manager.mcp_declarations)
            if config.tools is None:
                config.tools = []
            config.tools.append(mcp_tool)
        
        processed_contents = self._prepare_contents(contents)
        
        return await loop_func(
            client=self._wrapper.base_client,
            model_name=model,
            contents=processed_contents,
            config=config,
            adapters_map=manager.adapters_map,
            interceptor=interceptor
        )

    async def generate_content(self, model: str, contents, config: types.GenerateContentConfig | None = None, **kwargs) -> types.GenerateContentResponse:
        return await self._setup_and_run(generate_content_loop, model, contents, config, **kwargs)

    async def generate_content_stream(self, model: str, contents, config: types.GenerateContentConfig | None = None, **kwargs):
        manager = self._wrapper.mcp
        
        interceptor = kwargs.pop("interceptor", self._wrapper.interceptor)
        
        if config is None:
            config = types.GenerateContentConfig()
        elif isinstance(config, dict):
            config = types.GenerateContentConfig(**config)
            
        # Always disable Google's built-in Automated Function Calling (AFC)
        # to ensure our custom orchestrator fully manages the lifecycle and triggers interceptors.
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
            
        if manager.mcp_declarations:
            mcp_tool = types.Tool(function_declarations=manager.mcp_declarations)
            if config.tools is None:
                config.tools = []
            config.tools.append(mcp_tool)

        processed_contents = self._prepare_contents(contents)

        async for chunk in stream_generate_content_loop(
            client=self._wrapper.base_client,
            model_name=model,
            contents=processed_contents,
            config=config,
            adapters_map=manager.adapters_map,
            interceptor=interceptor
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
    def __init__(self, base_client: genai.Client, mcp_servers: dict = None, excluded_tools: list = None, interceptor = None):
        self.base_client = base_client
        self._initial_mcp_servers = mcp_servers or {}
        self.interceptor = interceptor
        
        self.mcp = MCPConnectionManager(excluded_tools=excluded_tools)
        self.aio = AioNamespace(self)
        self.models = self.aio.models
        self.chats = self.aio.chats

    async def __aenter__(self):
        if self._initial_mcp_servers:
            try:
                await self.mcp.connect_all_from_config(self._initial_mcp_servers)
            except BaseException:
                await self.mcp.close()
                raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mcp.close()
