import json
import base64
import logging
from fastapi import HTTPException
from google.genai import types
from mcphero import MCPToolAdapterGemini, MCPServerConfig

logger = logging.getLogger(__name__)

def patch_adapter(adapter: MCPToolAdapterGemini):
    """
    Patch for mcphero: uses the new `parameters_json_schema` field 
    to bypass strict Schema validation and pass the raw JSON Schema natively.
    """
    def safe_to_gemini(tool):
        # We completely bypass the original conversion logic and just create the declaration
        # using parameters_json_schema, passing the raw MCP input_schema dict.
        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.input_schema
        )
        
    adapter._to_gemini_declaration = safe_to_gemini

    # Patch process_function_calls_as_parts to clean up JSON-RPC envelope
    original_process = adapter.process_function_calls_as_parts
    
    async def clean_process_function_calls(calls):
        parts = await original_process(calls)
        for part in parts:
            if part.function_response and part.function_response.response:
                resp = part.function_response.response
                
                # Cleanup JSON-RPC wrapping
                if "result" in resp:
                    res_val = resp["result"]
                    
                    if isinstance(res_val, dict):
                        is_error = res_val.get("isError", False)
                        
                        # Handle structured content (e.g., Xcode MCP bridge)
                        if "structuredContent" in res_val:
                            final_val = res_val["structuredContent"]
                        # Handle standard MCP content array
                        elif "content" in res_val and isinstance(res_val["content"], list):
                            texts = [c.get("text") for c in res_val["content"] if c.get("type") == "text" and c.get("text")]
                            if texts:
                                combined_text = "\n".join(texts)
                                try:
                                    final_val = json.loads(combined_text)
                                except json.JSONDecodeError:
                                    final_val = combined_text
                            else:
                                final_val = "Empty or non-text response"
                        else:
                            final_val = res_val
                            
                        # Route to error or result based on MCP isError flag
                        if is_error:
                            part.function_response.response = {"error": final_val}
                        else:
                            part.function_response.response = {"result": final_val}
                    else:
                        part.function_response.response = {"result": res_val}
                        
                # Handle JSON-RPC level errors
                elif "error" in resp:
                    part.function_response.response = {"error": resp["error"]}
                    
        return parts
        
    adapter.process_function_calls_as_parts = clean_process_function_calls

async def get_mcp_adapters_and_tools(mcp_header: str, excluded_tools_header: str = None) -> tuple[dict, list[types.FunctionDeclaration]]:
    adapters_map = {}
    all_declarations = []
    
    if not mcp_header:
        return adapters_map, all_declarations
        
    excluded_tools = set()
    if excluded_tools_header:
        try:
            decoded = base64.b64decode(excluded_tools_header).decode("utf-8")
            excluded_tools = set(json.loads(decoded))
        except Exception as e:
            logger.warning(f"Failed to parse excluded tools header: {e}")
            
    try:
        decoded_json = base64.b64decode(mcp_header).decode("utf-8")
        connections = json.loads(decoded_json)
        
        server_configs = []
        for name, config in connections.items():
            if config.get("transport") == "stdio":
                raise HTTPException(status_code=400, detail=f"Transport 'stdio' is not allowed.")
                
            url = config.get("url")
            headers = config.get("headers")
            if url:
                server_configs.append(MCPServerConfig(url=url, name=name, headers=headers))
                
        if not server_configs:
            return adapters_map, all_declarations
            
        # Create a single adapter for all servers to allow mcphero to resolve naming collisions
        adapter = MCPToolAdapterGemini(server_configs)
        patch_adapter(adapter)
        
        declarations = await adapter.get_function_declarations()
        for decl in declarations:
            if decl.name not in excluded_tools:
                # The adapter internally knows which server owns which tool based on the name
                adapters_map[decl.name] = adapter
                all_declarations.append(decl)
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse X-MCP-Servers header: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse X-MCP-Servers: {str(e)}")
        
    return adapters_map, all_declarations

async def get_mcp_raw_tools(mcp_header: str) -> list[dict]:
    """
    Parses the X-MCP-Servers header, connects to MCP servers, and returns a flat list
    of raw tool definitions for frontend consumption.
    """
    if not mcp_header:
        return []
        
    try:
        decoded_json = base64.b64decode(mcp_header).decode("utf-8")
        connections = json.loads(decoded_json)
        
        server_configs = []
        for name, config in connections.items():
            if config.get("transport") == "stdio":
                continue # Skip stdio for tool listing
                
            url = config.get("url")
            headers = config.get("headers")
            if url:
                server_configs.append(MCPServerConfig(url=url, name=name, headers=headers))
                
        all_tools = []
        if server_configs:
            # Create a single adapter to properly resolve collision names
            adapter = MCPToolAdapterGemini(server_configs)
            
            # Fetch raw tools using base adapter method
            tools = await adapter.discover_tools()
            for t in tools:
                all_tools.append({
                    "serverName": t.server_name,
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema
                })
                    
        return all_tools
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to list MCP tools: {str(e)}")
