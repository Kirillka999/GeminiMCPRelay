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
                        if "structuredContent" in res_val:
                            part.function_response.response = {"result": res_val["structuredContent"]}
                            continue
                        elif "content" in res_val and isinstance(res_val["content"], list):
                            texts = [c.get("text") for c in res_val["content"] if c.get("type") == "text" and c.get("text")]
                            if texts:
                                combined_text = "\n".join(texts)
                                try:
                                    parsed_text = json.loads(combined_text)
                                    part.function_response.response = {"result": parsed_text}
                                except json.JSONDecodeError:
                                    part.function_response.response = {"result": combined_text}
                                continue
                    
                    part.function_response.response = {"result": res_val}
                elif "error" in resp:
                    part.function_response.response = {"error": resp["error"]}
                    
        return parts
        
    adapter.process_function_calls_as_parts = clean_process_function_calls

async def get_mcp_adapters_and_tools(mcp_header: str) -> tuple[dict, list[types.FunctionDeclaration]]:
    adapters_map = {}
    all_declarations = []
    
    if not mcp_header:
        return adapters_map, all_declarations
        
    try:
        decoded_json = base64.b64decode(mcp_header).decode("utf-8")
        connections = json.loads(decoded_json)
        
        for name, config in connections.items():
            if config.get("transport") == "stdio":
                raise HTTPException(status_code=400, detail=f"Transport 'stdio' is not allowed.")
                
            url = config.get("url")
            headers = config.get("headers")
            if url:
                mcp_config = MCPServerConfig(url=url, name=name, headers=headers)
                adapter = MCPToolAdapterGemini(mcp_config)
                patch_adapter(adapter)
                
                declarations = await adapter.get_function_declarations()
                for decl in declarations:
                    adapters_map[decl.name] = adapter
                    all_declarations.append(decl)
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse X-MCP-Servers header: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse X-MCP-Servers: {str(e)}")
        
    return adapters_map, all_declarations
