import os
import json
import logging
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

from app.mcp_manager import MCPConnectionManager
from app.formatters import parse_request_payload
from app.orchestrator import generate_content_loop, stream_generate_content_loop

logger = logging.getLogger(__name__)
router = APIRouter()

GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL")

@router.get("/v1/mcp/tools")
async def list_mcp_tools(request: Request):
    """
    Returns a list of all available tools from the configured MCP servers.
    The response format is designed for frontend consumption, providing
    server names, tool names, descriptions, and their JSON schemas.
    """
    mcp_header = request.headers.get("x-mcp-servers")
    if not mcp_header:
        return {"tools": []}
        
    try:
        manager = MCPConnectionManager(mcp_header)
        try:
            await manager.connect(fetch_raw_tools=True)
            return {"tools": manager.raw_tools}
        finally:
            await manager.close()
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _initialize_client_and_config(api_key: str, payload: dict, mcp_declarations: list) -> tuple:
    contents, config = parse_request_payload(payload)

    if mcp_declarations:
        mcp_tool = types.Tool(function_declarations=mcp_declarations)
        if config.tools is None:
            config.tools = []
        config.tools.append(mcp_tool)

    http_opts = types.HttpOptions(base_url=GEMINI_BASE_URL) if GEMINI_BASE_URL else None
    client = genai.Client(
        api_key=api_key,
        http_options=http_opts
    )
    return client, contents, config

async def handle_generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    mcp_header = request.headers.get("x-mcp-servers")
    excluded_header = request.headers.get("x-mcp-excluded-tools")
    
    manager = MCPConnectionManager(mcp_header, excluded_header)
    try:
        await manager.connect()
        
        payload = await request.json()
        client, contents, config = _initialize_client_and_config(api_key, payload, manager.mcp_declarations)
        
        response_dict = await generate_content_loop(client, model_name, contents, config, manager.adapters_map)
        return response_dict
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        # re-raise HTTP exceptions (e.g. from manager.connect)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await manager.close()

async def handle_stream_generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    mcp_header = request.headers.get("x-mcp-servers")
    excluded_header = request.headers.get("x-mcp-excluded-tools")
    
    payload = await request.json()

    async def event_generator():
        manager = MCPConnectionManager(mcp_header, excluded_header)
        try:
            await manager.connect()
            client, contents, config = _initialize_client_and_config(api_key, payload, manager.mcp_declarations)
            
            async for event in stream_generate_content_loop(client, model_name, contents, config, manager.adapters_map):
                yield event
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}", exc_info=True)
            error_json = {"error": {"code": 500, "message": str(e), "status": "INTERNAL"}}
            yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"
        finally:
            await manager.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

async def transparent_proxy(request: Request, full_path: str):
    target_base = GEMINI_BASE_URL
    target_url = f"{target_base.rstrip('/')}/{full_path}"
    
    query_params = request.url.query
    if query_params:
        target_url += f"?{query_params}"
        
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            headers[k] = v
            
    body = await request.body()
    
    client = httpx.AsyncClient()
    req = client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body
    )
    
    resp = await client.send(req, stream=True)
    
    resp_headers = dict(resp.headers)
    resp_headers.pop("content-encoding", None)
    resp_headers.pop("content-length", None)
    
    async def response_generator():
        async for chunk in resp.aiter_bytes():
            yield chunk
        await client.aclose()
        
    return StreamingResponse(
        response_generator(),
        status_code=resp.status_code,
        headers=resp_headers
    )

@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_gateway(request: Request, full_path: str):
    """
    Catch-all router that intercepts generation endpoints and transparently proxies everything else.
    """
    if request.method == "POST":
        if full_path.endswith(":generateContent"):
            model_name = full_path.split("/")[-1].split(":")[0]
            return await handle_generate_content(model_name, request)
            
        elif full_path.endswith(":streamGenerateContent"):
            model_name = full_path.split("/")[-1].split(":")[0]
            return await handle_stream_generate_content(model_name, request)
            
    return await transparent_proxy(request, full_path)
