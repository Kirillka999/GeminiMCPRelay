import os
import json
import logging
import base64
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

from gemini_mcp_relay.mcp_manager import MCPConnectionManager
from gemini_mcp_relay.formatters import parse_request_payload
from gemini_mcp_relay.orchestrator import generate_content_loop, stream_generate_content_loop

logger = logging.getLogger(__name__)
router = APIRouter()

def _get_base_url(request: Request) -> str:
    base_url_header = request.headers.get("x-base-url")
    if not base_url_header:
        raise HTTPException(status_code=400, detail="Missing x-base-url header")
    try:
        decoded_bytes = base64.b64decode(base_url_header)
        base_url = decoded_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode x-base-url header: {str(e)}")
    if not base_url.strip():
        raise HTTPException(status_code=400, detail="The decoded x-base-url cannot be empty")
    return base_url.strip()

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
        manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header)
        try:
            await manager.connect_all_from_config(parsed_config, ignore_connection_errors=True)
            return {"tools": manager.raw_tools}
        finally:
            await manager.close()
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _initialize_client_and_config(api_key: str, base_url: str, payload: dict, mcp_declarations: list) -> tuple:
    contents, config = parse_request_payload(payload)

    if mcp_declarations:
        mcp_tool = types.Tool(function_declarations=mcp_declarations)
        if config.tools is None:
            config.tools = []
        config.tools.append(mcp_tool)

    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    client = genai.Client(
        api_key=api_key,
        http_options=http_opts
    )
    return client, contents, config

async def handle_generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    base_url = _get_base_url(request)

    mcp_header = request.headers.get("x-mcp-servers")
    excluded_header = request.headers.get("x-mcp-excluded-tools")
    
    manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header, excluded_header)
    try:
        await manager.connect_all_from_config(parsed_config)
        
        payload = await request.json()
        client, contents, config = _initialize_client_and_config(api_key, base_url, payload, manager.mcp_declarations)
        
        response_obj = await generate_content_loop(client, model_name, contents, config, manager.adapters_map)
        from gemini_mcp_relay.formatters import convert_bytes_to_b64
        response_dict = response_obj.model_dump(exclude_none=True, by_alias=True)
        return convert_bytes_to_b64(response_dict)
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

    base_url = _get_base_url(request)

    mcp_header = request.headers.get("x-mcp-servers")
    excluded_header = request.headers.get("x-mcp-excluded-tools")
    
    payload = await request.json()

    async def event_generator():
        manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header, excluded_header)
        from gemini_mcp_relay.formatters import convert_bytes_to_b64
        try:
            await manager.connect_all_from_config(parsed_config)
            client, contents, config = _initialize_client_and_config(api_key, base_url, payload, manager.mcp_declarations)
            
            async for chunk_obj in stream_generate_content_loop(client, model_name, contents, config, manager.adapters_map):
                chunk_dict = chunk_obj.model_dump(exclude_none=True, by_alias=True)
                convert_bytes_to_b64(chunk_dict)
                yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}", exc_info=True)
            error_json = {"error": {"code": 500, "message": str(e), "status": "INTERNAL"}}
            yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"
        finally:
            await manager.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

_shared_proxy_client: httpx.AsyncClient | None = None

def get_proxy_client() -> httpx.AsyncClient:
    global _shared_proxy_client
    if _shared_proxy_client is None:
        _shared_proxy_client = httpx.AsyncClient()
    return _shared_proxy_client

@router.on_event("shutdown")
async def shutdown_event():
    global _shared_proxy_client
    if _shared_proxy_client is not None:
        await _shared_proxy_client.aclose()
        _shared_proxy_client = None

async def transparent_proxy(request: Request, full_path: str):
    target_base = _get_base_url(request)
    target_url = f"{target_base.rstrip('/')}/{full_path}"
    
    query_params = request.url.query
    if query_params:
        target_url += f"?{query_params}"
        
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            headers[k] = v
            
    body = await request.body()
    
    client = get_proxy_client()
    resp = None
    try:
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
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
            
        return StreamingResponse(
            response_generator(),
            status_code=resp.status_code,
            headers=resp_headers
        )
    except Exception as e:
        if resp is not None:
            await resp.aclose()
        logger.error(f"Failed to send proxy request: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")

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
