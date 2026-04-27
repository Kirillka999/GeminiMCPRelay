import os
import json
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

from app.mcp_manager import get_mcp_adapters_and_tools
from app.formatters import parse_request_payload
from app.orchestrator import generate_content_loop, stream_generate_content_loop

logger = logging.getLogger(__name__)
router = APIRouter()

GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL")

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

@router.post("/v1beta/models/{model_name}:generateContent")
async def generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    mcp_header = request.headers.get("x-mcp-servers")
    adapters_map, mcp_declarations = await get_mcp_adapters_and_tools(mcp_header)

    payload = await request.json()
    client, contents, config = _initialize_client_and_config(api_key, payload, mcp_declarations)
    
    try:
        response_dict = await generate_content_loop(client, model_name, contents, config, adapters_map)
        return response_dict
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1beta/models/{model_name}:streamGenerateContent")
async def stream_generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    mcp_header = request.headers.get("x-mcp-servers")
    adapters_map, mcp_declarations = await get_mcp_adapters_and_tools(mcp_header)

    payload = await request.json()
    client, contents, config = _initialize_client_and_config(api_key, payload, mcp_declarations)

    async def event_generator():
        try:
            async for event in stream_generate_content_loop(client, model_name, contents, config, adapters_map):
                yield event
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}", exc_info=True)
            error_json = {"error": {"code": 500, "message": str(e), "status": "INTERNAL"}}
            yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
