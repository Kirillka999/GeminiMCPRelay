import os
import json
import base64
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

from app.mcp_manager import get_mcp_adapters_and_tools
from app.gemini_parser import parse_request_payload

logger = logging.getLogger(__name__)
router = APIRouter()

GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL")

@router.post("/v1beta/models/{model_name}:generateContent")
async def generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    mcp_header = request.headers.get("x-mcp-servers")
    adapters_map, mcp_declarations = await get_mcp_adapters_and_tools(mcp_header)

    payload = await request.json()
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

    current_contents = contents.copy()
    accumulated_parts = []
    
    try:
        while True:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=current_contents,
                config=config
            )
            
            if response.candidates and response.candidates[0].content:
                model_content = response.candidates[0].content
                current_contents.append(model_content)
                if model_content.parts:
                    accumulated_parts.extend(model_content.parts)
            
            if not response.function_calls:
                final_response = types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            index=0,
                            finish_reason=types.FinishReason.STOP,
                            content=types.Content(
                                role="model",
                                parts=accumulated_parts
                            )
                        )
                    ]
                )
                
                response_dict = final_response.model_dump(exclude_none=True, by_alias=True)
                
                def convert_bytes(obj):
                    if isinstance(obj, dict):
                        for k, v in list(obj.items()):
                            if isinstance(v, bytes):
                                obj[k] = base64.b64encode(v).decode('utf-8')
                            else:
                                convert_bytes(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            convert_bytes(item)
                
                convert_bytes(response_dict)
                return response_dict
                
            response_parts = []
            for fc in response.function_calls:
                adapter = adapters_map.get(fc.name)
                if adapter:
                    parts = await adapter.process_function_calls_as_parts([fc])
                    # Copy the ID from functionCall to each functionResponse part
                    if hasattr(fc, 'id') and fc.id:
                        for part in parts:
                            if part.function_response:
                                part.function_response.id = fc.id
                    response_parts.extend(parts)
                else:
                    logger.warning(f"Tool '{fc.name}' not found in provided MCP adapters.")
                    response_parts.append(types.Part.from_function_response(
                        name=fc.name, 
                        response={"error": "Tool not found"},
                        id=getattr(fc, 'id', None)
                    ))
                    
            current_contents.append(types.Content(role="user", parts=response_parts))
            accumulated_parts.extend(response_parts)
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

    async def event_generator():
        current_contents = contents.copy()
        
        try:
            while True:
                response_stream = await client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=current_contents,
                    config=config
                )
                
                function_calls_to_process = []
                model_parts_this_turn = []
                
                async for chunk in response_stream:
                    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                        model_parts_this_turn.extend(chunk.candidates[0].content.parts)

                    chunk_dict = chunk.model_dump(exclude_none=True, by_alias=True)
                    
                    def convert_bytes(obj):
                        if isinstance(obj, dict):
                            for k, v in list(obj.items()):
                                if isinstance(v, bytes):
                                    obj[k] = base64.b64encode(v).decode('utf-8')
                                else:
                                    convert_bytes(v)
                        elif isinstance(obj, list):
                            for item in obj:
                                convert_bytes(item)
                    
                    convert_bytes(chunk_dict)
                    
                    yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                    
                    if chunk.function_calls:
                        function_calls_to_process.extend(chunk.function_calls)
                        
                if not function_calls_to_process:
                    break
                    
                current_contents.append(types.Content(role="model", parts=model_parts_this_turn))
                
                response_parts = []
                for fc in function_calls_to_process:
                    adapter = adapters_map.get(fc.name)
                    if adapter:
                        parts = await adapter.process_function_calls_as_parts([fc])
                        # Copy ID to response parts
                        if hasattr(fc, 'id') and fc.id:
                            for part in parts:
                                if part.function_response:
                                    part.function_response.id = fc.id
                        response_parts.extend(parts)
                    else:
                        response_parts.append(types.Part.from_function_response(
                            name=fc.name, 
                            response={"error": "Tool not found"},
                            id=getattr(fc, 'id', None)
                        ))
                
                current_contents.append(types.Content(role="user", parts=response_parts))
                
                # Yield synthetic chunk with tool results so the client SDK can see them
                synthetic_chunk = types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            index=0,
                            content=types.Content(
                                role="model",
                                parts=response_parts
                            )
                        )
                    ]
                )
                synthetic_dict = synthetic_chunk.model_dump(exclude_none=True, by_alias=True)
                convert_bytes(synthetic_dict)
                yield f"data: {json.dumps(synthetic_dict, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}", exc_info=True)
            error_json = {"error": {"code": 500, "message": str(e), "status": "INTERNAL"}}
            yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
