import json
import logging
from google.genai import types

from gemini_mcp_relay.formatters import build_squashed_response, build_synthetic_chunk, convert_bytes_to_b64

logger = logging.getLogger(__name__)

async def generate_content_loop(client, model_name: str, contents, config: types.GenerateContentConfig, adapters_map: dict) -> types.GenerateContentResponse:
    """
    Executes the sync function calling loop.
    Communicates with Google API, executes tools via MCP adapters, and squashes the results.
    """
    if not isinstance(contents, list):
        current_contents = [contents]
    else:
        current_contents = list(contents)
        
    accumulated_parts = []
    
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
            return types.GenerateContentResponse(
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
            
        response_parts = []
        for fc in response.function_calls:
            adapter = adapters_map.get(fc.name)
            if adapter:
                parts = await adapter.process_function_calls_as_parts([fc])
                response_parts.extend(parts)
            else:
                logger.warning(f"Tool '{fc.name}' not found in provided MCP adapters.")
                fallback_part = types.Part.from_function_response(
                    name=fc.name, 
                    response={"error": "Tool not found"}
                )
                fallback_part.function_response.id = fc.id
                response_parts.append(fallback_part)
                
        current_contents.append(types.Content(role="user", parts=response_parts))
        accumulated_parts.extend(response_parts)

async def stream_generate_content_loop(client, model_name: str, contents, config: types.GenerateContentConfig, adapters_map: dict):
    """
    Executes the streaming function calling loop.
    Yields GenerateContentResponse objects, while managing tool execution behind the scenes.
    """
    if not isinstance(contents, list):
        current_contents = [contents]
    else:
        current_contents = list(contents)
    
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

            yield chunk
            
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
                response_parts.extend(parts)
            else:
                fallback_part = types.Part.from_function_response(
                    name=fc.name, 
                    response={"error": "Tool not found"}
                )
                fallback_part.function_response.id = fc.id
                response_parts.append(fallback_part)
        
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
        yield synthetic_chunk
