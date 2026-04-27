import json
import logging
from google.genai import types

from app.formatters import build_squashed_response, build_synthetic_chunk, convert_bytes_to_b64

logger = logging.getLogger(__name__)

async def generate_content_loop(client, model_name: str, contents: list[types.Content], config: types.GenerateContentConfig, adapters_map: dict):
    """
    Executes the sync function calling loop.
    Communicates with Google API, executes tools via MCP adapters, and squashes the results.
    """
    current_contents = contents.copy()
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
            return build_squashed_response(accumulated_parts)
            
        response_parts = []
        for fc in response.function_calls:
            adapter = adapters_map.get(fc.name)
            if adapter:
                parts = await adapter.process_function_calls_as_parts([fc])
                response_parts.extend(parts)
            else:
                logger.warning(f"Tool '{fc.name}' not found in provided MCP adapters.")
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, 
                    response={"error": "Tool not found"}
                ))
                
        current_contents.append(types.Content(role="user", parts=response_parts))
        accumulated_parts.extend(response_parts)

async def stream_generate_content_loop(client, model_name: str, contents: list[types.Content], config: types.GenerateContentConfig, adapters_map: dict):
    """
    Executes the streaming function calling loop.
    Yields events in the standard SSE format, while managing tool execution behind the scenes.
    """
    current_contents = contents.copy()
    
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
            convert_bytes_to_b64(chunk_dict)
            
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
                response_parts.extend(parts)
            else:
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, 
                    response={"error": "Tool not found"}
                ))
        
        current_contents.append(types.Content(role="user", parts=response_parts))
        
        # Yield synthetic chunk with tool results so the client SDK can see them
        synthetic_dict = build_synthetic_chunk(response_parts)
        yield f"data: {json.dumps(synthetic_dict, ensure_ascii=False)}\n\n"
