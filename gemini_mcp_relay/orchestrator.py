import asyncio
import inspect
import logging
from google.genai import types
from google.genai._extra_utils import (
    convert_argument_from_function,
    convert_number_values_for_function_call_args,
)

logger = logging.getLogger(__name__)

async def execute_single_tool(
    fc: types.FunctionCall,
    adapters_map: dict,
    local_tools: dict,
    interceptor = None
) -> types.Part:
    """
    Executes a single tool (MCP or local) with interceptor hooks, and returns a types.Part.
    """
    real_call_skipped = False
    response_dict = None

    if interceptor:
        try:
            intercept_result = await interceptor.before_tool_call(fc)
            if isinstance(intercept_result, dict):
                response_dict = intercept_result
                real_call_skipped = True
            elif isinstance(intercept_result, types.FunctionCall):
                fc = intercept_result
        except Exception as e:
            logger.error(f"Error in interceptor before_tool_call for '{fc.name}': {e}", exc_info=True)
            response_dict = {"error": f"Interceptor error: {str(e)}"}
            real_call_skipped = True

    if not real_call_skipped:
        adapter = adapters_map.get(fc.name)
        if adapter:
            _, response_dict = await adapter.call_tool_raw(fc)
        elif fc.name in local_tools:
            func = local_tools[fc.name]
            args = convert_number_values_for_function_call_args(fc.args) if fc.args else {}
            try:
                args = convert_argument_from_function(args, func)
                
                if inspect.iscoroutinefunction(func):
                    res = await func(**args)
                else:
                    res = await asyncio.to_thread(func, **args)
                
                response_dict = res if isinstance(res, dict) else {"result": res}
            except Exception as e:
                logger.error(f"Error calling local tool '{fc.name}': {e}", exc_info=True)
                response_dict = {"error": str(e)}
        else:
            logger.warning(f"Tool '{fc.name}' not found in provided MCP adapters or local tools.")
            response_dict = {"error": "Tool not found"}

    if interceptor:
        try:
            response_dict = await interceptor.after_tool_call(fc, response_dict)
        except Exception as e:
            logger.error(f"Error in interceptor after_tool_call for '{fc.name}': {e}", exc_info=True)
            response_dict = {"error": f"Interceptor post-processing error: {str(e)}"}

    part = types.Part.from_function_response(
        name=fc.name,
        response=response_dict
    )
    part.function_response.id = fc.id
    return part


async def generate_content_loop(
    client, 
    model_name: str, 
    contents, 
    config: types.GenerateContentConfig, 
    adapters_map: dict,
    fix_gemini_empty_response: bool = True,
    interceptor = None
) -> types.GenerateContentResponse:
    """
    Executes the sync function calling loop.
    Communicates with Google API, executes tools via MCP adapters, and squashes the results.
    """
    if not isinstance(contents, list):
        current_contents = [contents]
    else:
        current_contents = list(contents)
        
    local_tools = {}
    if config and getattr(config, "tools", None):
        for tool in config.tools:
            if callable(tool):
                local_tools[tool.__name__] = tool

    accumulated_parts = []
    retry_count = 0
    max_retries = 5
    
    while True:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=current_contents,
            config=config
        )
        
        # Workaround for Gemini empty text response bug
        is_empty_response = False
        if fix_gemini_empty_response and not response.function_calls:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                if len(parts) == 1 and getattr(parts[0], "text", None) == "":
                    if retry_count < max_retries:
                        retry_count += 1
                        is_empty_response = True
                        logger.warning(f"Empty text response received from Gemini API. Retrying ({retry_count}/{max_retries})...")
                    else:
                        logger.error("Max retries reached for empty text response workaround.")

        if is_empty_response:
            continue

        retry_count = 0

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
            part = await execute_single_tool(fc, adapters_map, local_tools, interceptor)
            response_parts.append(part)
                
        current_contents.append(types.Content(role="user", parts=response_parts))
        accumulated_parts.extend(response_parts)


async def stream_generate_content_loop(
    client, 
    model_name: str, 
    contents, 
    config: types.GenerateContentConfig, 
    adapters_map: dict,
    fix_gemini_empty_response: bool = True,
    interceptor = None
):
    """
    Executes the streaming function calling loop.
    Yields GenerateContentResponse objects, while managing tool execution behind the scenes.
    """
    if not isinstance(contents, list):
        current_contents = [contents]
    else:
        current_contents = list(contents)
    
    local_tools = {}
    if config and getattr(config, "tools", None):
        for tool in config.tools:
            if callable(tool):
                local_tools[tool.__name__] = tool

    retry_count = 0
    max_retries = 5
    
    while True:
        response_stream = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=current_contents,
            config=config
        )
        
        function_calls_to_process = []
        model_parts_this_turn = []
        chunks_this_turn = []
        
        async for chunk in response_stream:
            chunks_this_turn.append(chunk)
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                model_parts_this_turn.extend(chunk.candidates[0].content.parts)
            
            if chunk.function_calls:
                function_calls_to_process.extend(chunk.function_calls)

        # Workaround for Gemini empty text response bug
        is_empty_response = False
        if fix_gemini_empty_response and not function_calls_to_process:
            if len(model_parts_this_turn) == 1 and getattr(model_parts_this_turn[0], "text", None) == "":
                if retry_count < max_retries:
                    retry_count += 1
                    is_empty_response = True
                    logger.warning(f"Empty text response received from Gemini API in stream. Retrying ({retry_count}/{max_retries})...")
                else:
                    logger.error("Max retries reached for empty text response workaround in stream.")

        if is_empty_response:
            continue

        retry_count = 0

        for chunk in chunks_this_turn:
            yield chunk

        if not function_calls_to_process:
            break
            
        current_contents.append(types.Content(role="model", parts=model_parts_this_turn))
        
        response_parts = []
        for fc in function_calls_to_process:
            part = await execute_single_tool(fc, adapters_map, local_tools, interceptor)
            response_parts.append(part)
        
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
