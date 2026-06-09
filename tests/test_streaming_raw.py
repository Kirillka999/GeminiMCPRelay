"""
Tests the raw streaming behavior of the wrapper.

Verifies that the wrapper correctly yields `GenerateContentResponse` objects,
and ensures that synthetic `functionResponse` chunks are injected into the stream 
so that the calling application is aware of internal tool executions.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_math_mcp_raw_streaming_sse():
    mcp_config = {
        "math_server": {
            "url": "https://mathematics.fastmcp.app/mcp"
        }
    }
    
    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    client = MCPClientWrapper(base_client)
    
    prompt = "Using the calculate_expression tool, calculate 200 + 300. Reply with just the number."

    found_call = False
    found_response = False
    found_text = False
    
    async with client:
        await client.mcp.add_server("math_server", mcp_config["math_server"])
        
        stream = client.models.generate_content_stream(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        
        async for chunk in stream:
            # chunk is a types.GenerateContentResponse
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                parts = chunk.candidates[0].content.parts
                for part in parts:
                    if part.function_call:
                        found_call = True
                    elif part.function_response:
                        found_response = True
                    elif part.text:
                        found_text = True
                        
    assert found_call, "Stream is missing a chunk with 'function_call'"
    assert found_response, "Stream is missing a synthetic chunk with 'function_response'"
    assert found_text, "Stream is missing a chunk with final 'text'"
