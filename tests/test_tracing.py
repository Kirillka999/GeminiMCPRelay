"""
Tests for internal tracing of function calls and ID mapping.

When the wrapper returns the final response to the client, it includes 
all intermediate tool executions in the 'model' response. 
This test verifies that the parts contain `function_call`, 
`function_response`, and `text`, and ensures that `id` fields match correctly.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_math_mcp_tracing():
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

    prompt = "Using the calculate_expression tool, calculate 123 * 456. Output only the answer as a single number, nothing else."

    async with client:
        await client.mcp.add_server("math_server", mcp_config["math_server"])
        
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

    found_call = False
    found_response = False
    found_text = False

    # The squashed response contains all intermediate steps in response.candidates[0].content.parts
    parts = response.candidates[0].content.parts
    
    call_id = None
    
    for part in parts:
        if part.function_call:
            found_call = True
            call_id = part.function_call.id
            assert call_id is not None, "function_call is missing the required 'id' field!"
            assert part.function_call.name == "calculate_expression"
            assert "expr" in part.function_call.args
        elif part.function_response:
            found_response = True
            assert part.function_response.id == call_id, f"Response ID ({part.function_response.id}) does not match Call ID ({call_id})!"
            assert part.function_response.name == "calculate_expression"
            assert part.function_response.response is not None
        elif part.text:
            found_text = True

    assert found_call, "part.function_call is missing from the response"
    assert found_response, "part.function_response is missing from the response"
    assert found_text, "part.text is missing from the response"
