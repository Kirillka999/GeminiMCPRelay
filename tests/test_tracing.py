"""
Tests for internal tracing of function calls and ID mapping.

When the proxy returns the final response to the client, it squashes 
all intermediate tool executions into a single 'model' response. 
This test verifies that the squashed parts contain `functionCall`, 
`functionResponse`, and `text`, and ensures that `id` fields match correctly.
"""
import base64
import json
import os
from google import genai

def test_math_mcp_tracing():
    mcp_config = {
        "math_server": {
            "url": "https://mathematics.fastmcp.app/mcp"
        }
    }
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")

    client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options={
            "base_url": os.environ.get("TEST_GEMINI_BASE_URL"),
            "headers": {"x-mcp-servers": mcp_header}
        }
    )

    prompt = "Using the calculate_expression tool, calculate 123 * 456. Output only the answer as a single number, nothing else."

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
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
