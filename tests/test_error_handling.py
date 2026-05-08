"""
Tests the MCP error propagation mechanism.

Verifies that when an MCP server returns an error (e.g., ValueError for invalid math),
the proxy gracefully intercepts it, wraps it inside `functionResponse.response["error"]`, 
and feeds it back to the model without crashing the generation loop.
"""
import base64
import json
import os
from google import genai

def test_mcp_error_handling():
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")

    client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options={
            "base_url": os.environ.get("TEST_GEMINI_BASE_URL"),
            "headers": {"x-mcp-servers": mcp_header}
        }
    )

    # Force the model to send an invalid math expression to trigger an MCP error
    prompt = "Using the calculate_expression tool, evaluate the expression '2 + a'. Pass exactly this string to the tool without any modifications."
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
    )

    found_error = False
    parts = response.candidates[0].content.parts
    for part in parts:
        if part.function_response and part.function_response.name == "calculate_expression":
            resp_data = part.function_response.response
            
            # Check if the 'error' field (injected by our mcphero patch) is present
            if "error" in resp_data:
                found_error = True
            elif "result" in resp_data:
                res = resp_data["result"]
                if isinstance(res, dict) and (res.get("isError") or res.get("error")):
                    found_error = True
                elif isinstance(res, str) and "error" in res.lower():
                    found_error = True
    
    assert found_error, "Expected the MCP server to return a calculation error, but it did not."
