"""
Integration test for basic MCP tool execution.

This test verifies that the proxy correctly parses the request, 
discovers the tools from the MCP server, autonomously executes 
the 'calculate_expression' tool, and returns the final text response.
"""
import base64
import json
import os
from google import genai

def test_math_mcp_calculation():
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

    prompt = (
        "Using the calculate_expression tool, evaluate the expression 157*73/14^2*13+145-12/13*99^2-13. "
        "In your response, provide only the number with exactly 6 decimal places in the format 0000.000000. "
        "Do not write anything else. Do not round the result, just truncate the extra digits."
    )
    expected_result = "-8154.908555"

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
    )

    # Verify the final text matches the expected calculation
    assert response.text.strip() == expected_result, f"Expected {expected_result}, got {response.text.strip()}"
