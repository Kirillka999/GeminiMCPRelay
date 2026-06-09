"""
Integration test for basic MCP tool execution using the local SDK wrapper.

This test verifies that the MCPClientWrapper correctly connects to an MCP server,
autonomously executes the 'calculate_expression' tool, and returns the final text response.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_math_mcp_calculation():
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

    prompt = (
        "Using the calculate_expression tool, evaluate the expression 157*73/14^2*13+145-12/13*99^2-13. "
        "In your response, provide only the number with exactly 6 decimal places in the format 0000.000000. "
        "Do not write anything else. Do not round the result, just truncate the extra digits."
    )
    expected_result = "-8154.908555"

    async with client:
        await client.mcp.add_server("math_server", mcp_config["math_server"])
        
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

    # Verify the final text matches the expected calculation
    assert response.text.strip() == expected_result, f"Expected {expected_result}, got {response.text.strip()}"
