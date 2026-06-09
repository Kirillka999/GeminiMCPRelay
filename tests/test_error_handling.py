"""
Tests the MCP error propagation mechanism.

Verifies that when an MCP server returns an error (e.g., ValueError for invalid math),
the orchestrator gracefully intercepts it, wraps it inside `functionResponse.response["error"]`, 
and feeds it back to the model without crashing the generation loop.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_mcp_error_handling():
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}
    
    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None

    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    client = MCPClientWrapper(base_client)

    # Force the model to send an invalid math expression to trigger an MCP error
    prompt = "Using the calculate_expression tool, evaluate the expression '2 + a'. Pass exactly this string to the tool without any modifications."
    
    async with client:
        await client.mcp.add_server("math_server", mcp_config["math_server"])
        
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

    found_error = False
    parts = response.candidates[0].content.parts
    for part in parts:
        if part.function_response and part.function_response.name == "calculate_expression":
            resp_data = part.function_response.response
            
            if "error" in resp_data:
                found_error = True
            elif "result" in resp_data:
                res = resp_data["result"]
                if isinstance(res, dict) and (res.get("isError") or res.get("error")):
                    found_error = True
                elif isinstance(res, str) and "error" in res.lower():
                    found_error = True
    
    assert found_error, "Expected the MCP server to return a calculation error, but it did not."
