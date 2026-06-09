"""
Tests the 'excluded_tools' functionality.

This test verifies two layers of security:
1. The wrapper must remove the excluded tool from its internal `mcp_declarations` list.
2. The model must not execute the excluded tool, even when prompted to do so.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_excluded_tools():
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}
    excluded = ["calculate_expression"]

    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    # Initialize wrapper with the excluded tool
    client = MCPClientWrapper(base_client, excluded_tools=excluded)

    async with client:
        await client.mcp.add_server("math_server", mcp_config["math_server"])
        
        # VERIFICATION 1: The tool must not be present in the internal declarations
        available_tools = [decl.name for decl in client.mcp.mcp_declarations]
        assert "calculate_expression" not in available_tools, f"The tool 'calculate_expression' is present in declarations despite being excluded! Available: {available_tools}"
        
        # STEP 2: Make a real request to ensure the model cannot call it
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Using the calculate_expression tool, calculate 2+2. Can you do this?",
        )
        
        # VERIFICATION 2: The model must not emit a function_call for the excluded tool
        for part in response.candidates[0].content.parts:
            if part.function_call:
                assert part.function_call.name != "calculate_expression", "The model successfully called an excluded tool!"
