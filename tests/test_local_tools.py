"""
Integration test for simultaneous usage of MCP tools and local Python tools.

Verifies that the MCPClientWrapper correctly delegates calls to both:
1. Remote MCP server tools (e.g., math_server's calculate_expression)
2. User-provided local Python callables (e.g., get_weather)
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    
    Args:
        city: The name of the city to get weather for.
    """
    if "tokyo" in city.lower():
        return "The weather in Tokyo is sunny, 18°C."
    else:
        return f"The weather in {city} is pleasant, 20°C."


@pytest.mark.asyncio
async def test_simultaneous_mcp_and_local_tools():
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

    # Prompt requesting both the calculation tool (MCP) and weather tool (local)
    prompt = (
        "First, use the `calculate_expression` tool to evaluate 245 * 18. "
        "Second, use the `get_weather` tool to find the weather in Tokyo. "
        "In your final response, provide both the calculation result and describe the weather in Tokyo."
    )

    async with client:
        # Step 1: Add the real math_server MCP server
        await client.mcp.add_server("math_server", mcp_config["math_server"])
        
        # Step 2: Run generation with both MCP server and local tool registered
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[get_weather]
            )
        )

    text_resp = response.text.lower()
    
    # 245 * 18 = 4410 (could be formatted as 4,410)
    assert "4410" in text_resp or "4,410" in text_resp, f"Expected calculation result '4410' or '4,410' in response, got: {response.text}"
    # Tokyo weather details
    assert "tokyo" in text_resp, f"Expected 'tokyo' in response, got: {response.text}"
    assert "sunny" in text_resp or "18" in text_resp, f"Expected Tokyo weather details in response, got: {response.text}"
