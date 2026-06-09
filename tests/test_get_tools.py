"""
Tests the capability to successfully connect to an MCP server and 
verifies that the get_tools method works correctly.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_get_tools():
    mcp_config = {
        "Exa Search": {
            "url": "https://mcp.exa.ai/mcp"
        }
    }

    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None

    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    client = MCPClientWrapper(base_client)

    async with client:
        # Step 1: Connect to Exa Search MCP server
        await client.mcp.add_server("Exa Search", mcp_config["Exa Search"])
        
        # Verify the server was successfully registered and connected
        assert "Exa Search" in client.mcp.servers, "Exa Search server should be connected"
        
        # Step 2: Call get_tools() without arguments and verify results
        all_tools = client.mcp.get_tools()
        assert isinstance(all_tools, list), "get_tools() must return a list"
        assert len(all_tools) > 0, "Exa Search must expose at least one tool"
        
        # Verify structure of each returned tool
        for tool in all_tools:
            assert isinstance(tool, dict), "Each tool item must be a dictionary"
            assert "serverName" in tool, "Tool must contain key 'serverName'"
            assert "name" in tool, "Tool must contain key 'name'"
            assert "description" in tool, "Tool must contain key 'description'"
            assert "parameters" in tool, "Tool must contain key 'parameters'"
            assert isinstance(tool["parameters"], dict), "Tool parameters must be a dictionary schema"

        # Step 3: Call get_tools() with the specific server name (filtering)
        exa_tools = client.mcp.get_tools("Exa Search")
        assert isinstance(exa_tools, list), "get_tools('Exa Search') must return a list"
        assert len(exa_tools) == len(all_tools), "All active tools should belong to 'Exa Search'"
        for tool in exa_tools:
            assert tool["serverName"] == "Exa Search", "Filtered tool must belong to 'Exa Search'"

        # Step 4: Call get_tools() with a non-existent server name
        empty_tools = client.mcp.get_tools("NonExistentServer")
        assert isinstance(empty_tools, list), "get_tools('NonExistentServer') must return a list"
        assert len(empty_tools) == 0, "get_tools with non-existent server should return an empty list"
