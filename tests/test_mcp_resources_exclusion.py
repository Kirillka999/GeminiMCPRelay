"""
Tests the wrapper's capability to correctly exclude synthetic MCP resource tools.

Verifies that if a user passes `list_resources` and/or `read_resource` (or their prefixed equivalents) 
in the `excluded_tools` list, the wrapper successfully hides them from the model.
The model should not be able to call them even if prompted to do so.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_mcp_resources_exclusion():
    mcp_config = {
        "Exa Search": {
            "url": "https://mcp.exa.ai/mcp"
        }
    }
    
    excluded_tools = ["list_resources", "read_resource"]

    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    client = MCPClientWrapper(base_client, excluded_tools=excluded_tools)

    async with client:
        await client.mcp.add_server("Exa Search", mcp_config["Exa Search"])
        
        # Verify tools are strictly removed from declarations
        available_tools = [decl.name for decl in client.mcp.mcp_declarations]
        assert "list_resources" not in available_tools, f"list_resources was not excluded: {available_tools}"
        assert "read_resource" not in available_tools, f"read_resource was not excluded: {available_tools}"

        # Ensure the model behaves according to the restrictions
        prompt = "Using the `list_resources` tool, list all resources. If you can't, say exactly 'I cannot list resources'."
        
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
    
    called_list = False
    
    parts = response.candidates[0].content.parts
    for part in parts:
        if part.function_response and part.function_response.name == "list_resources":
            called_list = True
        if part.function_call and part.function_call.name == "list_resources":
            called_list = True
            
    assert not called_list, "Model called `list_resources` even though it was excluded!"
    assert "cannot list resources" in response.text.lower() or "not available" in response.text.lower(), "Model should admit it cannot list resources."
