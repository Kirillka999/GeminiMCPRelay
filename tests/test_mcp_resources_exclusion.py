"""
Tests the proxy's capability to correctly exclude synthetic MCP resource tools.

Verifies that if a user passes `list_resources` and/or `read_resource` (or their prefixed equivalents) 
in the `X-MCP-Excluded-Tools` header, the proxy successfully hides them from the model.
The model should not be able to call them even if prompted to do so.
"""
import base64
import json
import os
import pytest
from google import genai
from google.genai import types

def test_mcp_resources_exclusion():
    mcp_config = {
        "Exa Search": {
            "url": "https://mcp.exa.ai/mcp"
        }
    }
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")
    
    excluded_tools = ["list_resources", "read_resource"]
    excluded_header = base64.b64encode(json.dumps(excluded_tools).encode("utf-8")).decode("utf-8")

    client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options={
            "base_url": os.environ.get("TEST_GEMINI_BASE_URL"),
            "headers": {
                "x-mcp-servers": mcp_header,
                "x-mcp-excluded-tools": excluded_header
            }
        }
    )

    prompt = "Using the `list_resources` tool, list all resources. If you can't, say exactly 'I cannot list resources'."
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
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
