"""
Tests the proxy's capability to inject and correctly handle synthetic MCP resource tools.

Verifies that if an MCP server supports resources:
1. `list_resources` and `read_resource` tools are added.
2. The model can successfully call `list_resources`.
3. The model can successfully call `read_resource` with a returned URI.
"""
import base64
import json
import os
from google import genai
from google.genai import types

def test_mcp_resources():
    mcp_config = {
        "Exa Search": {
            "url": "https://mcp.exa.ai/mcp"
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
        "Using the `list_resources` tool, find the available resources on the Exa Search server. "
        "Find the resource related to the list of tools"
        "and then use the `read_resource` tool to read its content. "
        "Output the raw JSON content of that resource exactly as it was returned to you."
    )
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )

    called_list = False
    called_read = False
    
    parts = response.candidates[0].content.parts
    for part in parts:
        if part.function_response:
            if part.function_response.name == "list_resources":
                called_list = True
                assert "result" in part.function_response.response, "Expected 'result' in list_resources response"
            elif part.function_response.name == "read_resource":
                called_read = True
                assert "result" in part.function_response.response, "Expected 'result' in read_resource response"
                result_str = str(part.function_response.response["result"])
                assert "web_search_exa" in result_str or "web_fetch_exa" in result_str, "Expected tool definitions in the resource content"

    assert called_list, "The model failed to call `list_resources`."
    assert called_read, "The model failed to call `read_resource`."
