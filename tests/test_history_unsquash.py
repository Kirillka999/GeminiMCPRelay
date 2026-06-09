"""
Tests the proxy's 'unsquashing' logic for chat history.

The orchestrator compresses previous tool calls and responses into a single 'model' 
turn (squashing) so the SDK doesn't choke on them. This test verifies that the wrapper 
correctly parses this squashed history on subsequent turns and unrolls it back into the strict 
'user -> model -> user -> model' sequence required by the official Google API.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_real_chat_history_unsquashing():
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}

    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    client = MCPClientWrapper(base_client)

    async with client:
        await client.mcp.add_server("math_server", mcp_config["math_server"])

        # STEP 1: Perform a real request to generate an actual tool execution history
        chat = client.chats.create(model="gemini-3.5-flash")

        resp1 = await chat.send_message("Using the calculate_expression tool, calculate 2+2. Output only the answer.")
        assert resp1.text.strip() == "4"

        history = chat.get_history()
        # The history stored in the SDK should be squashed: User -> Model (containing call, resp, text)
        assert len(history) == 2, "History should consist of exactly 2 messages (user prompt and squashed model response)"
        
        # STEP 2: Send a second message
        # If unsquashing fails, the Google API will return a 400 Bad Request because the sequence
        # of roles in the history will be invalid. If this succeeds, unsquashing works perfectly.
        resp2 = await chat.send_message("Multiply that result by 10. Output only the answer.")
        assert resp2.text.strip() == "40"
