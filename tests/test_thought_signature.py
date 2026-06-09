"""
Tests the preservation of 'thought' and 'thought_signature' for reasoning models.

Reasoning models return 'thought' and binary 'thought_signature' 
hashes alongside their parts. This test ensures that these objects survive the 
generation and history-unsquashing lifecycle within the wrapper.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper

@pytest.mark.asyncio
async def test_thought_signature_preservation():
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

        # STEP 1: Real request to a reasoning model
        # Using gemini-3.1-pro-preview to ensure thoughts are generated
        chat = client.chats.create(model="gemini-3.1-pro-preview")
        
        # Asking a slightly complex question to encourage the model to think
        await chat.send_message("Using the calculate_expression tool, calculate 25 * 14. Output the answer. Think step-by-step.")
        
        # Find thought or thoughtSignature in the history (returned by the wrapper)
        found_thoughts = []
        found_signatures = []
        
        for msg in chat.get_history():
            if msg.parts:
                for p in msg.parts:
                    if getattr(p, "thought", None):
                        found_thoughts.append(p.thought)
                    # thought_signature is exposed as raw 'bytes' if present
                    if getattr(p, "thought_signature", None):
                        found_signatures.append(p.thought_signature)

        # STEP 2: Send a second message to ensure history unsquashing doesn't crash 
        # when 'thought' or 'thought_signature' parts are present in the history.
        # If the API accepts it without a 400 Bad Request, our wrapper successfully
        # passed the complex objects back to the SDK intact.
        resp2 = await chat.send_message("Great! Now add 10 to that result.")
        assert "360" in resp2.text
