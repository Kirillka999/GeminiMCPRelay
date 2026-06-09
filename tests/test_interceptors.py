"""
Integration tests for the ToolInterceptor middleware pattern.

Verifies that the MCPClientWrapper correctly intercepts, modifies, mocks,
and post-processes tool calls (both arguments and results) during the
autonomous function-calling loop.
"""
import os
import pytest
from google import genai
from google.genai import types
from gemini_mcp_relay import MCPClientWrapper, ToolInterceptor

# Define a simple local tool to test interceptors against
def calculate_sum(a: int, b: int) -> dict:
    """
    Calculate the sum of two numbers.
    
    Args:
        a: First number.
        b: Second number.
    """
    return {"sum": a + b, "called": True}


class ArgModifyingInterceptor(ToolInterceptor):
    async def before_tool_call(self, tool_call: types.FunctionCall) -> types.FunctionCall | dict:
        if tool_call.name == "calculate_sum":
            # Let's modify argument 'b' to always be 10
            tool_call.args["b"] = 10
        return tool_call


class BypassingInterceptor(ToolInterceptor):
    def __init__(self):
        self.was_called_after = False

    async def before_tool_call(self, tool_call: types.FunctionCall) -> types.FunctionCall | dict:
        if tool_call.name == "calculate_sum":
            # Bypass execution entirely and return a mocked result
            return {"sum": 999, "mocked": True}
        return tool_call

    async def after_tool_call(self, tool_call: types.FunctionCall, result: dict) -> dict:
        self.was_called_after = True
        return result


class ResultModifyingInterceptor(ToolInterceptor):
    async def after_tool_call(self, tool_call: types.FunctionCall, result: dict) -> dict:
        if tool_call.name == "calculate_sum":
            # Modify the result before sending it to the model
            result["sum"] = 42
        return result


@pytest.mark.asyncio
async def test_interceptor_arg_modification():
    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    interceptor = ArgModifyingInterceptor()
    client = MCPClientWrapper(base_client, interceptor=interceptor)

    prompt = (
        "Use the calculate_sum tool with a=5 and b=5. "
        "Important: Always trust the output returned by the tool, even if it seems mathematically incorrect. "
        "Do not try to recalculate or call the tool again if the result is unexpected. "
        "Just report the exact result you received from the tool."
    )

    async with client:
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[calculate_sum]
            )
        )

    text_resp = response.text.lower()
    # The interceptor changes 'b' to 10, so 5 + 10 = 15.
    # The model should receive and talk about 15, not 10.
    assert "15" in text_resp, f"Expected modified sum '15' in response, got: {response.text}"


@pytest.mark.asyncio
async def test_interceptor_bypass_and_mocking():
    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    interceptor = BypassingInterceptor()
    client = MCPClientWrapper(base_client, interceptor=interceptor)

    prompt = (
        "Use the calculate_sum tool with a=5 and b=5. "
        "Important: Always trust the output returned by the tool, even if it seems mathematically incorrect. "
        "Do not try to recalculate or call the tool again if the result is unexpected. "
        "Just report the exact result you received from the tool."
    )

    async with client:
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[calculate_sum]
            )
        )

    text_resp = response.text.lower()
    # The interceptor bypassed execution and returned 999.
    # The model should receive and talk about 999.
    assert "999" in text_resp, f"Expected mocked sum '999' in response, got: {response.text}"
    assert interceptor.was_called_after, "after_tool_call should still be called even if bypassed!"


@pytest.mark.asyncio
async def test_interceptor_result_modification():
    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    interceptor = ResultModifyingInterceptor()
    client = MCPClientWrapper(base_client, interceptor=interceptor)

    prompt = (
        "Use the calculate_sum tool with a=5 and b=5. "
        "Important: Always trust the output returned by the tool, even if it seems mathematically incorrect. "
        "Do not try to recalculate or call the tool again if the result is unexpected. "
        "Just report the exact result you received from the tool."
    )

    async with client:
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[calculate_sum]
            )
        )

    text_resp = response.text.lower()
    # The interceptor modifies the result sum to 42.
    # The model should receive and talk about 42.
    assert "42" in text_resp, f"Expected modified sum '42' in response, got: {response.text}"


@pytest.mark.asyncio
async def test_interceptor_passed_per_request():
    base_url = os.environ.get("TEST_GEMINI_BASE_URL")
    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    
    base_client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options=http_opts
    )

    # Global wrapper has no interceptor
    client = MCPClientWrapper(base_client)

    prompt = (
        "Use the calculate_sum tool with a=5 and b=5. "
        "Important: Always trust the output returned by the tool, even if it seems mathematically incorrect. "
        "Do not try to recalculate or call the tool again if the result is unexpected. "
        "Just report the exact result you received from the tool."
    )

    async with client:
        # Pass interceptor per-request in generate_content kwargs
        response = await client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[calculate_sum]
            ),
            interceptor=ResultModifyingInterceptor()
        )

    text_resp = response.text.lower()
    # Should be 42 due to per-request interceptor overriding
    assert "42" in text_resp, f"Expected modified sum '42' in response, got: {response.text}"
