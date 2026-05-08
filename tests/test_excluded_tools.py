"""
Tests the 'X-MCP-Excluded-Tools' header functionality.

This test verifies two layers of security:
1. The proxy must remove the excluded tool from the tool declarations sent to Google API.
2. The model must not execute the excluded tool, even when prompted to do so.
"""
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import pytest
from google import genai
from fastapi.testclient import TestClient

class FakeGoogleHandler(BaseHTTPRequestHandler):
    intercepted_requests = []
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        FakeGoogleHandler.intercepted_requests.append(json.loads(post_data.decode('utf-8')))
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "I cannot use this tool because it is disabled."}]}}],
            "modelVersion": "gemini-3.1-flash-lite-preview"
        }).encode('utf-8'))
    def log_message(self, format, *args): pass

@pytest.fixture(scope="module")
def fake_google_server():
    server = HTTPServer(('127.0.0.1', 9994), FakeGoogleHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield
    server.shutdown()
    server.server_close()

def test_excluded_tools(fake_google_server):
    FakeGoogleHandler.intercepted_requests.clear()
    
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")
    
    # Exclude the calculation tool
    excluded = ["calculate_expression"]
    excluded_header = base64.b64encode(json.dumps(excluded).encode("utf-8")).decode("utf-8")

    # STEP 1: Intercept the raw payload going to Google via local TestClient
    os.environ["GEMINI_BASE_URL"] = "http://127.0.0.1:9994"
    import app.api
    importlib.reload(app.api)
    from main import app as proxy_app
    proxy_client = TestClient(proxy_app)
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Using the calculate_expression tool, calculate 2+2."}]}]
    }
    
    response_fake = proxy_client.post(
        "/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
        json=payload,
        headers={"x-goog-api-key": "fake", "x-mcp-servers": mcp_header, "x-mcp-excluded-tools": excluded_header}
    )
    assert response_fake.status_code == 200
    
    # VERIFICATION 1: The tool must not be present in the 'tools' array sent to Google
    assert len(FakeGoogleHandler.intercepted_requests) == 1
    google_payload = FakeGoogleHandler.intercepted_requests[0]
    
    if "tools" in google_payload:
        for tool_obj in google_payload["tools"]:
            for func_decl in tool_obj.get("functionDeclarations", []):
                assert func_decl["name"] != "calculate_expression", "The tool 'calculate_expression' was sent to Google despite being excluded!"
                
    # STEP 2: Make a real request to ensure the model cannot call it
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

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents="Using the calculate_expression tool, calculate 2+2. Can you do this?",
    )
    
    # VERIFICATION 2: The model must not emit a function_call for the excluded tool
    for part in response.candidates[0].content.parts:
        if part.function_call:
            assert part.function_call.name != "calculate_expression", "The model successfully called an excluded tool!"
