"""
Tests the proxy's 'unsquashing' logic for chat history.

The Google SDK compresses previous tool calls and responses into a single 'model' 
turn (squashing). This test verifies that the proxy correctly parses this squashed
history and unrolls it back into the strict 'user -> model -> user -> model' sequence 
required by the official Google API.
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
        payload = json.loads(post_data.decode('utf-8'))
        FakeGoogleHandler.intercepted_requests.append(payload)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Fake response"}]}}],
            "modelVersion": "gemini-3.1-flash-lite-preview"
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
        
    def log_message(self, format, *args): pass

@pytest.fixture(scope="module")
def fake_google_server():
    server = HTTPServer(('127.0.0.1', 9999), FakeGoogleHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield
    server.shutdown()
    server.server_close()

def test_real_chat_history_unsquashing(fake_google_server):
    FakeGoogleHandler.intercepted_requests.clear()
    
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode('utf-8')).decode('utf-8')

    # STEP 1: Perform a real request to generate an actual tool execution history
    client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options={
            "base_url": os.environ.get("TEST_GEMINI_BASE_URL"),
            "headers": {"x-mcp-servers": mcp_header}
        }
    )

    chat = client.chats.create(model="gemini-3.1-flash-lite-preview")
    chat.send_message("Using the calculate_expression tool, calculate 2+2. Output only the answer.")
    
    history = chat.get_history()
    assert len(history) == 2, "History should consist of exactly 2 messages (user prompt and squashed model response)"
    
    # STEP 2: Intercept the squashed payload that the SDK tries to send on the next turn
    client_fake = genai.Client(
        api_key="fake",
        http_options={
            "base_url": "http://127.0.0.1:9999",
            "headers": {"x-mcp-servers": mcp_header}
        }
    )
    chat_fake = client_fake.chats.create(model="gemini-3.1-flash-lite-preview", history=history)
    chat_fake.send_message("Thank you!")
    
    assert len(FakeGoogleHandler.intercepted_requests) == 1
    sdk_payload = FakeGoogleHandler.intercepted_requests[0]
    
    model_turn = sdk_payload["contents"][1]
    assert model_turn["role"] == "model"
    assert len(model_turn["parts"]) > 1, "The SDK should have sent the squashed history (call and response in a single message)"
    
    # STEP 3: Route the intercepted SDK payload through the local proxy
    os.environ["GEMINI_BASE_URL"] = "http://127.0.0.1:9999"
    import gemini_mcp_relay.api
    importlib.reload(gemini_mcp_relay.api) 
    from main import app as proxy_app
    
    proxy_client = TestClient(proxy_app)
    
    response = proxy_client.post(
        "/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
        json=sdk_payload,
        headers={
            "x-goog-api-key": "fake",
            "x-mcp-servers": mcp_header
        }
    )
    assert response.status_code == 200
    
    # STEP 4: Verify the unrolled (unsquashed) sequence sent to Google
    assert len(FakeGoogleHandler.intercepted_requests) == 2
    proxy_payload = FakeGoogleHandler.intercepted_requests[1]
    contents = proxy_payload["contents"]
    
    # Validation of the unrolled sequence:
    # 0. user (Initial prompt)
    # 1. model (functionCall)
    # 2. user (functionResponse)
    # 3. model (text = 4)
    # 4. user (Thank you!)
    assert len(contents) == 5, f"Expected 5 history messages, got {len(contents)}"
    
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert "functionCall" in contents[1]["parts"][0]
    
    assert contents[2]["role"] == "user"
    assert "functionResponse" in contents[2]["parts"][0]
    
    assert contents[3]["role"] == "model"
    assert "text" in contents[3]["parts"][0]
    
    assert contents[4]["role"] == "user"
    assert contents[4]["parts"][0]["text"] == "Thank you!"
