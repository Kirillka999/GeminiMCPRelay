"""
Tests the preservation of 'thoughtSignature' for reasoning models.

Reasoning models (e.g., Pro, Flash-Thinking) return binary 'thoughtSignature' 
hashes alongside their parts. This test ensures that these binary hashes 
survive the squashing and unsquashing lifecycle and are correctly serialized 
to Base64 without crashing the JSON parser.
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
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Fake response"}]}}],
            "modelVersion": "gemini-3.1-pro-preview"
        }).encode('utf-8'))
    def log_message(self, format, *args): pass

@pytest.fixture(scope="module")
def fake_google_server():
    server = HTTPServer(('127.0.0.1', 9993), FakeGoogleHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield
    server.shutdown()
    server.server_close()

def test_thought_signature_preservation(fake_google_server):
    FakeGoogleHandler.intercepted_requests.clear()
    
    mcp_config = {"math_server": {"url": "https://mathematics.fastmcp.app/mcp"}}
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")

    # STEP 1: Real request to the PRO model via ngrok
    client = genai.Client(
        api_key=os.environ.get("TEST_GEMINI_API_KEY"),
        http_options={
            "base_url": os.environ.get("TEST_GEMINI_BASE_URL"),
            "headers": {"x-mcp-servers": mcp_header}
        }
    )

    chat = client.chats.create(model="gemini-3.1-pro-preview")
    chat.send_message("Using the calculate_expression tool, calculate 2+2. Output the answer.")
    
    # Find thoughtSignature in the squashed history (returned by the proxy)
    found_signatures_in_sdk = []
    for msg in chat.get_history():
        if msg.parts:
            for p in msg.parts:
                # In the Python SDK, thought_signature is exposed as raw 'bytes'
                if p.thought_signature:
                    found_signatures_in_sdk.append(p.thought_signature)
                    
    assert len(found_signatures_in_sdk) > 0, "Proxy did not return thoughtSignatures to the client (or the PRO model omitted them)!"
    
    # STEP 2: Intercept the payload SDK forms for the next chat turn
    client_fake = genai.Client(
        api_key="fake",
        http_options={
            "base_url": "http://127.0.0.1:9993",
            "headers": {"x-mcp-servers": mcp_header}
        }
    )
    
    chat_fake = client_fake.chats.create(model="gemini-3.1-pro-preview", history=chat.get_history())
    chat_fake.send_message("Great!")
    
    sdk_payload = FakeGoogleHandler.intercepted_requests[0]
    
    # Verify that the SDK embedded the signatures as Base64 strings in the payload
    found_signatures_in_payload = []
    for content in sdk_payload["contents"]:
        for part in content.get("parts", []):
            if "thoughtSignature" in part:
                found_signatures_in_payload.append(part["thoughtSignature"])
                
    assert len(found_signatures_in_payload) > 0, "The official SDK did not attach thoughtSignatures to the new request!"

    # STEP 3: Route the SDK payload through the local proxy parser
    os.environ["GEMINI_BASE_URL"] = "http://127.0.0.1:9993"
    import app.api
    importlib.reload(app.api)
    from main import app as proxy_app
    proxy_client = TestClient(proxy_app)
    
    proxy_client.post(
        "/v1beta/models/gemini-3.1-pro-preview:generateContent",
        json=sdk_payload,
        headers={"x-goog-api-key": "fake", "x-mcp-servers": mcp_header}
    )
    
    # STEP 4: Verify the final Unsquashed payload sent to Google
    proxy_payload = FakeGoogleHandler.intercepted_requests[1]
    
    found_signatures_in_final = []
    for content in proxy_payload["contents"]:
        for part in content.get("parts", []):
            if "thoughtSignature" in part:
                found_signatures_in_final.append(part["thoughtSignature"])
                
    assert len(found_signatures_in_final) > 0, "Function convert_bytes_to_b64 (or unsquash_contents) lost the thoughtSignatures!"
    
    # Ensure the exact same number of signatures is preserved after unsquashing
    assert len(found_signatures_in_payload) == len(found_signatures_in_final), "The number of thoughtSignatures changed after unsquashing the history!"
