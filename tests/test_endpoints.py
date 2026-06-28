"""
Tests for HTTP endpoints and request validation.

Validates the behavior of auxiliary routes and proxy security features:
- GET /v1/mcp/tools: Returns a unified list of tool schemas.
- 401 Unauthorized handling for missing API keys.
- 400 Bad Request handling for invalid Base64 headers.
"""
import base64
import json
from fastapi.testclient import TestClient
from gemini_mcp_relay.server.main import app

client = TestClient(app)

def test_get_mcp_tools():
    mcp_config = {
        "math_server": {
            "url": "https://mathematics.fastmcp.app/mcp"
        }
    }
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")
    
    response = client.get("/v1/mcp/tools", headers={"x-mcp-servers": mcp_header})
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
    
    data = response.json()
    assert "tools" in data, "Response is missing 'tools' key"
    
    tool_names = [t["name"] for t in data["tools"]]
    assert "calculate_expression" in tool_names, f"Tool 'calculate_expression' not found. Available tools: {tool_names}"

def test_auth_missing_key():
    response = client.post("/v1beta/models/gemini-3.5-flash:generateContent", json={})
    assert response.status_code == 401, f"Expected status 401, got {response.status_code}"
    assert "Missing API Key" in response.json()["detail"]

def test_invalid_mcp_header():
    # Pass a valid x-base-url so it gets past base_url validation to the mcp header validation.
    base_url_b64 = base64.b64encode(b"https://example.com").decode("utf-8")
    response = client.post(
        "/v1beta/models/gemini-3.5-flash:generateContent", 
        headers={
            "x-goog-api-key": "fake_key", 
            "x-mcp-servers": "invalid_base64_string!",
            "x-base-url": base_url_b64
        },
        json={"contents": []}
    )
    assert response.status_code == 400, f"Expected status 400, got {response.status_code}"

def test_missing_base_url_header():
    response = client.post(
        "/v1beta/models/gemini-3.5-flash:generateContent", 
        headers={"x-goog-api-key": "fake_key"},
        json={"contents": []}
    )
    assert response.status_code == 400, f"Expected status 400, got {response.status_code}"
    assert "Missing x-base-url header" in response.json()["detail"]

def test_invalid_base_url_header():
    response = client.post(
        "/v1beta/models/gemini-3.5-flash:generateContent", 
        headers={"x-goog-api-key": "fake_key", "x-base-url": "not-base-64!!"},
        json={"contents": []}
    )
    assert response.status_code == 400, f"Expected status 400, got {response.status_code}"
    assert "Failed to decode x-base-url header" in response.json()["detail"]

def test_empty_base_url_header():
    empty_b64 = base64.b64encode(b"   ").decode("utf-8")
    response = client.post(
        "/v1beta/models/gemini-3.5-flash:generateContent", 
        headers={"x-goog-api-key": "fake_key", "x-base-url": empty_b64},
        json={"contents": []}
    )
    assert response.status_code == 400, f"Expected status 400, got {response.status_code}"
    assert "The decoded x-base-url cannot be empty" in response.json()["detail"]
