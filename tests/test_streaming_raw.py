"""
Tests the raw Server-Sent Events (SSE) streaming format.

Verifies that the proxy correctly streams events in the official format,
and ensures that synthetic `functionResponse` chunks are injected directly 
into the SSE stream so that the client SDK is aware of internal tool executions.
"""
import base64
import json
import os
import requests

def test_math_mcp_raw_streaming_sse():
    mcp_config = {
        "math_server": {
            "url": "https://mathematics.fastmcp.app/mcp"
        }
    }
    mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")
    
    url = f"{os.environ.get('TEST_GEMINI_BASE_URL')}/v1beta/models/gemini-3.1-flash-lite-preview:streamGenerateContent"
    headers = {
        "x-goog-api-key": os.environ.get("TEST_GEMINI_API_KEY"),
        "x-mcp-servers": mcp_header,
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": "Using the calculate_expression tool, calculate 200 + 300. Reply with just the number."}]
        }]
    }

    # Request with stream=True for manual SSE reading
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    assert response.status_code == 200
    
    found_call = False
    found_response = False
    found_text = False
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            
            # Validate standard SSE format
            assert decoded_line.startswith("data: ")
            
            json_str = decoded_line[6:] # Strip "data: " prefix
            chunk_data = json.loads(json_str)
            
            candidates = chunk_data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts"):
                parts = candidates[0]["content"]["parts"]
                for part in parts:
                    if "functionCall" in part:
                        found_call = True
                    elif "functionResponse" in part:
                        found_response = True
                    elif "text" in part:
                        found_text = True
                        
    assert found_call, "SSE stream is missing a chunk with 'functionCall'"
    assert found_response, "SSE stream is missing a synthetic chunk with 'functionResponse'"
    assert found_text, "SSE stream is missing a chunk with final 'text'"
