import os
import json
import logging
import base64
import asyncio
import httpx
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

from gemini_mcp_relay.mcp_manager import MCPConnectionManager
from gemini_mcp_relay.formatters import parse_request_payload, convert_bytes_to_b64
from gemini_mcp_relay.orchestrator import generate_content_loop, stream_generate_content_loop
from gemini_mcp_relay.server.task_store import task_store, BufferedTask

logger = logging.getLogger(__name__)
router = APIRouter()

def _get_base_url(request: Request) -> str:
    base_url_header = request.headers.get("x-base-url")
    if not base_url_header:
        raise HTTPException(status_code=400, detail="Missing x-base-url header")
    try:
        decoded_bytes = base64.b64decode(base_url_header)
        base_url = decoded_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode x-base-url header: {str(e)}")
    if not base_url.strip():
        raise HTTPException(status_code=400, detail="The decoded x-base-url cannot be empty")
    return base_url.strip()

@router.get("/v1/mcp/tools")
async def list_mcp_tools(request: Request):
    """
    Returns a list of all available tools from the configured MCP servers.
    The response format is designed for frontend consumption, providing
    server names, tool names, descriptions, and their JSON schemas.
    """
    mcp_header = request.headers.get("x-mcp-servers")
    if not mcp_header:
        return {"tools": []}
        
    try:
        manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header)
        try:
            await manager.connect_all_from_config(parsed_config, ignore_connection_errors=True)
            return {"tools": manager.raw_tools}
        finally:
            await manager.close()
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _initialize_client_and_config(api_key: str, base_url: str, payload: dict, mcp_declarations: list) -> tuple:
    contents, config = parse_request_payload(payload)

    if mcp_declarations:
        mcp_tool = types.Tool(function_declarations=mcp_declarations)
        if config.tools is None:
            config.tools = []
        config.tools.append(mcp_tool)

    http_opts = types.HttpOptions(base_url=base_url) if base_url else None
    client = genai.Client(
        api_key=api_key,
        http_options=http_opts
    )
    return client, contents, config

async def handle_generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    base_url = _get_base_url(request)

    mcp_header = request.headers.get("x-mcp-servers")
    excluded_header = request.headers.get("x-mcp-excluded-tools")
    
    manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header, excluded_header)
    try:
        await manager.connect_all_from_config(parsed_config, ignore_connection_errors=True)
        
        payload = await request.json()
        client, contents, config = _initialize_client_and_config(api_key, base_url, payload, manager.mcp_declarations)
        
        response_obj = await generate_content_loop(client, model_name, contents, config, manager.adapters_map)
        from gemini_mcp_relay.formatters import convert_bytes_to_b64
        response_dict = response_obj.model_dump(exclude_none=True, by_alias=True)
        return convert_bytes_to_b64(response_dict)
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return {
            "candidates": [
                {
                    "index": 0,
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [{"text": f"⚠️ **Error:** {str(e)}"}]
                    }
                }
            ]
        }
    finally:
        await manager.close()

async def with_heartbeat(generator, interval: float = 10.0):
    """Wraps an async generator to yield SSE comment pings (: ping\\n\\n) if no data is produced for interval seconds."""
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    async def producer():
        try:
            async for item in generator:
                await queue.put(item)
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(done)

    task = asyncio.create_task(producer())

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=interval)
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        except asyncio.TimeoutError:
            yield ": ping\n\n"

    await task

async def handle_stream_generate_content(model_name: str, request: Request):
    api_key = request.headers.get("x-goog-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    base_url = _get_base_url(request)

    mcp_header = request.headers.get("x-mcp-servers")
    excluded_header = request.headers.get("x-mcp-excluded-tools")
    
    payload = await request.json()

    async def event_generator():
        manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header, excluded_header)
        from gemini_mcp_relay.formatters import convert_bytes_to_b64
        try:
            await manager.connect_all_from_config(parsed_config, ignore_connection_errors=True)
            client, contents, config = _initialize_client_and_config(api_key, base_url, payload, manager.mcp_declarations)
            
            async for chunk_obj in stream_generate_content_loop(client, model_name, contents, config, manager.adapters_map):
                chunk_dict = chunk_obj.model_dump(exclude_none=True, by_alias=True)
                convert_bytes_to_b64(chunk_dict)
                yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}", exc_info=True)
            synthetic_error_chunk = {
                "candidates": [
                    {
                        "index": 0,
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": f"\n\n⚠️ **Error:** {str(e)}"}]
                        }
                    }
                ]
            }
            yield f"data: {json.dumps(synthetic_error_chunk, ensure_ascii=False)}\n\n"
            error_json = {"error": {"code": 500, "message": str(e), "status": "INTERNAL"}}
            yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"
        finally:
            await manager.close()

    return StreamingResponse(with_heartbeat(event_generator(), interval=10.0), media_type="text/event-stream")

_shared_proxy_client: httpx.AsyncClient | None = None

def get_proxy_client() -> httpx.AsyncClient:
    global _shared_proxy_client
    if _shared_proxy_client is None:
        _shared_proxy_client = httpx.AsyncClient()
    return _shared_proxy_client

@router.on_event("shutdown")
async def shutdown_event():
    global _shared_proxy_client
    if _shared_proxy_client is not None:
        await _shared_proxy_client.aclose()
        _shared_proxy_client = None

async def transparent_proxy(request: Request, full_path: str):
    target_base = _get_base_url(request)
    target_url = f"{target_base.rstrip('/')}/{full_path}"
    
    query_params = request.url.query
    if query_params:
        target_url += f"?{query_params}"
        
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            headers[k] = v
            
    body = await request.body()
    
    client = get_proxy_client()
    resp = None
    try:
        req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body
        )
        resp = await client.send(req, stream=True)
        
        resp_headers = dict(resp.headers)
        resp_headers.pop("content-encoding", None)
        resp_headers.pop("content-length", None)
        
        async def response_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
            
        return StreamingResponse(
            response_generator(),
            status_code=resp.status_code,
            headers=resp_headers
        )
    except Exception as e:
        if resp is not None:
            await resp.aclose()
        logger.error(f"Failed to send proxy request: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")

async def _run_background_generation(
    task: BufferedTask,
    api_key: str,
    base_url: str,
    model_name: str,
    mcp_header: Optional[str],
    excluded_header: Optional[str],
    payload: dict
):
    """Executes the AI model and tool calling in background, writing chunks to the task buffer."""
    manager, parsed_config = MCPConnectionManager.from_http_headers(mcp_header or "", excluded_header or "")
    try:
        await manager.connect_all_from_config(parsed_config, ignore_connection_errors=True)
        client, contents, config = _initialize_client_and_config(api_key, base_url, payload, manager.mcp_declarations)

        async for chunk_obj in stream_generate_content_loop(client, model_name, contents, config, manager.adapters_map):
            chunk_dict = chunk_obj.model_dump(exclude_none=True, by_alias=True)
            convert_bytes_to_b64(chunk_dict)
            task.add_chunk(chunk_dict)
        
        task.complete()
        logger.info(f"✅ [TaskStore] Task '{task.task_id}' finished successfully ({len(task.chunks)} chunks cached)")
    except Exception as e:
        logger.error(f"❌ [TaskStore] Task '{task.task_id}' failed: {e}", exc_info=True)
        err_msg = str(e)
        synthetic_error_chunk = {
            "candidates": [
                {
                    "index": 0,
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [{"text": f"\n\n⚠️ **Error:** {err_msg}"}]
                    }
                }
            ]
        }
        task.add_chunk(synthetic_error_chunk)
        task.fail(err_msg)
    finally:
        await manager.close()


@router.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket generation endpoint with resilient reconnection and 1-hour buffer preservation.
    Even if the client disconnects, generation proceeds in the background. Client can reconnect
    with the same task_id / turn_id and resume receiving buffered chunks.
    """
    await websocket.accept()
    logger.info("🔌 WebSocket client connected to /ws/generate")

    task_id = "unknown"
    try:
        # 1. Read initial initiation message
        init_data = await websocket.receive_json()
        action = init_data.get("action", "start")
        task_id = init_data.get("task_id") or init_data.get("turn_id")

        if not task_id:
            await websocket.send_json({"type": "error", "message": "Missing 'task_id' or 'turn_id'"})
            await websocket.close(code=1008)
            return

        task = await task_store.get_or_create(task_id)

        if action == "start" and len(task.chunks) == 0 and not task.is_completed:
            # New task: extract parameters and launch background worker
            api_key = init_data.get("api_key")
            base_url = init_data.get("base_url") or "https://universal-gateway.ainetwork.qzz.io"
            model_name = init_data.get("model_name", "gemini-smartest")
            mcp_header = init_data.get("mcp_servers")
            excluded_header = init_data.get("mcp_excluded_tools")
            payload = init_data.get("payload", {})

            if not api_key:
                await websocket.send_json({"type": "error", "message": "Missing 'api_key'"})
                await websocket.close(code=1008)
                return

            # Start detached background execution
            asyncio.create_task(
                _run_background_generation(
                    task=task,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    mcp_header=mcp_header,
                    excluded_header=excluded_header,
                    payload=payload
                )
            )

        # 2. Concurrently stream chunks to client while listening for ACK / ping messages
        last_sent_index = int(init_data.get("resume_from_index", 0))

        async def send_stream():
            nonlocal last_sent_index
            while True:
                # Send all newly available chunks
                while last_sent_index < len(task.chunks):
                    chunk = task.chunks[last_sent_index]
                    await websocket.send_json({
                        "type": "chunk",
                        "task_id": task_id,
                        "index": last_sent_index,
                        "data": chunk
                    })
                    last_sent_index += 1

                if task.is_completed and last_sent_index >= len(task.chunks):
                    await websocket.send_json({
                        "type": "done",
                        "task_id": task_id,
                        "total_chunks": len(task.chunks),
                        "error": task.error
                    })
                    # Keep sender alive waiting for final ack
                    while not task.is_acked_by_client:
                        await asyncio.sleep(0.1)
                    break

                # Wait for next chunk event or completion
                await task.update_event.wait()

        # Start sender task
        sender_task = asyncio.create_task(send_stream())

        # Let receiver listen until client sends final ACK or disconnects
        try:
            # Reset client ack on new session/reconnection
            task.is_acked_by_client = False
            while not task.is_acked_by_client:
                msg = await websocket.receive_json()
                msg_type = msg.get("type") or msg.get("action")
                if msg_type == "ack":
                    chunk_idx = msg.get("index")
                    final = msg.get("final", False)
                    task.acknowledge(chunk_idx, final=final)
                    if final:
                        logger.info(f"💌 [TaskStore] Final ACK received from client for task '{task_id}'")
                        break
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
            
            # Allow sender task to finish gracefully
            await asyncio.wait_for(sender_task, timeout=2.0)
        except (WebSocketDisconnect, asyncio.CancelledError, asyncio.TimeoutError):
            pass
        finally:
            sender_task.cancel()

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for task '{task_id}'. Task continues running in background for 1 hour.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_gateway(request: Request, full_path: str):
    """
    Catch-all router that intercepts generation endpoints and transparently proxies everything else.
    """
    if request.method == "POST":
        if full_path.endswith(":generateContent"):
            model_name = full_path.split("/")[-1].split(":")[0]
            return await handle_generate_content(model_name, request)
            
        elif full_path.endswith(":streamGenerateContent"):
            model_name = full_path.split("/")[-1].split(":")[0]
            return await handle_stream_generate_content(model_name, request)
            
    return await transparent_proxy(request, full_path)
