import argparse
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from gemini_mcp_relay.server.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemini MCP Relay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

def start():
    parser = argparse.ArgumentParser(description="Start the Gemini MCP Relay proxy server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    args = parser.parse_args()

    logger.info(f"👾 Starting Gemini MCP Relay on {args.host}:{args.port}...")
    uvicorn.run("gemini_mcp_relay.server.main:app", host=args.host, port=args.port)

if __name__ == "__main__":
    start()
