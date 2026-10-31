import time
import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class BufferedTask:
    def __init__(self, task_id: str, ttl_seconds: float = 3600.0):
        self.task_id = task_id
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.expires_at = self.created_at + ttl_seconds
        
        # Buffer of streamed events/chunks (in order)
        self.chunks: List[dict] = []
        # Flag indicating if execution has finished
        self.is_completed: bool = False
        self.error: Optional[str] = None
        
        # Async event to notify waiting WebSocket subscribers of new chunks
        self.update_event: asyncio.Event = asyncio.Event()
        # Acknowledgements from client: set of chunk indices or "done"
        self.acked_indices: set[int] = set()
        self.is_acked_by_client: bool = False

    def add_chunk(self, chunk_data: dict):
        self.chunks.append(chunk_data)
        self.update_event.set()
        self.update_event = asyncio.Event()

    def complete(self):
        self.is_completed = True
        self.update_event.set()
        self.update_event = asyncio.Event()

    def fail(self, error_message: str):
        self.error = error_message
        self.is_completed = True
        self.update_event.set()
        self.update_event = asyncio.Event()

    def acknowledge(self, chunk_idx: Optional[int] = None, final: bool = False):
        if chunk_idx is not None:
            self.acked_indices.add(chunk_idx)
        if final:
            self.is_acked_by_client = True

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class TaskStore:
    def __init__(self, default_ttl: float = 3600.0):
        self.default_ttl = default_ttl
        self.tasks: Dict[str, BufferedTask] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, task_id: str) -> BufferedTask:
        async with self._lock:
            self._cleanup_expired()
            if task_id not in self.tasks:
                self.tasks[task_id] = BufferedTask(task_id, self.default_ttl)
                logger.info(f"📦 [TaskStore] Created buffered task '{task_id}' (TTL: {self.default_ttl}s)")
            return self.tasks[task_id]

    async def get(self, task_id: str) -> Optional[BufferedTask]:
        async with self._lock:
            self._cleanup_expired()
            return self.tasks.get(task_id)

    async def remove(self, task_id: str):
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"🗑️ [TaskStore] Removed acknowledged task '{task_id}'")

    def _cleanup_expired(self):
        now = time.time()
        expired_ids = [
            tid for tid, task in self.tasks.items()
            if task.is_expired() or (task.is_completed and task.is_acked_by_client and now > task.created_at + 300)
        ]
        for tid in expired_ids:
            del self.tasks[tid]
            logger.info(f"🧹 [TaskStore] Expired task '{tid}' evicted from memory")


# Global singleton instance
task_store = TaskStore(default_ttl=3600.0)
