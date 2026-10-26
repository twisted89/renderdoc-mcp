"""Dedicated replay thread for all RenderDoc API calls.

mcp 2.x runs synchronous tool handlers on anyio worker threads, potentially
several at once. RenderDoc's ReplayController is not thread-safe and its
GPU/driver contexts are tied to the thread that created them, and tools also
share mutable state (the current event set by SetFrameEvent). So every call
into RenderDoc is funnelled through one long-lived thread, one call at a time,
while the asyncio event loop stays free to service the MCP protocol.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ReplayThread:
    """A single daemon thread that executes submitted callables in order."""

    def __init__(self, name: str = "renderdoc-replay"):
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fut, fn, args, kwargs = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # propagate to the awaiting caller
                fut.set_exception(exc)
            else:
                fut.set_result(result)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        self._queue.put((fut, fn, args, kwargs))
        return fut

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run fn on the replay thread and block until it finishes."""
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        return self.submit(fn, *args, **kwargs).result()

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run fn on the replay thread and await its result without blocking the event loop."""
        return await asyncio.wrap_future(self.submit(fn, *args, **kwargs))

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()


_replay_thread: ReplayThread | None = None
_lock = threading.Lock()


def get_replay_thread() -> ReplayThread:
    global _replay_thread
    with _lock:
        if _replay_thread is None:
            _replay_thread = ReplayThread()
        return _replay_thread


def on_replay_thread(fn: Callable[..., T]) -> Callable[..., Any]:
    """Wrap a synchronous function as an async one that runs on the replay thread.

    functools.wraps keeps __wrapped__, so MCPServer still derives the tool's
    name, description and input schema from the original signature.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await get_replay_thread().run(fn, *args, **kwargs)

    return wrapper
