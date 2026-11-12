"""
src/realtime/queues.py

A bounded queue that drops the OLDEST item when full, rather than blocking or
raising — so the inference thread always processes the most recent frame
available and never builds up a stale backlog.

Per docs/instructions/05_TEMPORAL_SMOOTHING_AND_REALTIME.md §B2 and
PHASE_5_KICKOFF_PROMPT.md §9 (Task 8).

Design note — why not queue.Queue(maxsize) with block=False?
  queue.Queue.put_nowait raises queue.Full when the queue is full, which would
  require the caller to then manually drain the oldest item and retry — i.e.
  exactly what DropOldestQueue encapsulates.  The stdlib Queue also offers
  get_nowait, which raises queue.Empty rather than returning a sentinel; for
  the inference loop's "poll without blocking" pattern, returning None is more
  ergonomic.

Thread safety: all operations delegate to queue.Queue which is already
thread-safe; no external locking is needed.
"""
from __future__ import annotations

import queue
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """Bounded FIFO that silently drops the oldest item when full.

    Args:
        maxsize: Maximum number of items to hold.  When a put() would exceed
                 this, the head (oldest) item is removed first to make room.
                 Must be ≥ 1.
    """

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be ≥ 1, got {maxsize}")
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self.drops: int = 0  # cumulative count of items dropped due to overflow

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def put(self, item: T) -> None:
        """Push an item, dropping the oldest if the queue is full.

        This operation is non-blocking: it never waits for space.  If the
        queue is full, the head item is removed first, then the new item is
        inserted.  A drop counter is incremented for diagnostic purposes.
        """
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                # Drain the oldest item to make room, then retry.
                try:
                    self._q.get_nowait()
                    self.drops += 1
                except queue.Empty:
                    # Rare: another thread drained between the Full and here.
                    pass

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def get_nowait_or_none(self) -> T | None:
        """Return the next item without blocking, or None if the queue is empty."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def get(self, timeout: float | None = None) -> T:
        """Blocking get.  Raises queue.Empty if timeout expires."""
        return self._q.get(timeout=timeout)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def qsize(self) -> int:
        """Approximate number of items currently in the queue."""
        return self._q.qsize()

    @property
    def maxsize(self) -> int:
        """Maximum capacity of the queue."""
        return self._maxsize

    def empty(self) -> bool:
        """Return True if the queue appears empty (subject to race conditions)."""
        return self._q.empty()
