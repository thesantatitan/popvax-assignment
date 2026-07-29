"""Bounded latest-value helpers for multiprocessing queues."""

from __future__ import annotations

import queue
from multiprocessing.queues import Queue
from typing import TypeVar

T = TypeVar("T")


def put_latest(channel: Queue, item: T) -> None:
    """Put an item without blocking, discarding stale queued work."""

    try:
        channel.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        channel.get_nowait()
    except queue.Empty:
        pass
    try:
        channel.put_nowait(item)
    except queue.Full:
        # Another producer won the race. Its value is at least as fresh.
        pass


def drain_latest(channel: Queue) -> T | None:
    """Return only the newest currently available item."""

    latest: T | None = None
    while True:
        try:
            latest = channel.get_nowait()
        except queue.Empty:
            return latest
