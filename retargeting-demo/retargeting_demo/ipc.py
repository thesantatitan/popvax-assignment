"""Bounded latest-value helpers for multiprocessing queues."""

from __future__ import annotations

import ctypes
import os
import queue
import signal
import sys
from multiprocessing.queues import Queue
from typing import TypeVar

T = TypeVar("T")


def configure_parent_death_signal() -> int:
    """Ask Linux to terminate this worker if its parent process disappears."""

    parent_pid = os.getppid()
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None)
        if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
            raise OSError("Could not configure the worker parent-death signal")
        if os.getppid() != parent_pid:
            raise SystemExit("Worker parent exited during startup")
    return parent_pid


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
