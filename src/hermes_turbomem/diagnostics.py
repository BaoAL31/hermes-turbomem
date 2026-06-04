from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
LogCategory = Literal[
    "index",
    "search",
    "embed",
    "parse",
    "store",
    "project",
    "config",
    "general",
]


@dataclass
class LogEntry:
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str


@dataclass
class _IndexMetricsState:
    embed_call_count: int = 0
    index_run_count: int = 0
    search_call_count: int = 0
    parse_error_count: int = 0
    embed_error_count: int = 0
    total_index_duration_ms: float = 0.0
    total_search_duration_ms: float = 0.0


class IndexLogger:
    def __init__(self, max_entries: int = 1000) -> None:
        self._max = max_entries
        self._entries: deque[LogEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def log(
        self,
        category: LogCategory,
        level: LogLevel,
        message: str,
    ) -> None:
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            category=category,
            message=message,
        )
        with self._lock:
            self._entries.append(entry)

    def get_logs(
        self,
        category: LogCategory | None = None,
        level: LogLevel | None = None,
        limit: int = 50,
    ) -> list[LogEntry]:
        with self._lock:
            matched = list(self._entries)
        if category:
            matched = [e for e in matched if e.category == category]
        if level:
            matched = [e for e in matched if e.level == level]
        return matched[-limit:]


class IndexMetrics:
    """Per-process-lifetime counters and timings.
    Metrics accumulate from process start and reset only on restart.
    """

    def __init__(self) -> None:
        self._state = _IndexMetricsState()
        self._lock = threading.Lock()

    def increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            s = self._state
            if name == "embed_call":
                s.embed_call_count += count
            elif name == "index_run":
                s.index_run_count += count
            elif name == "search_call":
                s.search_call_count += count
            elif name == "parse_error":
                s.parse_error_count += count
            elif name == "embed_error":
                s.embed_error_count += count

    def record_timing(self, name: str, duration_ms: float) -> None:
        with self._lock:
            s = self._state
            if name == "index":
                s.index_run_count += 1
                s.total_index_duration_ms += duration_ms
            elif name == "search":
                s.search_call_count += 1
                s.total_search_duration_ms += duration_ms

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            s = self._state
            return {
                "embed_call_count": s.embed_call_count,
                "index_run_count": s.index_run_count,
                "search_call_count": s.search_call_count,
                "parse_error_count": s.parse_error_count,
                "embed_error_count": s.embed_error_count,
                "total_index_duration_ms": s.total_index_duration_ms,
                "total_search_duration_ms": s.total_search_duration_ms,
            }


_logger = IndexLogger()
_metrics = IndexMetrics()


def get_logger() -> IndexLogger:
    return _logger


def get_metrics() -> IndexMetrics:
    return _metrics
