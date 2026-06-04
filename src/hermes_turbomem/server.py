from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

from hermes_turbomem.config import load_config
from hermes_turbomem.diagnostics import get_logger, get_metrics
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import resolve_project
from hermes_turbomem.store import MemoryStore

mcp = FastMCP("hermes-turbomem")

_config = load_config()
_embedder = Embedder(_config.embedding_model)
_store = MemoryStore(_config, _embedder)


@mcp.tool()
def remember(
    text: str,
    category: str = "general",
    project_path: str | None = None,
) -> str:
    """Store a learned fact, preference, fix, or convention in persistent memory."""
    return _store.remember(text=text, category=category, project_path=project_path)


@mcp.tool()
def index_project(path: str, force: bool = False) -> str:
    """Index a project directory: parse code symbols, embed, and store Code Entries."""
    return _store.index_project(path=path, force=force)


@mcp.tool()
def recall_memory(
    query: str,
    limit: int | None = None,
    project_path: str | None = None,
    project_id: str | None = None,
    types: list[str] | None = None,
) -> str:
    """
    Unified semantic recall over experiences and code entries across all indexed projects.
    Optional filters: project_id, types (experience, code), or project_path for auto-index.
    """
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.recall(
        query=query,
        limit=limit,
        project_id=resolved_id,
        types=types,
        project_path=project_path,
    )


@mcp.tool()
def list_projects() -> str:
    """List indexed projects and their root paths."""
    return _store.list_projects()


@mcp.tool()
def index_logs(
    category: str | None = None,
    level: str | None = None,
    limit: int = 50,
) -> str:
    """Return recent log lines; optionally filter by category and level.
    Categories: index, search, embed, parse, store, project, config, general.
    Levels: DEBUG, INFO, WARN, ERROR.
    """
    logger = get_logger()
    entries = logger.get_logs(
        category=category,  # type: ignore[arg-type]
        level=level,  # type: ignore[arg-type]
        limit=limit,
    )
    if not entries:
        return "No log entries found."
    lines: list[str] = []
    for e in entries:
        ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
        lines.append(f"[{ts}] [{e.level}] [{e.category}] {e.message}")
    return "\n".join(lines)


@mcp.tool()
def index_metrics() -> str:
    """Return per-process-lifetime counters and timings for index, embed, search.
    Metrics reset on process restart. Accumulate until then.
    """
    m = get_metrics().snapshot()
    return (
        f"Embed calls: {m['embed_call_count']}\n"
        f"Index runs: {m['index_run_count']}\n"
        f"Search calls: {m['search_call_count']}\n"
        f"Parse errors: {m['parse_error_count']}\n"
        f"Embed errors: {m['embed_error_count']}\n"
        f"Total index duration: {m['total_index_duration_ms']:.0f} ms\n"
        f"Total search duration: {m['total_search_duration_ms']:.0f} ms\n"
        "\nMetrics reset on process restart. Timings and counters"
        " accumulate for the lifetime of this process."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
