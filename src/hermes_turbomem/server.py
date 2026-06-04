from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from hermes_turbomem.config import load_config
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import resolve_project
from hermes_turbomem.store import MemoryStore
from hermes_turbomem.watcher import ProjectWatcher

logger = logging.getLogger(__name__)

mcp = FastMCP("hermes-turbomem")

_config = load_config()
_embedder = Embedder(_config.embedding_model)
_store = MemoryStore(_config, _embedder)

_watchers: dict[str, ProjectWatcher] = {}


def _start_watcher(project_id: str, root_path: str) -> None:
    if not _config.watch_enabled:
        return
    if project_id in _watchers:
        return
    root = Path(root_path)
    if not root.is_dir():
        return
    watcher = ProjectWatcher(
        project_id=project_id,
        root=root,
        store=_store,
        poll_interval=_config.watch_poll_interval,
    )
    watcher.start()
    _watchers[project_id] = watcher


def _stop_all_watchers() -> None:
    for wid, w in list(_watchers.items()):
        w.stop()
    _watchers.clear()


def _start_watchers_for_registered() -> None:
    for proj in _store.get_registered_projects():
        _start_watcher(str(proj["project_id"]), str(proj["root_path"]))


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
    result = _store.index_project(path=path, force=force)
    if _config.watch_enabled:
        info = resolve_project(path)
        _start_watcher(info.project_id, str(info.root))
    return result


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
def index_status(project_path: str | None = None) -> str:
    """Report index readiness, chunk counts, branch, and embedder cache state."""
    lines: list[str] = []
    embedder_ready = _embedder.dimension > 0
    lines.append(f"embedding model: {_config.embedding_model}")
    lines.append(f"embedder ready: {embedder_ready}")
    lines.append(f"total entries in index: {len(_store._index)}")
    lines.append(f"watcher enabled: {_config.watch_enabled}")
    lines.append(f"auto_index_on_first_use: {_config.auto_index_on_first_use}")

    if project_path:
        info = resolve_project(project_path)
        pid = info.project_id
        indexed = _store.is_project_indexed(pid)
        lines.append(f"project {pid}: indexed={indexed}")
    else:
        projects = _store.get_registered_projects()
        if projects:
            for p in projects:
                w = "watched" if str(p["project_id"]) in _watchers else "unwatched"
                lines.append(f"  {p['project_id']}: {p['root_path']} ({w})")
        else:
            lines.append("  (no projects registered)")
    return "\n".join(lines)


def main() -> None:
    _start_watchers_for_registered()
    try:
        mcp.run(transport="stdio")
    finally:
        _stop_all_watchers()


if __name__ == "__main__":
    main()
