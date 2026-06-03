from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from hermes_turbomem.config import load_config
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
def code_peek(
    query: str,
    limit: int | None = None,
    project_path: str | None = None,
    project_id: str | None = None,
) -> str:
    """
    Metadata-only code search: returns paths, symbols, line ranges, and scores—no source bodies.
    Uses hybrid search (vectors + BM25) and flags low-confidence results.
    """
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.code_peek(
        query=query,
        limit=limit,
        project_id=resolved_id,
        project_path=project_path,
    )


@mcp.tool()
def list_projects() -> str:
    """List indexed projects and their root paths."""
    return _store.list_projects()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
