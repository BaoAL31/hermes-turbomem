from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

from hermes_turbomem.config import load_config
from hermes_turbomem.embedder import Embedder, is_model_cached
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
def list_code_projects() -> str:
    """List registered projects in the catalog with root paths and last indexed time."""
    return _store.list_projects()


@mcp.tool()
def preload_models() -> str:
    """Download and cache the embedding model weights for offline use."""
    try:
        _embedder.preload()
        return f"Model '{_config.embedding_model}' loaded and cached."
    except Exception as e:
        return f"Failed to load model '{_config.embedding_model}': {e}"


@mcp.tool()
def index_status() -> str:
    """Report index readiness, chunk counts, and embedding model cache state."""
    status = _store.index_status()
    cached = is_model_cached(_config.embedding_model)

    lines = [
        "Index status:",
        f"  Data dir: {status['data_dir']}",
        f"  Database: {status['projects']} project(s), {status['entries']} code entry/chunk(s)",
        f"  Embed model '{_config.embedding_model}': {'cached' if cached else 'not cached'}",
    ]

    if status["last_indexed"] is not None:
        lines.append(
            f"  Last indexed: {time.strftime('%Y-%m-%d %H:%M', time.localtime(status['last_indexed']))}"
        )
    else:
        lines.append("  Last indexed: never")

    if status["projects"] == 0:
        lines.append("")
        lines.append("No projects indexed yet. Use `index_codebase(<path>)` to index a project.")
    if not cached:
        lines.append("Embedding model not cached. Use `preload_models()` to download before offline use.")

    return "\n".join(lines)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
