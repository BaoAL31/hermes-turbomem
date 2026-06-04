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
def index_codebase(path: str, force: bool = False) -> str:
    """Index a project root: parse code symbols, embed, and store Code Entries."""
    return _store.index_codebase(path=path, force=force)


@mcp.tool()
def code_recall(
    query: str,
    limit: int | None = None,
    project_path: str | None = None,
    project_id: str | None = None,
) -> str:
    """
    Hybrid semantic search over Code Entries across all indexed projects.
    Optional filters: project_id or project_path to narrow scope.
    """
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.code_recall(
        query=query,
        limit=limit,
        project_id=resolved_id,
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
    Metadata-only code search: paths and symbols without source bodies.
    Saves tokens before deciding to read a file.
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
def code_call_graph(
    name: str,
    direction: str = "callers",
    project_path: str | None = None,
    project_id: str | None = None,
    symbol_id: int | None = None,
) -> str:
    """
    Find callers or callees of a named symbol in a project.
    Requires project_id or project_path. Returns unsupported notice for languages
    where call graph extraction is not available.
    """
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.code_call_graph(
        name=name,
        direction=direction,
        project_id=resolved_id,
        project_path=project_path,
        symbol_id=symbol_id,
    )


@mcp.tool()
def list_code_projects() -> str:
    """List indexed projects and their root paths."""
    return _store.list_code_projects()


@mcp.tool()
def index_status(project_path: str | None = None, project_id: str | None = None) -> str:
    """Index readiness: chunk counts, branch, embed model cached? Pass project_id or project_path for per-project status."""
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.index_status(project_id=resolved_id, project_path=project_path)


@mcp.tool()
def index_health_check(project_path: str | None = None, project_id: str | None = None) -> str:
    """Drop stale/orphan index rows and projects."""
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.index_health_check(project_id=resolved_id, project_path=project_path)


@mcp.tool()
def index_logs(category: str | None = None, level: str | None = None, limit: int | None = 50) -> str:
    """Filtered debug log tail for indexing and embedding events."""
    return _store.index_logs(category=category, level=level, limit=limit)


@mcp.tool()
def index_metrics() -> str:
    """Index/search timing counters and entry counts."""
    return _store.index_metrics()


@mcp.tool()
def preload_models() -> str:
    """Download embedding model weights and prepare vector index before offline use."""
    return _store.preload_models()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
