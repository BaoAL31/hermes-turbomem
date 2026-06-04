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
    return _store.index_project(path=path, force=force)


@mcp.tool()
def code_recall(
    query: str,
    limit: int | None = None,
    project_path: str | None = None,
    project_id: str | None = None,
) -> str:
    """
    Semantic code recall over Code Entries across indexed projects.
    Optional filters: project_id or project_path to narrow scope.
    Returns ranked hits or no-hit guidance.
    """
    resolved_id = project_id
    if project_path and not resolved_id:
        resolved_id = resolve_project(project_path).project_id
    return _store.recall(
        query=query,
        limit=limit,
        project_id=resolved_id,
        types=["code"],
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
    Metadata-only code recall: returns path, symbol, line range—no source body.
    Optional filters: project_id or project_path to narrow scope.
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
def index_health_check(
    project_path: str | None = None,
    project_id: str | None = None,
) -> str:
    """Remove stale Code Entries for deleted/renamed files. Reports cleanup counts."""
    return _store.health_check(project_id=project_id, project_path=project_path)


@mcp.tool()
def index_status(
    project_path: str | None = None,
    project_id: str | None = None,
) -> str:
    """Report index readiness, entry counts per project, embedding model state."""
    return _store.index_status(project_id=project_id, project_path=project_path)


@mcp.tool()
def list_code_projects() -> str:
    """List indexed projects and their root paths."""
    return _store.list_projects()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
