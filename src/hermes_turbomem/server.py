from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from hermes_turbomem.config import load_config
from hermes_turbomem.embedder import Embedder
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
    project_id: str | None = None,
    project_path: str | None = None,
) -> str:
    """
    Semantic recall over Code Entries across all indexed projects.
    Optional filters: project_id, or project_path to narrow scope.
    """
    return _store.code_recall(
        query=query,
        limit=limit,
        project_id=project_id,
        project_path=project_path,
    )


@mcp.tool()
def list_code_projects() -> str:
    """List indexed projects and their root paths."""
    return _store.list_code_projects()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
