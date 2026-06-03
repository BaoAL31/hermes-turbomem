# hermes-turbomem

Local MCP **code memory** for [Hermes Agent](https://github.com/NousResearch/hermes-agent): TurboQuant-compressed **Code Entries** ([turbovec](https://github.com/RyanCodrai/turbovec)), hybrid **Code recall** across indexed **Projects**, complementing Hindsight (or MEMORY.md) for facts.

- **Domain glossary:** [CONTEXT.md](./CONTEXT.md)
- **Architecture:** [docs/adr/0001-global-code-memory-complement.md](./docs/adr/0001-global-code-memory-complement.md)
- **PRD (v1):** [docs/PRD-v1-global-code-memory-mcp.md](./docs/PRD-v1-global-code-memory-mcp.md) · [GitHub issue #1](https://github.com/BaoAL31/hermes-turbomem/issues/1)

> The Python scaffold in this repo predates the PRD; implement against the PRD and CONTEXT, not the old tool names below.

## MCP tools (v1 target — see PRD)

| Tool | Purpose |
|------|---------|
| `index_codebase` | Index a project root |
| `code_recall` | Hybrid search across the catalog |
| `code_peek` | Metadata-only hits |
| `code_call_graph` | Callers / callees |
| `list_code_projects` | List indexed projects |
| `index_status` / `index_health_check` / `index_logs` / `index_metrics` | Diagnostics |
| `preload_models` | Cache embedding weights offline |

## Install

```bash
cd hermes-turbomem
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[index]"
```

First run downloads the embedding model (`nomic-ai/nomic-embed-text-v1` by default).

## Hermes configuration

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  turbomem:
    command: python
    args: ["-m", "hermes_turbomem.server"]
    env:
      TURBOMEM_DATA_DIR: "C:/Users/you/.hermes/turbomem"
```

Or after `pip install -e .`:

```yaml
mcp_servers:
  turbomem:
    command: hermes-turbomem
```

Reload in Hermes: `/reload-mcp`.

## Server config

`~/.hermes/turbomem/config.yaml` (optional):

```yaml
auto_index_on_first_use: false   # default: manual index_project only
embedding_model: nomic-ai/nomic-embed-text-v1
bit_width: 4
default_recall_limit: 8
```

## Usage flow

1. **Index a repo** (once per machine / after big changes):

   `index_project(path="C:/Users/you/Projects/myapi")`

2. **Remember lessons**:

   `remember(text="Use yarn not npm in myapi", category="tooling", project_path=".../myapi")`

3. **Recall anywhere**:

   `recall_memory(query="authenticate middleware")`

## Data layout

```text
~/.hermes/turbomem/
├── config.yaml
├── index.tvim      # TurboQuant IdMapIndex
└── metadata.db     # entry text, paths, project ids
```

## Project identity

Projects are keyed by **git remote** when available, else **`local:{canonical path}`** — so the same clone path or remote resolves to one bank.

## License

MIT
