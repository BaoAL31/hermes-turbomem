# hermes-turbomem

Local MCP **code memory** for [Hermes Agent](https://github.com/NousResearch/hermes-agent): TurboQuant-compressed **Code Entries** ([turbovec](https://github.com/RyanCodrai/turbovec)), hybrid **Code recall** across indexed **Projects**, complementing Hindsight (or MEMORY.md) for facts.

- **Domain glossary:** [CONTEXT.md](./CONTEXT.md)
- **Architecture:** [docs/adr/0001-global-code-memory-complement.md](./docs/adr/0001-global-code-memory-complement.md)
- **PRD (v1):** [docs/PRD-v1-global-code-memory-mcp.md](./docs/PRD-v1-global-code-memory-mcp.md)

## MCP tools (v1)

| Tool | Purpose |
|------|---------|
| `preload_models` | Download/cache embedding weights before offline use |
| `index_codebase` | Index a project root |
| `code_recall` | Hybrid search across the catalog |
| `code_peek` | Metadata-only hits |
| `code_call_graph` | Callers / callees |
| `list_code_projects` | List indexed projects |
| `index_status` / `index_health_check` / `index_logs` / `index_metrics` | Diagnostics |

## Install

```bash
cd hermes-turbomem
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[index]"
```

First run downloads the embedding model (`nomic-ai/nomic-embed-text-v1` by default). To pre-cache before offline use, run:

```bash
python -m hermes_turbomem.server --preload
```
or call the `preload_models()` tool after the server starts.

## Hermes configuration

Copy-paste into `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  turbomem:
    command: python
    args: ["-m", "hermes_turbomem.server"]
```

After `pip install -e .`, or for faster startup:

```yaml
mcp_servers:
  turbomem:
    command: hermes-turbomem
```

Reload in Hermes: `/reload-mcp`.

## Server config

`~/.hermes/turbomem/config.yaml` (optional, all keys have defaults):

```yaml
auto_index_on_first_use: false   # default: manual index_project only
embedding_model: nomic-ai/nomic-embed-text-v1
bit_width: 4
default_recall_limit: 8
```

## Usage flow

1. **Pre-cache the model** (one-time, or for offline use):

   `preload_models()`

2. **Index a repo** (once per machine / after big changes):

   `index_codebase(path="C:/Users/you/Projects/myapi")`

3. **Recall code anywhere**:

   `code_recall(query="authenticate middleware")`

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
