# hermes-turbomem

Local MCP **Persistent Memory** for [Hermes Agent](https://github.com/NousResearch/hermes-agent): TurboQuant-compressed vectors ([turbovec](https://github.com/RyanCodrai/turbovec)) plus unified semantic **Recall** over **Experiences** and **Code Entries** — across projects, even when you are not in that repo's working directory.

Domain terms: [CONTEXT.md](./CONTEXT.md).

## MCP tools

| Tool | Purpose |
|------|---------|
| `remember` | Store an experience (fact, fix, preference) |
| `index_project` | Index a codebase (Tree-sitter when installed, regex fallback) |
| `recall_memory` | Search all memory (optional `project_id`, `types`, `project_path`) |
| `list_projects` | List indexed project roots |

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
