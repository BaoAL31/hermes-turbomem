# hermes-turbomem

Local MCP **code memory** for [Hermes Agent](https://github.com/NousResearch/hermes-agent): TurboQuant-compressed **Code Entries** ([turbovec](https://github.com/RyanCodrai/turbovec)), hybrid **Code recall** across indexed **Projects**, complementing Hindsight (or MEMORY.md) for facts.

- **Domain glossary:** [CONTEXT.md](./CONTEXT.md)
- **Architecture:** [docs/adr/0001-global-code-memory-complement.md](./docs/adr/0001-global-code-memory-complement.md)
- **PRD (v1):** [docs/PRD-v1-global-code-memory-mcp.md](./docs/PRD-v1-global-code-memory-mcp.md) · [GitHub issue #1](https://github.com/BaoAL31/hermes-turbomem/issues/1)
- **Memory routing skill:** [docs/skills/hermes-memory-routing.md](./docs/skills/hermes-memory-routing.md)

## MCP tools (v1 complement mode)

| Tool | Purpose |
|------|---------|
| `index_codebase` | First/full/incremental index for a project root |
| `code_recall` | Hybrid semantic search across the project catalog |
| `code_peek` | Metadata-only hits (path, symbol, lines — save tokens) |
| `code_call_graph` | Callers / callees where extraction exists |
| `list_code_projects` | List registered projects and roots |
| `index_status` | Check readiness, chunk counts, branch, model cache |
| `index_health_check` | Drop stale/orphan index rows |
| `index_logs` | Filtered debug log tail |
| `index_metrics` | Index/search timing counters |
| `preload_models` | Download embedding weights before offline use |

Not in complement v1: `remember`, `reflect`, fact consolidation.

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

## Config

`~/.hermes/turbomem/config.yaml` (optional):

```yaml
auto_index_on_first_use: false   # default: manual index_codebase only
embedding_model: nomic-ai/nomic-embed-text-v1
bit_width: 4
default_recall_limit: 8
```

## Usage flow

1. **Index a repo** (once per machine / after big changes):

   `index_codebase(path="C:/Users/you/Projects/myapi")`

2. **Search code from anywhere**:

   `code_recall(query="authenticate middleware")`

3. **Peek without source body** (saves tokens):

   `code_peek(query="validate_user")`

4. **Trace call graph**:

   `code_call_graph(name="validate_user", direction="callers", project_path="C:/Users/you/Projects/myapi")`

5. **Check diagnostics**:

   `index_status(project_path="C:/Users/you/Projects/myapi")`

## Data layout

```text
~/.hermes/turbomem/
├── config.yaml
├── index.tvim      # TurboQuant IdMapIndex
└── metadata.db     # entry text, paths, project ids
```

Per-project indexes live under `{repo}/.turbomem/`.

## Project identity

Projects are keyed by **git remote** when available, else **`local:{canonical path}`** — so the same clone path or remote resolves to one bank.

## Memory routing

hermes-turbomem runs in **complement mode**: facts belong to Hindsight (or MEMORY.md), code locations belong to turbomem. Install the [memory routing skill](./docs/skills/hermes-memory-routing.md) to teach your agent which tool to use:

```bash
mkdir -p ~/.hermes/skills
cp docs/skills/hermes-memory-routing.md ~/.hermes/skills/hermes-memory-routing.md
```

## License

MIT
