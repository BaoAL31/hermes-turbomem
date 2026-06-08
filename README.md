# hermes-turbomem

Local **persistent memory** for [Hermes Agent](https://github.com/NousResearch/hermes-agent): TurboQuant-compressed hybrid recall ([turbovec](https://github.com/RyanCodrai/turbovec)) over **Experiences** and **Code Entries**. **Self-hosted, no required cloud API** — replaces Hindsight for users who want semantic memory on their own machine.

## Docs

| Doc | Purpose |
|-----|---------|
| [CONTEXT.md](./CONTEXT.md) | Domain glossary |
| [docs/references.md](./docs/references.md) | **Research links** (Hermes providers, Holographic, OpenViking, Turbo Puffer, etc.) |
| [docs/nightshift-pr-triage.md](./docs/nightshift-pr-triage.md) | Open PRs #11–#19: keep / adjust / merge order |
| [docs/backlog.md](./docs/backlog.md) | Integration + provider work queue (B-01…) |
| [docs/PRD-v2-memory-provider.md](./docs/PRD-v2-memory-provider.md) | Current product spec (provider-first) |
| [docs/adr/0002-hermes-memory-provider-local-only.md](./docs/adr/0002-hermes-memory-provider-local-only.md) | Architecture decision |
| [docs/PRD-v1-global-code-memory-mcp.md](./docs/PRD-v1-global-code-memory-mcp.md) | Historical MCP-only scope |

## Quick start (Hermes memory provider — recommended)

```bash
cd hermes-turbomem
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e ".[index]"

# Install plugin for Hermes discovery
mkdir %USERPROFILE%\.hermes\plugins\memory 2>nul
xcopy /E /I plugins\memory\turbomem %USERPROFILE%\.hermes\plugins\memory\turbomem

hermes memory setup             # select turbomem — no API key
hermes memory status
```

```yaml
# ~/.hermes/config.yaml
memory:
  provider: turbomem
```

Data directory: `~/.hermes/turbomem/` (profile-scoped). **Cost: $0** recurring (CPU + disk; embedding model may download once from Hugging Face).

### Provider tools

| Tool | Purpose |
|------|---------|
| `memory_store` | Experiences: `retain`, `recall`, `list` |
| `index_codebase` | Index a repository |
| `code_recall` / … | Code tools (see PRD v2) |

Auto **prefetch** before turns and background **retain** after turns follow the [Holographic / Hindsight plugin patterns](docs/references.md) — see `plugins/memory/turbomem/__init__.py`.

## Install (library only)

```bash
pip install -e ".[index]"
```

## Config

Two files under `$HERMES_HOME/turbomem/`:

**`config.yaml`** — shared store (read by `MemoryStore`):

```yaml
auto_index_on_first_use: false
embedding_model: nomic-ai/nomic-embed-text-v1
bit_width: 4
default_recall_limit: 8
confidence_threshold: 0.3
```

**`provider.yaml`** — Hermes hook knobs (via `hermes memory setup`):

```yaml
auto_retain_turns: true
retain_every_n_turns: 1
retain_max_chars_per_side: 1200
recall_max_chars: 4000
```

### Manual install smoke checklist

1. `pip install -e ".[index]"` from repo root
2. Copy `plugins/memory/turbomem` → `~/.hermes/plugins/memory/turbomem`
3. `hermes memory setup` → select **turbomem** (no API key)
4. `hermes memory status` shows turbomem active
5. Set `memory.provider: turbomem` in `~/.hermes/config.yaml`

## Research before you build

Start with [docs/references.md](./docs/references.md). Recommended order:

1. Hermes [Memory Provider developer guide](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/memory-provider-plugin.md)
2. **Holographic** plugin (minimal local provider)
3. **OpenViking** (tiered recall)
4. Turbo Puffer docs (hybrid search semantics only)

## License

MIT
