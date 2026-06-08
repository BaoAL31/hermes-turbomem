# turbomem — Hermes memory provider

Local, free persistent memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Replaces Hindsight for users who want self-hosted hybrid recall without a knowledge-graph service.

## Requirements

- Hermes Agent installed (provides `agent.memory_provider`)
- Python 3.11+
- `pip install hermes-turbomem[index]` (or editable install from this repo)

## Install plugin

```bash
# From hermes-turbomem repo root
pip install -e ".[index]"

# Hermes discovers ~/.hermes/plugins/memory/<name>/
mkdir -p ~/.hermes/plugins/memory
cp -r plugins/memory/turbomem ~/.hermes/plugins/memory/turbomem   # Linux/macOS
# Windows: copy plugins\memory\turbomem to %USERPROFILE%\.hermes\plugins\memory\turbomem

hermes memory setup    # select turbomem
hermes memory status
```

## Activate

```yaml
# ~/.hermes/config.yaml
memory:
  provider: turbomem
```

No API keys. Data: `$HERMES_HOME/turbomem/`.

## Tools (provider surface)

| Tool | Purpose |
|------|---------|
| `memory_store` | Experiences: `retain`, `recall`, `list` |
| `index_codebase` | Index a project into the global store |
| `code_recall` | Hybrid code search (RRF) |
| `code_peek` | Metadata-only code hits |
| `code_call_graph` | Callers / callees |
| `list_code_projects` | Project catalog |
| `index_status` | Index readiness |
| `index_health_check` | Stale entry cleanup |
| `index_logs`, `index_metrics` | Diagnostics (when store supports) |
| `preload_models` | Warm embedding cache |

Agent routing: [docs/skills/hermes-memory-routing.md](../../../docs/skills/hermes-memory-routing.md). Schemas in `tools.py`.

## Research

See [docs/references.md](../../../docs/references.md) for Holographic, OpenViking, Hindsight wiring, and Turbo Puffer patterns.

## Cost

**$0** recurring — local disk + CPU only. Embedding weights download once from Hugging Face (optional `preload_models`).
