# Memory routing — hermes-turbomem (provider mode)

Hermes **`memory.provider: turbomem`** — local semantic memory. **Builtin** MEMORY.md stays enabled; turbomem mirrors writes into the semantic index (`on_memory_write`).

## Auto behavior (no tool call)

- **Before each turn:** Prefetch injects durable **Experiences** (not `conversation` transcripts, not code).
- **After each turn:** Raw user/assistant summary retained (`conversation` + `session:…` tags).
- **Before context compression:** Truncated transcript retained (`compression` tag).
- **On session end:** Final session summary (`session-end` tag) — especially when `retain_every_n_turns` > 1.

## Rules

| Question type | Action |
|---------------|--------|
| Durable fact, preference, fix, **code overview** (pattern/decision in prose) | `memory_store` **`retain`** — optional **`tags`** (loose, e.g. `topic:auth`); `project_path` only when repo-specific |
| Soft path hint before index (e.g. *"auth under `src/auth/`"*) | `memory_store` **`retain`** — fallback only; then `index_codebase` + `code_recall` |
| What do we know? / prior decisions / how we built X | Prefetch, or `memory_store` action **`recall`** |
| Browse stored facts | `memory_store` action **`list`** |
| Where is symbol / file / line? | **`code_recall`** — not `memory_store`; Experiences are not the location index |
| Who calls whom? | **`code_call_graph`** |
| Index a repo | **`index_codebase`** |

## Tool table

| Tool | When |
|------|------|
| `memory_store` | **retain** / **recall** / **list** for experiences only |
| `index_codebase` | First or forced re-index |
| `code_recall` | Code hybrid search |
| `code_peek` | Metadata-only code hits |
| `code_call_graph` | Callers / callees |
| `list_code_projects` | Indexed projects |
| `index_status` / `index_health_check` | Diagnostics |
| `index_logs` / `index_metrics` | Debug |
| `preload_models` | Offline embed cache |

## Examples

- *“Where is `authenticate`?”* → `code_recall(query="authenticate")`
- *“What did we decide about auth?”* → `memory_store(action="recall", query="auth decisions")`
- *“Remember: use yarn in this repo”* → `memory_store(action="retain", text="...", project_path="...")`

## Install

```bash
mkdir -p ~/.hermes/skills
cp docs/skills/hermes-memory-routing.md ~/.hermes/skills/hermes-memory-routing.md
```
