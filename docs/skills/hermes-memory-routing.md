# Memory routing skill for hermes-turbomem (complement mode)

**hermes-turbomem** runs alongside a fact memory system (Hindsight or built-in Hermes memory). This skill teaches the agent which tool to call for what.

## Rule

- **Facts** (preferences, fixes, conventions, environment quirks, project rules) → `hindsight_retain` / `hindsight_recall` (or built-in Hermes remember/recall).
- **Code location** (symbols, paths, line ranges, callers/callees) → hermes-turbomem `code_*` tools.
- **Never** duplicate the same information in both stores.

## Tool routing

| Category | Tool | When to use |
|----------|------|-------------|
| Fact store | `hindsight_retain` | Save a new fact (preference, fix, convention). |
| Fact store | `hindsight_recall` | Retrieve previously learned facts. |
| Code location | `index_codebase` | First-time or incremental index of a project root. |
| Code location | `code_recall` | Semantic search for code across all indexed projects. |
| Code location | `code_peek` | Lightweight search — paths and symbols only, no source body. |
| Code location | `code_call_graph` | Find callers or callees of a symbol. |
| Code location | `list_code_projects` | List all indexed projects and their roots. |
| Diagnostics | `index_status` | Check if an index is ready, chunk counts, branch, model cache. |
| Diagnostics | `index_health_check` | Remove stale/orphan index entries. |
| Diagnostics | `index_logs` | Tail filtered index/embedding logs. |
| Diagnostics | `index_metrics` | View indexing/search timing counters. |
| Setup | `preload_models` | Download embedding weights before going offline. |

## Examples

- User asks *"where is the `authenticate` function?"* → **`code_recall(query="authenticate")`**
- User asks *"what did we decide about auth tokens?"* → **`hindsight_recall(query="auth tokens")`**
- User asks *"who calls `validate_user`?"* → **`code_call_graph(name="validate_user", direction="callers")`**

## Installation

Place or symlink this file in the agent's skills directory so the memory routing rule is loaded at session start:

```bash
mkdir -p ~/.hermes/skills
cp docs/skills/hermes-memory-routing.md ~/.hermes/skills/hermes-memory-routing.md
```
