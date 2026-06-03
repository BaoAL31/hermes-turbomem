## Parent

#1

## What to build

Stand up the v1 MCP server shell: Python FastMCP over stdio, config loading from `~/.hermes/turbomem/config.yaml`, and catalog scaffolding under `~/.hermes/turbomem/`. Expose tools that work on an empty system: `preload_models` (download/cache local embedding weights), `list_code_projects` (empty catalog), and `index_status` (reports not indexed / model cache state). Document Hermes `mcp_servers` wiring in README. No indexing or search yet—responses must be honest and actionable.

## Acceptance criteria

- [ ] `hermes-turbomem` starts via stdio and registers with FastMCP under the expected name
- [ ] Config loads with documented defaults (manual first index, watcher settings placeholders OK)
- [ ] `preload_models` downloads or confirms cached local embedding model
- [ ] `list_code_projects` returns empty catalog without error
- [ ] `index_status` reports readiness, zero chunks, and embed cache state
- [ ] README shows copy-paste Hermes `mcp_servers` entry

## Blocked by

None - can start immediately
