# ADR 0001: Global code memory MCP complementing Hindsight

## Status

Accepted (grill session).

## Context

Hermes agents need to find code by meaning across projects while cwd may be elsewhere. Hindsight provides structured fact memory (retain/recall/consolidation) but not first-class cross-repo symbol search. Turbovec compresses vector indexes efficiently when many projects are registered globally. An opencode-codebase-index-style plugin per repo underuses that strength.

## Decision

Ship **hermes-turbomem** as a **single global MCP server** (Python FastMCP, stdio) that:

- Indexes code per repo under `{root}/.turbomem/` with Turbovec + SQLite (BM25, call graph).
- Registers projects in `~/.hermes/turbomem/` catalog for cross-project `code_recall`.
- Runs in **complement mode**: no `remember`; facts stay in Hindsight or MEMORY.md.
- Follows opencode indexing/search/freshness/diagnostics patterns where proven.

## Consequences

- Hermes config: one `mcp_servers.turbomem` entry; optional `project_id`/`project_path` filters on tools.
- Users need a memory-routing skill (facts → `hindsight_*`, code → `code_*`).
- v1 does not replace Hindsight or implement observation-style consolidation.
- Implementation is Python-first (tree-sitter-languages, sentence-transformers, turbovec)—not a TS/Rust MCP fork of opencode.

## Alternatives considered

- **Turbovec inside Hindsight**: wrong layer; Postgres/graph stack, not pluggable via Hermes config.
- **Replace Hindsight with turbomem facts**: duplicates consolidation/graph work.
- **Per-repo MCP servers** (opencode `--project`): simpler but poor fit for global Hermes agent.
- **Vectors-only v1**: rejected; hybrid BM25 matches opencode and helps exact symbol names.
