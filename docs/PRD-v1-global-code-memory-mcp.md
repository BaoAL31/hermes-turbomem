# PRD: hermes-turbomem v1 — Global code memory MCP (complement mode)

## Problem Statement

Hermes agents often run outside the repository they need to reason about. Built-in MEMORY.md is tiny and not semantic; session search is chat history, not code structure. Hindsight handles facts and consolidation well but does not provide first-class, cross-project symbol location (file, line, “where is `authenticate`?”). Per-repo indexing tools (e.g. opencode-style plugins) do not optimize for a **global agent** that jumps between many codebases. Developers need fast, local, privacy-friendly **code recall** across indexed projects without duplicating fact memory or maintaining a heavy memory platform.

## Solution

Ship **hermes-turbomem**: a single **global MCP server** (Python FastMCP, stdio) that indexes code per **Project** (repo root), stores compressed vectors with **Turbovec** under `{repo}/.turbomem/`, registers projects in a global **Project catalog** under `~/.hermes/turbomem/`, and exposes opencode-aligned tools for **Indexing**, hybrid **Code recall**, **Call graph**, and diagnostics. Run in **Complement mode**: facts stay in Hindsight (or MEMORY.md); turbomem owns **Code Entries** only—no `remember` tool. Provide a Hermes **memory routing** skill so agents use `hindsight_*` for facts and `code_*` for locations.

## User Stories

1. As a Hermes user, I want to register the turbomem MCP server once in config, so that every session can use code memory without per-repo MCP entries.
2. As a Hermes user, I want to run `index_codebase` on a project root, so that its symbols are searchable later from any working directory.
3. As a Hermes user, I want `code_recall` to search all indexed projects by default, so that I can find code while cwd is elsewhere.
4. As a Hermes user, I want optional `project_id` or `project_path` filters on tools, so that I can narrow recall to one codebase when I know the target.
5. As a Hermes user, I want indexing to respect `.gitignore` and skip vendor/build artifacts, so that indexes stay fast and relevant.
6. As a Hermes user, I want git-tracked-file scoping in git repos, so that untracked junk is not embedded.
7. As a Hermes user, I want a max file size and max chunks per file, so that huge generated files do not blow up the index.
8. As a Hermes user, I want semantic chunks from tree-sitter for common languages, so that search finds functions and classes, not arbitrary text splits.
9. As a Hermes user, I want hybrid vector + BM25 recall, so that both conceptual queries and exact symbol names work.
10. As a Hermes user, I want Turbovec-backed storage per project, so that many indexed repos use less RAM than raw float indexes.
11. As a Hermes user, I want stable project identity via git remote when available, so that the same repo cloned to different paths shares one logical Project where appropriate.
12. As a Hermes user, I want fallback project identity via canonical local path, so that non-git folders can still be indexed.
13. As a Hermes user, I want incremental re-indexing by content hash, so that repeated indexing only re-embeds changed files.
14. As a Hermes user, I want a file watcher to update the index after saves (configurable), so that recall stays fresh during active development.
15. As a Hermes user, I want git-aware branch membership and updates on branch switch, so that recall matches the current branch without full re-embed every time.
16. As a Hermes user, I want to disable the watcher in config, so that I control CPU use on large monorepos.
17. As a Hermes user, I want `code_peek` to return paths and line ranges without full source bodies, so that the agent saves context tokens before `read_file`.
18. As a Hermes user, I want `code_call_graph` for callers and callees, so that I can trace execution flow without manual grep chains.
19. As a Hermes user, I want call graph support for TypeScript, JavaScript, Python, Go, and Rust in v1, so that my typical stacks are covered.
20. As a Hermes user, I want other languages to still be chunk-indexed and searchable when parsers exist, so that call graph gaps do not block basic recall.
21. As a Hermes user, I want a clear message when call graph is unsupported for a symbol’s language, so that the agent can fall back to `code_recall` or grep.
22. As a Hermes user, I want local embeddings with automatic model download on first use, so that I do not manually fetch weights.
23. As a Hermes user, I want `preload_models` to warm the cache before offline work, so that airplanes and air-gapped labs still work after one online preload.
24. As a Hermes user, I want no cloud embedding API in v1, so that code memory stays privacy-first.
25. As a Hermes user, I want `list_code_projects` to show indexed roots and ids, so that I know what the catalog contains.
26. As a Hermes user, I want `index_status` to report readiness, chunk counts, branch, and model cache state, so that I can debug “why is search empty?”
27. As a Hermes user, I want `index_health_check` to remove stale orphans, so that deleted files do not pollute recall.
28. As a Hermes user, I want `index_logs` with category and level filters, so that I can diagnose indexing and embedding failures.
29. As a Hermes user, I want `index_metrics` for timing and throughput, so that I can see whether slowness is embed vs search vs index.
30. As a Hermes user, I want `code_recall` to return explicit no-hit guidance pointing at status/health tools, so that failures match opencode observability patterns.
31. As a Hermes user, I want low-confidence hits flagged when scores are weak, so that the agent does not treat noise as ground truth.
32. As a Hermes user, I want complement mode without a `remember` tool, so that I do not duplicate facts in Hindsight and turbomem.
33. As a Hermes user, I want a documented skill that routes facts to `hindsight_retain`/`hindsight_recall` and code to `code_*`, so that the model picks the right store.
34. As a Hermes user, I want first indexing to be manual by default, so that I control cost and time of large repos.
35. As a Hermes user, I want optional auto-index on first recall (off by default), so that power users can reduce friction.
36. As a Hermes user, I want config under `~/.hermes/turbomem/config.yaml`, so that settings match other Hermes paths.
37. As a Hermes user, I want `.turbomem/` colocated in each repo, so that indexes travel with the codebase and can be gitignored.
38. As a Hermes user, I want README instructions for Hermes `mcp_servers` wiring, so that setup is copy-pasteable.
39. As a developer, I want deep modules with testable interfaces, so that indexing, search, and MCP wiring can be tested without a live Hermes session.
40. As a developer, I want the existing scaffold aligned to this spec, so that early commits do not contradict complement mode and hybrid storage.

## Implementation Decisions

### Architectural (see ADR 0001)

- **Complement mode** only for v1: code-only turbomem; facts via Hindsight or MEMORY.md.
- **One global MCP process**; cross-project **Code recall** default; per-tool project filters optional.
- **Hybrid storage**: per-**Project index** at `{root}/.turbomem/` + global **Project catalog** at `~/.hermes/turbomem/`.
- **Turbovec** `IdMapIndex` per project for all vectors (4-bit TurboQuant default).
- **Python FastMCP** + stdio transport; no TypeScript/Rust MCP host in v1.

### Deep modules (build or refactor)

| Module | Responsibility | Interface (conceptual) |
|--------|----------------|-------------------------|
| **Config** | Load `~/.hermes/turbomem/config.yaml`, env overrides | `load_config() -> TurbomemConfig` |
| **Project identity** | Resolve `project_id`, repo root, git remote | `resolve_project(path) -> ProjectInfo` |
| **Index scope** | Enumerate indexable files (gitignore, git ls-files, caps, deny list) | `iter_indexable_files(root) -> paths` |
| **Chunking** | Tree-sitter semantic chunks + overlap rules | `extract_chunks(path, root) -> CodeChunk[]` |
| **Call graph extract** | Build caller/callee edges for supported langs | `extract_edges(chunks, ast) -> CallEdge[]` |
| **Embedder** | Local model encode; lazy load; preload | `encode(texts) -> vectors`; `preload()` |
| **Project index store** | SQLite metadata, BM25, turbovec add/search/remove, call edges | `index_project`, `search`, `call_graph`, `health_check` |
| **Catalog** | Register/list projects; point to on-disk index paths | `register`, `list`, `get` |
| **Freshness** | File watcher + git branch/diff incremental pipeline | `start_watcher`, `on_git_event`, `incremental_index` |
| **Search fusion** | Vector top-k + BM25 top-k + RRF (or weighted) + score thresholds | `hybrid_search(query, filters) -> RankedHit[]` |
| **MCP surface** | Tool definitions, error messages, no auto-repair inside recall | FastMCP tool handlers delegating to above |
| **Diagnostics** | Status, metrics ring buffer, log tail | `status`, `metrics`, `logs` |

### API contracts (MCP tools)

- `index_codebase(path, force=False)` → summary string (added/skipped/removed counts).
- `code_recall(query, limit?, project_id?, project_path?)` → formatted ranked hits or no-hit + hints.
- `code_peek(query, limit?, project_id?, project_path?)` → metadata-only hits.
- `code_call_graph(name, direction, project_id?, project_path?, symbol_id?)` → caller/callee list or unsupported notice.
- `list_code_projects()` → catalog listing.
- `index_status(project_id?, project_path?)` → readiness, counts, branch, embed cache.
- `index_health_check(project_id?, project_path?)` → cleanup report.
- `index_logs(category?, level?, limit?)` → log lines.
- `index_metrics()` → counters JSON/text.
- `preload_models()` → download/cache confirmation.

### Schema (conceptual)

- **Catalog row**: project_id, root_path, git_remote?, indexed_at, index_paths, current_branch?.
- **Code entry row**: id, project_id, path, symbol, kind, start_line, end_line, content_hash, embed_text.
- **Call edge row**: project_id, caller_id, callee_id, kind.
- **BM25** inverted index per project (SQLite FTS or external inverted table).
- **Turbovec** id map aligned with code entry ids.

### Non-goals in implementation

- No `remember`, `reflect`, or Hindsight consolidation.
- No symbol containment tree recall.
- No cloud embeddings.
- No auto-run health repair inside `code_recall`.

## Testing Decisions

**Good tests** exercise observable behavior: given a tiny fixture repo, index it, recall by query, peek without body, call graph on known snippet, health check after file delete, catalog lists project. Do not assert internal tree-sitter walk order or turbovec internals.

**Modules to test (recommended)**:

| Module | Why |
|--------|-----|
| **Project identity** | Pure; git vs local path edge cases |
| **Index scope** | gitignore/deny/caps with fixture dirs |
| **Chunking** | Known Python/TS snippet → expected symbols |
| **Search fusion** | Controlled BM25 + vector stubs → ranking order |
| **Project index store** | Temp dir integration: index → search → delete file → health |
| **Catalog** | Register two projects; list/filter |

**Lower priority for unit tests**: MCP stdio wiring (smoke test only), file watcher (integration/manual), embedder (mock encode in store tests).

**Prior art**: None in-repo yet; follow pytest patterns when added; store tests use `tmp_path` fixtures.

## Out of Scope

- Solo mode and `remember` tool.
- Hindsight integration code (user configures Hermes separately).
- Replacing or forking Hindsight; Turbovec inside Hindsight.
- Full opencode feature parity (`find_similar`, slash commands, Rust native indexer).
- Symbol AST parent hierarchy as a recall mode.
- Cloud embedding providers.
- Automatic consolidation / observations.
- HTTP MCP transport.
- Indexing `node_modules` or secrets (explicitly excluded by scope).
- `.turbomemindexingignore` split (Cursor-style) — optional later.
- Additional call-graph languages beyond v1 set.

## Further Notes

- Early repo scaffold may predate grill decisions; implementation must realign to complement + hybrid storage + tool names.
- Publish as **one** v1 issue; split post-v1 work (solo mode, more languages, `find_similar`) into separate issues only when scheduling independent work.
- Memory routing skill lives in user Hermes skills dir (documented in README, not necessarily shipped inside package v1).
- Label `ready-for-agent` may need to be created on the tracker if triage workflow requires it.
