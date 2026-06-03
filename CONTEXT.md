# hermes-turbomem

Local MCP **code memory** for Hermes (and other agents): TurboQuant-indexed **Code Entries** with global **Recall**, complementing Hindsight or Hermes built-in memory for facts—not replacing them.

## Language

**Persistent Memory**:
The unified store of retrievable knowledge an agent accumulates over time (experiences plus code symbols from indexed projects).
_Avoid_: "RAG database", "vector DB" (implementation terms).

**Experience**:
A short, natural-language fact the agent should remember (preference, fix, convention, environment quirk).
_Avoid_: "Episodic memory" (academic), "memory entry" (too generic).

**Code Entry**:
A searchable record tied to one semantic unit of source (e.g. function, method, class) with location metadata (project, path, line range, symbol name).
_Avoid_: "Chunk", "document" (RAG jargon).

**Project**:
One indexed codebase, identified stably by git remote when present, otherwise by canonical root path (`git:…` or `local:…`). Acts as the **parent node** for all **Code Entries** from that tree—the anchor for cross-repo recall when the agent is elsewhere.
_Avoid_: "Repo" (informal only), "workspace" (Hermes overloads this).

**Project catalog**:
Global registry (under `~/.hermes/turbomem/`) listing every indexed **Project**: root path, `project_id`, last indexed time. Enables `code_recall` across projects without requiring cwd to be inside any of them.

**Project index**:
Per-project payload colocated with source (e.g. `{repo}/.turbomem/`)—vectors and metadata for that **Project** only. Matches opencode-style “index travels with the repo.”

**Call graph**:
Directed edges between symbols (caller → callee), built during **Indexing** from tree-sitter. Answers “who calls this?” and “what does this call?”—not class/file containment.
_Avoid_: Using **Call graph** to mean AST parent-child nesting.

**Code recall**:
Hybrid semantic search over flat **Code Entries** (Turbovec vectors + BM25, fused opencode-style)—not a walk of an AST or **Call graph**. Returns ranked hits; may include low-confidence results with explicit flags when scores are weak.
_Avoid_: `recall_memory` as the primary tool name (use **`code_recall`** per memory routing).

**Vector storage**:
Per-**Project** TurboQuant index (`IdMapIndex` / `.tvim` under `{repo}/.turbomem/`). Global **Project catalog** does not duplicate vectors—it points at each **Project index**.

**Indexing**:
Scan scoped files → tree-sitter semantic chunks → embed → store in **Project index**; extract **Call graph** edges; register in **Project catalog**. Incremental by content hash. Aligned with opencode-codebase-index (flat chunks, not containment-tree recall).
_Avoid_: "Retain" for code (reserve for Hindsight-style fact extraction if added later).

**Auto-indexing** (optional):
When enabled, the first `code_recall` or `index_codebase` against a Project path triggers Indexing without an explicit prior index. Default is off (manual **first** index only).

**Index freshness**:
How **Code Entries** and **Call graph** stay current after the first **Indexing**. v1: **file watcher** re-indexes on save (on by default once a **Project** is registered) plus **git-aware** incremental updates and branch membership (opencode-style). User can disable watching in config.

**Index scope**:
Which files Indexing may read. Aligned with Cursor and opencode-codebase-index: respect `.gitignore`; apply a built-in deny list (dependencies, build output, locks, binaries, `.env*`); cap file size and chunks per file; in git repos prefer **tracked files only** (`git ls-files`) so vendor and junk never enter the index. Optional `.turbomemignore` (gitignore syntax) for project-specific exclusions.

**Implementation stack** (path of least resistance):
MCP server in **Python** with **FastMCP** + **stdio** (Hermes `command` + `args`). Indexing via **tree-sitter** / **tree-sitter-languages** for multi-language chunks (opencode-aligned); **turbovec** for vectors; **SQLite** for metadata, BM25, and **Call graph**. Not a Rust/TypeScript MCP unless a hard requirement appears later.

**Complement mode**:
hermes-turbomem runs alongside another fact system (typically Hindsight or MEMORY.md). Facts and consolidation live there; hermes-turbomem owns **Code Entries** and code **Recall** only—no duplicate fact stores. No **`remember`** tool in complement mode; facts use `hindsight_retain` (or built-in memory), code uses **`index_codebase`** only.

**Solo mode** (post-v1):
No external fact provider; turbomem could add **`remember`** and unified fact+code recall later. v1 targets **Complement mode** only.

**Consolidation**:
Deduping/synthesizing learned facts (Hindsight **observations**, npm×5 → one rule). **Not in turbomem v1**—Hindsight (or MEMORY.md) owns that in complement mode; turbomem stays fresh via **Index freshness**, not fact merge.

**Embeddings**:
Local model only (default `nomic-ai/nomic-embed-text-v1` via sentence-transformers). Weights auto-download on first use; **`preload_models`** for explicit cache-before-offline. No cloud embedder in v1.

**Memory routing**:
How the agent chooses between fact tools (e.g. `hindsight_*`, MEMORY.md) and code tools (e.g. `code_recall`, `index_codebase`). Implemented via a user **skill** (rules) plus **code-prefixed** MCP tool names—not by renaming Hindsight tools.

**Failure handling**:
Opencode-style behavior when retrieval/indexing fails: `code_recall` returns explicit no-hit or low-confidence output plus actionable follow-ups (`index_status`, `index_health_check`, `index_logs`, `index_metrics`, and re-index guidance). It does not auto-run repair tools inside recall.

**Diagnostics** (v1):
`index_status`, `index_health_check`, `index_logs`, and `index_metrics`—opencode-style observability for index readiness, cleanup, and debug.

**MCP deployment**:
One global MCP server process registered once in Hermes. **Code recall** searches the full **Project catalog** by default; tools accept optional `project_id` or `project_path` to narrow scope. Not one MCP server per repo.

## v1 MCP tools (complement mode)

| Tool | Role |
|------|------|
| `index_codebase` | First/full/incremental index for a project root |
| `code_recall` | Hybrid search across catalog (optional project filter) |
| `code_peek` | Metadata-only hits (path, symbol, lines—save tokens) |
| `code_call_graph` | Callers/callees where extraction exists |
| `list_code_projects` | Registered projects and roots |
| `index_status` | Ready? chunk counts, branch, embed model cached? |
| `index_health_check` | Drop stale/orphan index rows |
| `index_logs` | Filtered debug log tail |
| `index_metrics` | Index/search timing counters |
| `preload_models` | Download embedding weights before offline use |

Not in complement v1: `remember`, `reflect`, Hindsight-style consolidation.

## Relationships

- A **Project** has many **Code Entries** after **Indexing**.
- In **Complement mode**, facts belong to Hindsight (or built-in memory), not to hermes-turbomem; **`remember`** is not exposed.
- **Code recall** searches **Code Entries** across all Projects by default; optional filters narrow by **Project**.
- First **Indexing** is manual (`index_codebase`); **Auto-indexing** on first use is optional and off by default.
- After registration, **Index freshness** keeps the **Project index** updated via watcher + git-aware incremental logic unless disabled.
- **Indexing** only includes paths allowed by **Index scope**; skipped paths never become **Code Entries**.
- Every **Code Entry** belongs to exactly one **Project** (parent). **Project catalog** points to each **Project index**.
- **Call graph** links symbols by invocation; **Code recall** does not traverse it unless the agent uses **`code_call_graph`**.
- **Code Entries** + **Code recall** answer “where is this by meaning?”; **`code_call_graph`** answers “who calls it?”; learned **facts** answer “what should we do?” via Hindsight, not turbomem.
- **Memory routing**: facts → Hindsight (or built-in); code location → hermes-turbomem `code_*` tools; documented in a Hermes skill.
- **Failure handling** keeps search/tool failures observable: diagnose with status/health/logs first, then rerun index; avoid hidden auto-mutations during recall.
- **MCP deployment**: single Hermes `mcp_servers` entry; cross-project recall is default; per-tool filters optional.

## Example dialogue

> **Dev:** "We're in ~/notes, but where was `authenticate` in the API service?"
> **Domain expert:** "Call **`code_recall`** — results are scoped under the API **Project** parent, with path and line range, even though we're not in that tree. Use **`hindsight_recall`** only if they're asking what we *decided* about auth, not where it lives in source."

> **Dev:** "Can we index the same monorepo as two projects?"
> **Domain expert:** "Only if they are two distinct roots in the **Project catalog**—usually one parent **Project** per clone/root; don't split one AST tree arbitrarily."

> **Dev:** "Why did `code_recall` return no results?"
> **Domain expert:** "Treat it like opencode: check **`index_status`**, then **`index_health_check`** if unhealthy, and re-run **`index_codebase`** when needed."


## Grill decisions (locked)

| Topic | Decision |
|-------|----------|
| Product fit | Hermes **global** code memory MCP; Turbovec shines on multi-project catalog—not a per-repo-only opencode plugin |
| vs Hindsight | **Complement**: facts Hindsight/MEMORY; code turbomem; **no `remember`** |
| Storage | **Hybrid**: `{repo}/.turbomem/` + `~/.hermes/turbomem/` catalog |
| Vectors | **Turbovec** in every **Project index** (TurboQuant 4-bit default) |
| MCP | **Python FastMCP**, stdio, **one** global server (**C**) |
| Search | **Hybrid** vectors + BM25, RRF-style fusion |
| Embeddings | **Local only**, auto-download + **`preload_models`** |
| Index scope | **gitignore** + deny list + caps + **`git ls-files`** |
| Freshness | **Watcher on** after register + **git-aware** incremental |
| First index | **Manual**; auto-index-on-first-use **off** by default |
| Routing | **B+D** skill + `code_*` tool names |
| Errors | **Opencode-style**: clear no-hit + point to status/health/logs/metrics |
| Languages | **tree-sitter-languages** chunks; **call graph** TS/JS, Python, Go, Rust first |
| Consolidation | **Out of scope** v1 (Hindsight) |
| Solo + `remember` | **Deferred** post-v1 |

## Flagged ambiguities (resolved)

- Three memory layers in complement: MEMORY.md (tiny always-on) + Hindsight (facts) + turbomem (code locations).
- Do not call code indexing "retain."
- Symbol containment tree: out of v1.
- Started as opencode+turbovec idea; architecture pivoted when Turbovec value is **cross-project** recall for Hermes.
