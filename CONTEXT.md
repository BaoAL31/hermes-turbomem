# hermes-turbomem

A local, privacy-first persistent memory layer for Hermes and other MCP agents: semantic recall over learned experiences and indexed code symbols, across projects, without requiring the agent to be inside a project's working directory.

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
One indexed codebase, identified stably (e.g. git remote URL or canonical root path) so recall works even when the agent's current working directory is elsewhere.
_Avoid_: "Repo" (informal only), "workspace" (Hermes overloads this).

**Recall**:
A single semantic search over Persistent Memory, returning ranked Experiences and Code Entries, optionally filtered by project or type.
_Avoid_: Separate "codebase search" vs "memory search" as different product concepts (implementation may use one operation).

**Indexing**:
The process of scanning a Project's source, parsing semantic units, embedding them, and writing Code Entries into Persistent Memory.
_Avoid_: "Retain" for code (reserve for Hindsight-style fact extraction if added later).

**Auto-indexing** (optional):
When enabled, the first Recall or remember against a Project path triggers Indexing without an explicit `index_project` call. Default is off (manual Indexing only).

## Relationships

- A **Project** has many **Code Entries** (after **Indexing**) and may have many **Experiences** tagged with that Project (or none, for global facts).
- **Recall** searches across all Projects by default; filters can narrow to one **Project** or one entry kind (Experience vs Code Entry).
- **Indexing** is manual by default; **Auto-indexing** may trigger Indexing on first use when configured.
- An **Experience** may reference a **Project** but does not replace **Code Entries** for locating symbols.

## Example dialogue

> **Dev:** "We're in ~/notes, but where was `authenticate` in the API service?"
> **Domain expert:** "**Recall** globally — you should get a **Code Entry** for the API **Project** with path and line range, even though we're not in that tree."

## Flagged ambiguities

- Hermes **MEMORY.md** is a tiny curated prompt snapshot; **Persistent Memory** (this system) is a separate, searchable store. They complement each other; this project does not replace MEMORY.md.
- "Hindsight" names (retain/recall/reflect) inspired the design; v1 uses **Indexing** + `remember` for ingest and unified **Recall** for search (not a full Hindsight port).
